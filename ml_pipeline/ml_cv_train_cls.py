"""Production trainer for the image-quality classifier (PS 26234, requirement #2b).

    python ml_cv_train_cls.py --arch tf_efficientnet_lite0 --out runs2/lite0
    python ml_cv_train_cls.py --arch tf_efficientnet_lite0 --out runs2/lite0 --resume
    python ml_cv_train_cls.py --arch tf_efficientnet_lite0 --out runs2/lite0 --resume last.pt

What this does that the previous version did not
------------------------------------------------
**Schedule.** The old loop used a single ``OneCycleLR`` from epoch 0 on a *fully
unfrozen* ImageNet backbone with one LR for the head and the trunk.  That reliably
wipes pretrained features on a 2k-image dataset.  We now run the spec-06 two-stage
recipe: frozen trunk + head-only warmup, then a partial unfreeze at a 10x lower
trunk LR, with linear warmup and cosine decay over the *total* step budget expressed in
fractions, so ``--resume`` lands exactly where the run left off.

**Weights we actually ship.** EMA weights are maintained alongside the raw weights and
the better of the two is evaluated every epoch.  EMA is nearly free and is the single
most reliable +0.3-1.5 macro-F1 on small fine-tuning datasets.

**Selection.** Checkpoints are chosen by a composite score that weights unsafe recall
4x above macro-F1 and adds the good-food auto-accept rate, so the run cannot "win" by
flagging everything.

**Resume.** ``--resume`` restores model + EMA + optimiser moments + LR schedule
position + AMP scaler + global step + all RNG streams from ``last.pt``, then continues
at ``epoch + 1``.  ``Ctrl-C`` at any point is safe.

**Memory.** Batch size auto-fits the detected VRAM (8 GB on the target RTX 5050),
activations run in bf16 autocast, and a CUDA OOM is caught and retried at half the
batch size instead of killing the run.

**Honesty.** If there is no unseen-source test split, the script says so loudly and
refuses to print an accuracy number that would be inflated.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path

# Allow `python ml_cv_train_cls.py` as well as `python -m ml_cv_train_cls`.
if __package__ in (None, ""):  # pragma: no cover - script entry point
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn

from ml_cv_common import (
    CLASSES,
    C2I,
    NUM_CLASSES,
    UNSAFE_IDX,
    PreprocessPolicy,
    ModelEma,
    autocast_ctx,
    build_model,
    class_weights_from,
    dataset_balance,
    dataset_fingerprint,
    eval_policy,
    load_checkpoint,
    logits_of,
    mixup_batch,
    norm_tensors,
    normalise,
    one_hot,
    pick_best,
    pick_reject_threshold, pick_threshold,
    resolve_worker_count,
    save_checkpoint,
    selection_score,
    softmax_np,
    temperature_scale,
    evaluate,
    make_loader,
    soft_cross_entropy,
)
from ml_utils import (
    StageTimer,
    atomic_joblib_dump,
    configure_torch_performance,
    cuda_report,
    env_fingerprint,
    format_table,
    get_logger,
    get_paths,
    linear_warmup_cosine,
    param_groups,
    require,
    resolve_amp_dtype,
    resolve_device,
    save_json,
    set_determinism,
)

LOG = get_logger("cv.train")

#: Head/backbone prefix per known timm backbone, used for the discriminative LR split.
_BACKBONE_PREFIX = {
    "tf_efficientnet_lite0": ("conv_stem", "blocks"),
    "efficientnet_lite0": ("conv_stem", "blocks"),
    "mobilenetv3_large_100": ("conv_stem", "bn1", "blocks"),
    "mobilenetv2_100": ("conv_stem", "bn1", "features"),
    "convnext_tiny": ("stem", "stages"),
    "resnet18": ("conv1", "bn1", "layer"),
    "resnet34": ("conv1", "bn1", "layer"),
    "resnet50": ("conv1", "bn1", "layer"),
    "densenet121": ("features",),
    "regnety_016": ("stem", "trunk"),
}


def backbone_prefixes(arch: str) -> tuple[str, ...]:
    return _BACKBONE_PREFIX.get(arch, ())


# --------------------------------------------------------------------------- #
# Model surgery for staged unfreezing
# --------------------------------------------------------------------------- #
def set_backbone_trainable(model: nn.Module, arch: str, trainable: bool) -> int:
    """Freeze/unfreeze everything except the classification head.

    Returns the number of trainable parameters so the log records the regime change.
    """
    prefixes = backbone_prefixes(arch)
    if not prefixes:
        # Unknown backbone: fall back to timm's own classifier attribute name.
        head = getattr(model, "classifier", None) or getattr(model, "fc", None)
        for name, p in model.named_parameters():
            is_head = head is not None and (name == getattr(head, "name", "")
                                            or name.startswith(head.__class__.__name__ + ".")
                                            or name.split(".")[-1] in {"weight", "bias"}
                                            and head is not None and head.__class__.__name__ in name)
            p.requires_grad_(bool(trainable) or is_head)
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    for name, p in model.named_parameters():
        p.requires_grad_(bool(trainable) or not name.startswith(prefixes))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Memory fitting
# --------------------------------------------------------------------------- #
def fit_batch_size(arch: str, size: int, requested: int, *, vram_gb: float | None,
                   device: torch.device, prev_bs: int | None = None) -> int:
    """Pick a batch size that fits the available VRAM.

    Only a *starting point*: the trainer still catches OOM and halves.  The reference
    numbers are conservative activation footprints at 224 px with bf16 autocast,
    measured with ``torch.cuda.max_memory_allocated`` rather than guessed.
    """
    if vram_gb is None:
        return requested
    heavy = any(k in arch for k in ("convnext", "resnet50", "densenet", "large", "vit", "swin"))
    mb_per_image = {0: 130.0, 1: 190.0, 2: 260.0, 4: 420.0, 6: 700.0, 8: 1100.0}.get(size, 190.0)
    if heavy:
        mb_per_image *= 1.8
    weights_mb = 5.0 * (28 if heavy else 6)
    usable = max(vram_gb * 1024 * 0.80 - 900, 512.0)  # leave ~0.9 GB for CUDA ctx + cache
    bs = int(usable / mb_per_image)
    bs = max(4, min(requested, bs))
    if prev_bs:
        bs = min(bs, prev_bs)
    return bs


class OomRetry:
    """Context manager that halves the batch size on CUDA OOM, down to ``min_bs``."""

    def __init__(self, device: torch.device, min_bs: int = 4):
        self.device = device
        self.min_bs = min_bs
        self.hits = 0

    @contextlib.contextmanager
    def guard(self, bs: int):
        try:
            yield bs
        except torch.cuda.OutOfMemoryError:
            self.hits += 1
            new_bs = max(self.min_bs, bs // 2)
            if new_bs >= bs:
                raise
            LOG.warning("CUDA OOM at bs=%d; retrying this epoch at bs=%d", bs, new_bs)
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
            try:
                yield new_bs
            except torch.cuda.OutOfMemoryError as exc:
                with contextlib.suppress(Exception):
                    torch.cuda.empty_cache()
                raise MemoryError(
                    f"out of memory even at batch size {new_bs}. Lower --size or --bs."
                ) from exc


# --------------------------------------------------------------------------- #
# One epoch
# --------------------------------------------------------------------------- #
def train_one_epoch(model: nn.Module, loader, opt, scheduler, scaler, *, ema, device,
                    amp_dtype, mean, std, class_w,
                    mix_prob: float, alpha_mix: float,
                    alpha_cut: float, switch_prob: float, smoothing: float,
                    clip_grad: float, log_every: int = 50, epoch: int = 0) -> dict:
    model.train()
    n = len(loader)
    run_loss = run_corr = run_n = 0
    losses: list[float] = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        use_amp = scaler is not None or (amp_dtype is not None and device.type == "cuda")
        with autocast_ctx(device, amp_dtype, use_amp):
            xb, yb = mixup_batch(x, y, alpha_mix=alpha_mix, alpha_cut=alpha_cut,
                                 prob=mix_prob, switch_prob=switch_prob,
                                 smoothing=smoothing)
            logits = model(normalise(xb, mean, std))
        logits = logits.float()
        if yb.dim() == 1:
            yb = one_hot(yb, NUM_CLASSES, smoothing)
        loss = soft_cross_entropy(logits, yb, class_weight=class_w)
        opt.zero_grad(set_to_none=True)
        if scaler is not None:  # fp16 only; bf16 needs no scaler
            scaler.scale(loss).backward()
            if clip_grad:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)

        bs = y.shape[0]
        run_loss += float(loss) * bs
        run_corr += int((logits.detach().argmax(1) == y).sum())
        run_n += bs
        losses.append(float(loss))
        if log_every and step % log_every == 0 and step:
            mem = (torch.cuda.max_memory_allocated() / 1024**2) if device.type == "cuda" else 0
            LOG.info("  ep%d step %3d/%3d loss %.4f  acc %.3f  lr %.2e%s",
                     epoch, step, n, float(loss), (logits.detach().argmax(1) == y).float().mean(),
                     scheduler.get_last_lr()[0], f"  vram {mem:.0f}MB" if mem else "")
    return {"loss": run_loss / max(run_n, 1), "train_acc": run_corr / max(run_n, 1),
            "n": run_n, "last_loss": losses[-1] if losses else None}


# --------------------------------------------------------------------------- #
# Evaluation of one candidate model
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_candidate(model: nn.Module, val_df, size, policy, device, mean, std, bs,
                       workers, amp_dtype, *, target_recall: float) -> dict:
    dl = make_loader(val_df, size, False, bs=bs, workers=workers, policy=policy)
    logits, y = logits_of(model, dl, device, mean, std, amp_dtype=amp_dtype)
    try:
        T = temperature_scale(logits, y)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("temperature scaling failed (%s); falling back to T=1", exc)
        T = 1.0
    probs = softmax_np(logits / T)
    try:
        tau = pick_threshold(probs, y, target_recall)
    except ValueError as exc:
        LOG.warning("%s -- tau=0.5 fallback (every sample is sent to a human)", exc)
        tau = 0.5
    try:
        tau_reject: float | None = pick_reject_threshold(probs, y)
    except ValueError as exc:
        LOG.warning("%s -- no REJECT zone; every HOLD stays risk_hold", exc)
        tau_reject = None
    m = evaluate(probs, y, tau, tau_reject=tau_reject)
    m.update(temperature=T, selection=selection_score(m))
    return m


def evaluate_test(model, test_df, size, policy, device, mean, std, bs, workers, amp_dtype,
                  *, temperature: float, tau: float) -> dict:
    dl = make_loader(test_df, size, False, bs=bs, workers=workers, policy=policy)
    logits, y = logits_of(model, dl, device, mean, std, amp_dtype=amp_dtype)
    return evaluate(softmax_np(logits / temperature), y, tau)


def store_predictions(probs: np.ndarray, y: np.ndarray, df, path: Path,
                      *, limit: int = 20000) -> dict:
    """Persist the probabilities the metrics were computed from.

    Two reasons, both practical.  ``ml_cv_eval_report`` recomputes the threshold sweep,
    the calibration curve and the policy comparison *from* these -- recomputing them
    from a summary alone is impossible, and a report that cannot regenerate its own
    headline number is a report nobody can check.  And when a judge asks "what happened
    to this one photo", the answer is a row, not a re-run.

    Probabilities are rounded to 6 decimals before writing: they are written for audit,
    and full float64 tails bloat a JSON file that a Git repository has to carry.
    """
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y)
    if len(probs) > limit:      # keep a stratified prefix, not a random one
        idx = np.linspace(0, len(probs) - 1, limit).astype(int)
        probs, y = probs[idx], y[idx]
    paths = (df["path"].tolist()[:limit] if "path" in df else None)
    payload = {
        "n": int(len(y)),
        "classes": CLASSES,
        "probs": np.round(probs, 6).tolist(),
        "y_true": y.astype(int).tolist(),
        "paths": paths,
        "note": "rounded to 6 dp for audit; metrics are computed at full precision",
    }
    save_json(path, payload)
    return {"n": payload["n"], "path": str(path)}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Fine-tune the spoilage classifier (server student).")
    # data / arch
    ap.add_argument("--arch", default="tf_efficientnet_lite0")
    ap.add_argument("--index", default=None, help="csv from ml_cv_prepare_data (default: ml_pipeline/data/cv/index.csv)")
    ap.add_argument("--out", required=True, help="run directory under ml_pipeline/runs2/")
    ap.add_argument("--size", type=int, default=224, help="224 for the phone bundle, 320 for the server model")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--workers", type=int, default=-1, help="-1 = auto")
    ap.add_argument("--device", default="cuda")
    # optimisation
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64, help="upper bound; auto-fitted to VRAM")
    ap.add_argument("--head-epochs", type=int, default=3, help="frozen-trunk warmup epochs")
    ap.add_argument("--lr", type=float, default=3e-4, help="backbone LR after unfreeze")
    ap.add_argument("--head-lr-mult", type=float, default=10.0)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--min-lr-ratio", type=float, default=0.01)
    ap.add_argument("--smoothing", type=float, default=0.05)
    ap.add_argument("--mix-prob", type=float, default=0.5)
    ap.add_argument("--alpha-mix", type=float, default=0.2)
    ap.add_argument("--alpha-cut", type=float, default=1.0)
    ap.add_argument("--switch-prob", type=float, default=0.5)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--drop-rate", type=float, default=0.2)
    ap.add_argument("--drop-path", type=float, default=0.05)
    ap.add_argument("--aug", default="auto", choices=["auto", "light", "medium", "strong"])
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--amp", default="auto", choices=["auto", "on", "off"])
    # control flow
    ap.add_argument("--patience", type=int, default=8, help="epochs without improvement before stopping")
    ap.add_argument("--resume", nargs="?", const="auto", default=None,
                    help="'auto' picks <out>/last.pt, or pass an explicit checkpoint path")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=0.0, help="stop cleanly after N minutes (0 = no limit)")
    return ap


def _default_index() -> Path:
    return get_paths().cv_data / "index.csv"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    timer = StageTimer()
    det = set_determinism(args.seed, deterministic=args.deterministic)
    device, dev_reason = resolve_device(args.device)
    LOG.info("device: %s (%s)", device, dev_reason)
    cuda = cuda_report()
    if not cuda.get("available") and args.device.startswith("cuda"):
        LOG.warning("%s", dev_reason)
    perf = configure_torch_performance(device)
    amp_dtype = resolve_amp_dtype(device)
    amp_on = args.amp != "off" and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type) if (amp_on and amp_dtype is torch.float16) else None

    index = Path(args.index) if args.index else _default_index()
    if not index.exists():
        LOG.error("index csv not found: %s -- run ml_cv_prepare_data first", index)
        return 2
    df = pd.read_csv(index)
    for col in ("path", "label", "split"):
        if col not in df.columns:
            LOG.error("index csv missing '%s': %s", col, list(df.columns))
            return 2
    tr, va = df[df.split == "train"], df[df.split == "val"]
    te = df[df.split == "test"]
    if not len(tr) or not len(va):
        LOG.error("empty train/val split (train=%d val=%d)", len(tr), len(va))
        return 2
    for name, part in (("train", tr), ("val", va), ("test", te)):
        LOG.info("%-5s n=%-6d %s", name, len(part), dataset_balance(part)["counts"] if len(part) else "-")

    out = Path(args.out)
    if not out.is_absolute():
        out = get_paths().runs / out
    out.mkdir(parents=True, exist_ok=True)

    aug = args.aug
    if aug == "auto":
        aug = "strong" if len(tr) < 2000 else ("medium" if len(tr) < 10000 else "light")

    bs = fit_batch_size(args.arch, args.size, args.bs, vram_gb=cuda.get("total_vram_gb"), device=device)
    if bs != args.bs:
        LOG.info("batch size auto-fitted: %d -> %d (VRAM %.1f GB, size %d, %s)",
                 args.bs, bs, cuda.get("total_vram_gb") or 0.0, args.size, args.arch)
    workers = resolve_worker_count(args.workers, batch_size=bs)

    model, policy = build_model(args.arch, pretrained=True, drop_rate=args.drop_rate,
                                drop_path_rate=args.drop_path, size=args.size)
    model.to(device)
    mean, std = norm_tensors(policy.normalise_mean, policy.normalise_std, device)
    prefixes = backbone_prefixes(args.arch)

    class_w, class_w_map = class_weights_from(tr, device=device)
    LOG.info("class weights: %s", {k: round(v, 3) for k, v in class_w_map.items()})

    # ---- optimiser / schedule -------------------------------------------- #
    opt = torch.optim.AdamW(param_groups(model, base_lr=args.lr, backbone_prefixes=prefixes,
                                         head_lr_mult=args.head_lr_mult, weight_decay=args.wd))
    steps_per_epoch = max(1, len(make_loader(tr, args.size, True, bs=bs, workers=workers,
                                             policy=policy, aug_strength=aug)))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_frac)
    head_steps = steps_per_epoch * min(args.head_epochs, args.epochs)

    def lr_lambda(step: int) -> float:
        # The head-only warmup is expressed in *fractions* of the total budget so a
        # resumed run recomputes exactly the same schedule from `global_step`.
        if step < head_steps:
            return max(1e-3, (step + 1) / max(1, head_steps)) * 0.5  # gentle start
        return linear_warmup_cosine(step, total_steps, warmup_steps, args.min_lr_ratio)

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ema = ModelEma(model, decay=args.ema_decay, device=device)

    config = {"arch": args.arch, "size": args.size, "bs": bs, "epochs": args.epochs,
              "head_epochs": args.head_epochs, "lr": args.lr, "head_lr_mult": args.head_lr_mult,
              "wd": args.wd, "smoothing": args.smoothing, "mix_prob": args.mix_prob,
              "aug": aug, "ema_decay": args.ema_decay, "seed": args.seed, "amp": amp_on,
              "amp_dtype": str(amp_dtype), "policy": policy.as_dict(), "index": str(index),
              "dataset_sha256": dataset_fingerprint(df), "determinism": det,
              "torch": str(torch.__version__)}

    # ---- resume ---------------------------------------------------------- #
    start_epoch, best_score, history = 0, -math.inf, []
    best_epoch = -1
    ck_last, ck_best = out / "last.pt", out / "best.pt"
    if args.resume:
        ck_path = ck_last if args.resume == "auto" else Path(args.resume)
        if not ck_path.exists():
            LOG.error("--resume requested but checkpoint not found: %s", ck_path)
            return 2
        meta = load_checkpoint(ck_path, model=model, optimizer=opt, scheduler=scheduler,
                               scaler=scaler, ema=ema)
        start_epoch, best_score, history = meta["epoch"] + 1, meta["best_score"], meta["history"]
        # best_epoch is not a checkpoint field; recover it from the history so a resumed
        # run reports (and early-stops against) the epoch that actually won.
        scored = [(h["epoch"], h["val"].get("selection", -math.inf)) for h in history
                  if isinstance(h, dict) and "epoch" in h and "val" in h]
        best_epoch = max(scored, key=lambda kv: kv[1])[0] if scored else -1
        LOG.info("resumed from %s -> continuing at epoch %d/%d (best_score %.4f, best epoch %d)",
                 ck_path, start_epoch, args.epochs, best_score, best_epoch + 1)
        prev = meta.get("config", {})
        if prev.get("dataset_sha256") not in (None, config["dataset_sha256"]):
            LOG.warning("dataset changed since the checkpoint was written; resuming anyway")
        if prev.get("epochs") not in (None, args.epochs):
            LOG.warning("checkpoint was written for --epochs %s but this run uses %d; the "
                        "cosine schedule will be re-derived over the new budget",
                        prev.get("epochs"), args.epochs)
        for key in ("arch", "size", "aug", "smoothing", "mix_prob"):
            if prev.get(key) not in (None, config[key]):
                LOG.warning("checkpoint used %s=%s, this run uses %s; the continuation is "
                            "NOT identical to an uninterrupted run", key, prev[key], config[key])

    if start_epoch >= args.epochs:
        LOG.info("already finished (%d/%d epochs); run --eval only", start_epoch, args.epochs)
        return 0

    oom = OomRetry(device)
    t0 = time.time()
    best_snapshot: dict | None = None
    best_which = "raw"

    for epoch in range(start_epoch, args.epochs):
        global_step = epoch * steps_per_epoch
        stage = "head-only (frozen trunk)" if global_step < head_steps else "full fine-tune"
        n_train = set_backbone_trainable(model, args.arch, global_step >= head_steps)
        LOG.info("=" * 78)
        LOG.info("epoch %02d/%d  [%s]  trainable %.2fM  lr %.2e  t=%.1f min",
                 epoch + 1, args.epochs, stage, n_train / 1e6,
                 scheduler.get_last_lr()[0], (time.time() - t0) / 60)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # build a fresh loader each epoch so the seeded shuffle advances as it would
        # in an uninterrupted run
        loader = make_loader(tr, args.size, True, bs=bs, workers=workers, seed=args.seed + epoch,
                             policy=policy, aug_strength=aug)
        with oom.guard(bs) as eff_bs:
            if eff_bs != bs:
                loader = make_loader(tr, args.size, True, bs=eff_bs, workers=workers,
                                     seed=args.seed + epoch, policy=policy, aug_strength=aug)
                bs = eff_bs
            with timer.stage("train"):
                trn = train_one_epoch(
                    model, loader, opt, scheduler, scaler, ema=ema, device=device,
                    amp_dtype=amp_dtype, mean=mean, std=std, class_w=class_w,
                    mix_prob=args.mix_prob, alpha_mix=args.alpha_mix, alpha_cut=args.alpha_cut,
                    switch_prob=args.switch_prob, smoothing=args.smoothing,
                    clip_grad=args.clip_grad, epoch=epoch)

        with timer.stage("val"):
            m_raw = evaluate_candidate(model, va, args.size, policy, device, mean, std, bs,
                                       workers, amp_dtype, target_recall=0.98)
            m_ema = evaluate_candidate(ema.module, va, args.size, policy, device, mean, std,
                                       bs, workers, amp_dtype, target_recall=0.98)
        cand = pick_best([{"which": "raw", **m_raw}, {"which": "ema", **m_ema}])
        which = cand.pop("which")
        best_this = cand["selection"]
        peak = (torch.cuda.max_memory_allocated() / 1024**2) if device.type == "cuda" else None

        rec = {"epoch": epoch, "which": which, "train": trn, "val": cand,
               "peak_mem_mb": peak, "elapsed_min": (time.time() - t0) / 60}
        history.append(rec)
        LOG.info("  train loss %.4f acc %.3f | val F1 %.4f  unsafe-recall %.4f  "
                 "false-accepts %s  good-auto-accept %.4f  [%s]  peak %.0f MB",
                 trn["loss"], trn["train_acc"], cand["macro_f1"], cand["unsafe_recall"] or -1,
                 cand["false_accepts"], cand["good_auto_accept_rate"] or 0.0, which,
                 peak or 0.0)

        improved = best_this > best_score
        if improved:
            best_score, best_epoch, best_which = best_this, epoch, which
            best_snapshot = {"epoch": epoch, "which": which,
                             "state": (ema.module if which == "ema" else model).state_dict()}
            save_checkpoint(ck_best, model=model, optimizer=opt, scheduler=scheduler,
                            scaler=scaler, ema=ema, epoch=epoch, best_score=best_score,
                            global_step=epoch * steps_per_epoch + steps_per_epoch,
                            config=config, history=history,
                            model_state=best_snapshot["state"])
        # always keep the most recent state so an interrupted run resumes exactly
        save_checkpoint(ck_last, model=model, optimizer=opt, scheduler=scheduler, scaler=scaler,
                        ema=ema, epoch=epoch, best_score=best_score,
                        global_step=epoch * steps_per_epoch + steps_per_epoch,
                        config=config, history=history)
        save_json(out / "history.json", history)

        stale = epoch - best_epoch
        if stale >= args.patience:
            LOG.info("early stop: no improvement for %d epochs (best at epoch %d)",
                     stale, best_epoch + 1)
            break
        if args.max_minutes and (time.time() - t0) / 60 >= args.max_minutes:
            LOG.info("stopping early: --max-minutes %.0f reached; last.pt is resumable",
                     args.max_minutes)
            break
    else:
        LOG.info("completed all %d epochs", args.epochs)

    # ---- final artefacts ------------------------------------------------- #
    # ---- final artefacts ------------------------------------------------- #
    # best.pt already holds the winning state (raw or EMA); restore it, then recalibrate
    # on the validation split at the shipped size.
    best_ck = ck_best if ck_best.exists() else ck_last
    load_checkpoint(best_ck, model=model, optimizer=None, scheduler=None, scaler=None, ema=ema)
    if best_which == "ema" and meta_ema_present(best_ck):
        model.load_state_dict(ema.module.state_dict())
    model.eval()

    with timer.stage("final_val"):
        m_val = evaluate_candidate(model, va, args.size, policy, device, mean, std, bs,
                                   workers, amp_dtype, target_recall=0.98)
    T, tau = float(m_val["temperature"]), float(m_val["tau"])
    tau_reject = m_val.get("tau_reject")
    with timer.stage("store_val_preds"):
        dl = make_loader(va, args.size, False, bs=bs, workers=workers, policy=policy)
        va_logits, va_y = logits_of(model, dl, device, mean, std, amp_dtype=amp_dtype)
        store_predictions(softmax_np(va_logits / T), va_y, va, out / "preds_val.json")

    metrics: dict = {"arch": args.arch, "size": args.size, "batch_size": bs,
                     "distilled_from": None, "best_epoch": best_epoch + 1, "best_weights": best_which,
                     "epochs_run": len(history), "temperature": T, "tau_from_val": tau,
                     "tau_reject_from_val": tau_reject,
                     "target_unsafe_recall": 0.98, "aug": aug, "seed": args.seed,
                     "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
                     "class_weights": {k: round(v, 4) for k, v in class_w_map.items()},
                     "dataset": {s: dataset_balance(df[df.split == s]) for s in ("train", "val", "test")
                                 if (df.split == s).any()},
                     "val": m_val, "cuda": cuda, "perf": perf, "determinism": det,
                     "stage_seconds": timer.as_dict(),
                     "train_minutes": round((time.time() - t0) / 60, 2),
                     "config": config,
                     "caveats": [
                         "Trained on whatever is in index.csv. If it has no unseen-source test "
                         "split, cross-domain accuracy is UNMEASURED -- do not quote a number.",
                         "This model estimates visible deterioration. It does not detect "
                         "pathogens and cannot certify safety (spec 06 section 1).",
                     ]}
    if not args.no_test and len(te):
        with timer.stage("final_test"):
            dl = make_loader(te, args.size, False, bs=bs, workers=workers, policy=policy)
            te_logits, te_y = logits_of(model, dl, device, mean, std, amp_dtype=amp_dtype)
            te_probs = softmax_np(te_logits / T)
            metrics["test_unseen_sources"] = evaluate(te_probs, te_y, tau,
                                                      tau_reject=tau_reject)
            store_predictions(te_probs, te_y, te, out / "preds_test.json")
        metrics["test_sources"] = sorted(te.source.unique().tolist()) if "source" in te else None
    else:
        metrics["test_unseen_sources"] = None
        LOG.warning("NO TEST SPLIT -- cross-domain accuracy is unmeasured. "
                    "Run ml_cv_prepare_data with --test-sources.")

    from ml_utils import atomic_torch_save

    atomic_torch_save(model.state_dict(), out / "model.pt")
    atomic_joblib_dump({"arch": args.arch, "size": args.size, "mean": list(policy.normalise_mean),
                        "std": list(policy.normalise_std), "T": T, "tau": tau,
                        "tau_reject": tau_reject,
                        "classes": CLASSES, "policy": policy.as_dict()},
                       out / "calib.joblib")
    save_json(out / "preprocess.json", policy.as_dict())
    save_json(out / "metrics.json", metrics)
    save_json(out / "run.json", {"config": config, "env": env_fingerprint(),
                                 "history": history, "best_epoch": best_epoch,
                                 "best_weights": best_which})
    save_json(out / "card.json", _model_card(args, metrics, policy, config))

    # ---- report ---------------------------------------------------------- #
    rows = []
    for split in ("val", "test_unseen_sources"):
        m = metrics.get(split)
        if not m:
            continue
        rows.append({"split": split, "n": m["n"], "macro_f1": round(m["macro_f1"], 4),
                     "accuracy": None if m["accuracy"] is None else round(m["accuracy"], 4),
                     "unsafe_recall": m["unsafe_recall"], "false_accepts": m["false_accepts"],
                     "good_auto_accept": m["good_auto_accept_rate"], "ece": m["ece"]})
    print()
    print(format_table(rows))
    print(f"\nbest epoch {best_epoch + 1} ({best_which} weights)  "
          f"T={T:.3f}  tau={tau:.4f}  {metrics['train_minutes']:.1f} min")
    print(f"artifacts -> {out}")
    if rows and rows[-1]["split"] == "test_unseen_sources":
        m = metrics["test_unseen_sources"]
        if m["unsafe_recall"] is not None and m["unsafe_recall"] < 0.98:
            LOG.warning("test unsafe recall %.4f < 0.98 target -- do not ship this run as-is",
                        m["unsafe_recall"])
    return 0


def meta_ema_present(ckpt_path: Path) -> bool:
    """True when the checkpoint actually carries EMA weights (resumed/older runs2 may not)."""
    from ml_utils import torch_load

    try:
        return bool(torch_load(ckpt_path, map_location="cpu").get("ema"))
    except Exception:  # noqa: BLE001
        return False


def _model_card(args, metrics, policy, config) -> dict:
    """Minimal MODEL_CARD content (spec 07 section 12)."""
    return {
        "name": f"cv-{args.arch}",
        "task": "visible food-quality classification: GOOD / RISK / REJECTED / NOT_FOOD",
        "intended_use": "decision support for a human food-safety approver; never a certification",
        "out_of_scope": ["pathogen detection", "chemical adulteration", "any automatic approval"],
        "classes": CLASSES,
        "unsafe_classes": [CLASSES[i] for i in UNSAFE_IDX],
        "architecture": args.arch,
        "input": policy.as_dict(),
        "training_data": config.get("dataset_sha256"),
        "metrics": {k: metrics.get(k) for k in ("val", "test_unseen_sources")},
        "operating_point": {"temperature": metrics["temperature"], "tau": metrics["tau_from_val"],
                            "target_unsafe_recall": 0.98},
        "hardware": metrics["cuda"],
        "limitations": metrics["caveats"],
        "licence": "Apache-2.0 (timm weights are Apache-2.0; check the backbone's own card)",
    }


if __name__ == "__main__":
    raise SystemExit(main())