"""Teacher candidate: frozen DINOv2-small features + a trained linear head.

    python ml_cv_dino_probe.py --index data/cv/index.csv --out runs2/dino_probe

A frozen self-supervised backbone with a linear head is the right first model on a
*few* labelled photos: the backbone has no gradients to tune, so the head converges in
seconds and its failure modes are legible.  If a linear probe on DINOv2 cannot separate
mould from fresh paneer, no amount of fine-tuning on the same photos will.

What this script guarantees (all of it survives a Ctrl+C):

* **Pause and continue.**  Feature extraction is cached to ``features_cache/`` per
  split and flushed every few batches; head training checkpoints every ``--log-every``
  steps and at every epoch end.  Re-run the *same command* and it resumes exactly
  where it stopped -- the cache and checkpoint carry a fingerprint (index bytes +
  backbone + size + seed) and are discarded loudly if anything changed.
* **Live terminal telemetry.**  Extraction prints ``batch i/N | img/s | ETA | VRAM``
  in place, every batch.  Head training prints ``step/total | loss | lr | run-acc``
  every ``--log-every`` steps (default 400) and a full validation evaluation at every
  epoch end.
* **GPU health on small cards (RTX 5050 8 GB, 75 W).**  The backbone is frozen and in
  eval mode (no optimizer state, no gradients through 22M params), autocast runs2 bf16
  when the card supports it, dataloader workers are clamped for a laptop, VRAM is
  reported on every progress line, and any CUDA OOM halves the batch and continues
  instead of dying.
* **A stronger head than a plain fit.**  AdamW + cosine schedule with warmup, class
  balancing, label smoothing, early stopping on val macro-F1 with the best weights
  restored, and absent classes pinned to a strongly negative bias (softmax(-30) ~
  1e-13: "never seen" means what it says).

The probe is a *teacher* candidate and an export source (``ml_cv_export.py --teacher``).
It is fp32 on the server: a ViT at int8 loses more accuracy than the CNN student does,
and the student is what ships to the phone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from ml_cv_common import (
    CLASSES,
    LOG,
    PreprocessPolicy,
    autocast_ctx,
    evaluate,
    make_loader,
    norm_tensors,
    pick_reject_threshold,
    pick_threshold,
    resolve_worker_count,
    softmax_np,
    temperature_scale,
)
from ml_utils import atomic_joblib_dump, atomic_torch_save, cuda_report, resolve_device, save_json

TEACHER = "vit_small_patch14_dinov2.lvd142m"
#: Logit assigned to a class the training set never contained.  softmax(30) gap is
#: e^-30 ~ 1e-13, i.e. numerically absent, which is the honest encoding of "unknown".
ABSENT_CLASS_BIAS = -30.0

#: Batches between feature-cache flushes -- the most work a crash can lose.
CACHE_FLUSH_EVERY = 25


def _fingerprint(index_path: Path, *, backbone: str, size: int, seed: int, kind: str) -> str:
    """Cache/checkpoint validity: change anything and every cache is discarded."""
    h = hashlib.sha256()
    h.update(index_path.read_bytes())
    h.update(f"|{backbone}|{size}|{seed}|{kind}".encode())
    return h.hexdigest()[:16]


def _flush_cache(cache_dir: Path, split: str, feats: np.ndarray, labels: np.ndarray,
                 done: set, fingerprint: str) -> None:
    np.save(cache_dir / f"features_{split}.npy", feats)
    np.save(cache_dir / f"labels_{split}.npy", labels)
    save_json(cache_dir / f"done_{split}.json",
              {"fingerprint": fingerprint, "done": sorted(done)})


@torch.no_grad()
def extract_resumable(model, part: pd.DataFrame, split: str, cache_dir: Path, *,
                      size: int, dev, mean, std, bs: int, workers: int,
                      amp_dtype, fingerprint: str, fresh: bool = False):
    """Extract frozen-backbone features for one split, resume-safe and OOM-safe.

    Rows are scattered by the loader's returned index, so a partial cache is always
    consistent no matter how the run was interrupted.  A CUDA OOM halves the batch and
    continues (rows of the failed batch were never scattered, so nothing is corrupt).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    n, dim = len(part), int(model.num_features)
    feats = np.zeros((n, dim), np.float32)
    labels = np.zeros(n, np.int64)
    f_path = cache_dir / f"features_{split}.npy"
    y_path = cache_dir / f"labels_{split}.npy"
    d_path = cache_dir / f"done_{split}.json"
    done: set = set()
    if not fresh and d_path.exists():
        meta = json.loads(d_path.read_text(encoding="utf-8"))
        if (meta.get("fingerprint") == fingerprint and f_path.exists() and y_path.exists()
                and np.load(f_path, mmap_mode="r").shape == (n, dim)):
            done = set(meta["done"])
            feats = np.load(f_path)
            labels = np.load(y_path)
            LOG.info("%s: resuming feature cache -- %d/%d rows already done",
                     split, len(done), n)
        else:
            LOG.warning("%s: cached features are stale (fingerprint/shape mismatch) -- "
                        "re-extracting this split", split)
    remaining = np.array([i for i in range(n) if i not in done], dtype=np.int64)

    total_vram = (torch.cuda.get_device_properties(dev.index or 0).total_memory / 1e9
                  if dev.type == "cuda" else 0.0)
    t_start, n_start = time.time(), len(remaining)
    while len(remaining):
        sub = part.iloc[remaining]
        dl = make_loader(sub, size, train=False, bs=bs,
                         workers=resolve_worker_count(workers, batch_size=bs),
                         return_index=True)
        n_batches = math.ceil(len(sub) / bs)
        t0, i_batch, n_run = time.time(), 0, 0
        try:
            for x, y, idx in dl:
                orig = remaining[idx.numpy()]
                with autocast_ctx(dev, amp_dtype, enabled=amp_dtype is not None):
                    f = model((x.to(dev, non_blocking=True) - mean) / std)
                feats[orig] = f.float().cpu().numpy()
                labels[orig] = y.numpy()
                done.update(orig.tolist())
                n_run += len(orig)
                i_batch += 1
                rate = n_run / max(time.time() - t0, 1e-6)
                eta = (n_start - n_run) / max(rate, 1e-6)
                vram = (torch.cuda.memory_allocated(dev) / 1e9) if dev.type == "cuda" else 0.0
                print(f"\r  {split}: batch {i_batch}/{n_batches} | {n_run}/{n_start} imgs "
                      f"| {rate:6.0f} img/s | ETA {int(eta // 60):02d}:{int(eta % 60):02d}"
                      + (f" | VRAM {vram:.1f}/{total_vram:.0f} GB" if dev.type == "cuda" else "")
                      + "   ", end="", flush=True)
                if i_batch % CACHE_FLUSH_EVERY == 0:
                    _flush_cache(cache_dir, split, feats, labels, done, fingerprint)
        except torch.cuda.OutOfMemoryError:
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            if bs <= 8:
                raise
            bs = max(8, bs // 2)
            LOG.warning("%s: CUDA OOM -- retrying the remaining rows with bs=%d", split, bs)
            remaining = np.array([i for i in range(n) if i not in done], dtype=np.int64)
            continue
        except KeyboardInterrupt:
            print()
            _flush_cache(cache_dir, split, feats, labels, done, fingerprint)
            LOG.info("PAUSED at %d/%d %s rows. Run the same command again to resume.",
                     len(done), n, split)
            raise SystemExit(130)
        remaining = np.array([i for i in range(n) if i not in done], dtype=np.int64)
    print()
    _flush_cache(cache_dir, split, feats, labels, done, fingerprint)
    LOG.info("%s: %d features %s  labels %s  (%.0fs)", split, n, feats.shape, labels.shape,
             time.time() - t_start)
    return feats, labels


# --------------------------------------------------------------------------- #
# Stage B -- the head.  Two paths: a trained torch head (default) and the
# original sklearn logistic-regression grid (kept for reproducibility).
# --------------------------------------------------------------------------- #
def fit_head(xtr, ytr, xva, yva, *, n_classes: int, seed: int = 42) -> tuple:
    """Logistic regression over a small C grid; returns ``(W, b, best_C, scores)``."""
    scores = []
    best = None
    for C in (0.01, 0.1, 1.0, 10.0, 100.0):
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced",
                                 random_state=seed).fit(xtr, ytr)
        f1 = float(f1_score(yva, clf.predict(xva), average="macro", zero_division=0))
        scores.append({"C": C, "val_macro_f1": round(f1, 4)})
        if best is None or f1 > best[0]:
            best = (f1, C, clf)
    _, C, clf = best
    W = np.zeros((n_classes, xtr.shape[1]), np.float32)
    b = np.full(n_classes, ABSENT_CLASS_BIAS, np.float32)
    W[clf.classes_] = clf.coef_.astype(np.float32)
    b[clf.classes_] = clf.intercept_.astype(np.float32)
    missing = sorted(set(range(n_classes)) - set(int(c) for c in clf.classes_))
    if missing:
        LOG.warning("classes absent from the training set: %s -- their rows are pinned to "
                    "bias %.0f so they can never win the argmax",
                    [CLASSES[i] for i in missing], ABSENT_CLASS_BIAS)
    return W, b, C, scores


def _pin_absent_classes(head: nn.Linear, ytr: torch.Tensor) -> list:
    """Zero the rows of classes the training set never contained; bias them to -30."""
    present = set(int(c) for c in torch.unique(ytr).cpu().tolist())
    missing = sorted(set(range(len(CLASSES))) - present)
    with torch.no_grad():
        for c in missing:
            head.weight[c].zero_()
            head.bias[c] = ABSENT_CLASS_BIAS
    if missing:
        LOG.warning("classes absent from the training set: %s -- their rows are pinned to "
                    "bias %.0f so they can never win the argmax",
                    [CLASSES[i] for i in missing], ABSENT_CLASS_BIAS)
    return missing


def _save_head_ckpt(ckpt_path: Path, fingerprint: str, head, opt, sched, *,
                    step: int, next_epoch: int, best_f1: float, patience_ctr: int,
                    best_state, history: list) -> None:
    atomic_torch_save({"fingerprint": fingerprint, "head": head.state_dict(),
                       "opt": opt.state_dict(), "sched": sched.state_dict(),
                       "step": step, "epoch": next_epoch, "best_f1": best_f1,
                       "patience_ctr": patience_ctr, "best_state": best_state,
                       "history": history}, ckpt_path)


def train_head_torch(xtr, ytr, xva, yva, *, n_classes: int, epochs: int, patience: int,
                     min_epochs: int, lr: float, bs: int, log_every: int, seed: int,
                     ckpt_path: Path, fingerprint: str):
    """Train the linear head with checkpoint/resume, live telemetry, early stopping.

    Every ``log_every`` steps a progress line is printed and the checkpoint is saved
    (that is the pause point); every epoch ends with a full validation evaluation.
    Early stopping watches val macro-F1 and the **best** weights are what ships.
    """
    n, dim = xtr.shape
    head = nn.Linear(dim, n_classes).to(xtr.device)

    # class balancing: inverse-frequency weights over the classes that exist
    counts = torch.bincount(ytr, minlength=n_classes).float()
    w = torch.zeros(n_classes, device=xtr.device)
    present = counts > 0
    w[present] = counts.sum() / (present.sum() * counts[present])

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    steps_per_epoch = math.ceil(n / bs)
    total_steps = epochs * steps_per_epoch
    warmup = max(50, int(0.03 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        (s + 1) / warmup if s < warmup else
        0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warmup) /
                                                        max(1, total_steps - warmup))))))

    step, start_epoch, best_f1, patience_ctr = 0, 0, -1.0, 0
    best_state = None
    history: list = []
    if ckpt_path.exists():
        try:
            ck = torch.load(ckpt_path, map_location=xtr.device, weights_only=False)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("checkpoint unreadable (%s) -- starting head training fresh", exc)
            ck = None
        if ck and ck.get("fingerprint") == fingerprint:
            head.load_state_dict(ck["head"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            step, start_epoch = ck["step"], ck["epoch"]
            best_f1, patience_ctr = ck["best_f1"], ck["patience_ctr"]
            best_state, history = ck["best_state"], ck.get("history", [])
            LOG.info("RESUMED head training: epoch %d/%d, step %d/%d (best F1 %.4f)",
                     start_epoch + 1, epochs, step, total_steps, max(best_f1, 0))
        elif ck:
            LOG.warning("head checkpoint is stale (fingerprint mismatch) -- training fresh")

    epoch = start_epoch - 1          # bound even if the loop never runs2 (Ctrl+C early)
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    try:
        for epoch in range(start_epoch, epochs):
            head.train()
            perm = torch.randperm(n, generator=gen).to(xtr.device)
            run_loss = run_correct = run_n = 0
            for s in range(steps_per_epoch):
                idx = perm[s * bs:(s + 1) * bs]
                opt.zero_grad(set_to_none=True)
                logits = head(xtr[idx])
                loss = F.cross_entropy(logits, ytr[idx], weight=w, label_smoothing=0.05)
                loss.backward()
                opt.step()
                sched.step()
                step += 1
                run_loss += float(loss.detach()) * len(idx)
                run_correct += int((logits.argmax(1) == ytr[idx]).sum())
                run_n += len(idx)
                if step % log_every == 0 or step == total_steps:
                    lr_now = sched.get_last_lr()[0]
                    rate = run_n / max(time.time() - t0, 1e-6)
                    vram = (f" | VRAM {torch.cuda.memory_allocated(xtr.device) / 1e9:.1f} GB"
                            if xtr.device.type == "cuda" else "")
                    print(f"  step {step}/{total_steps} "
                          f"| loss {run_loss / max(run_n, 1):.4f} | lr {lr_now:.2e} "
                          f"| train acc {run_correct / max(run_n, 1):.4f} "
                          f"| {rate:6.0f} feat/s{vram}", flush=True)
                    _save_head_ckpt(ckpt_path, fingerprint, head, opt, sched,
                                    step=step, next_epoch=epoch + 1, best_f1=best_f1,
                                    patience_ctr=patience_ctr, best_state=best_state,
                                    history=history)

            # ---- every epoch ends with a full validation evaluation ---------- #
            head.eval()
            with torch.no_grad():
                v_logits = head(xva)
                v_loss = float(F.cross_entropy(v_logits, yva, weight=w,
                                               label_smoothing=0.05))
                v_pred = v_logits.argmax(1).cpu().numpy()
            yva_np = yva.cpu().numpy()
            v_f1 = float(f1_score(yva_np, v_pred, average="macro", zero_division=0))
            v_acc = float((v_pred == yva_np).mean())
            if v_f1 > best_f1:
                best_f1, patience_ctr = v_f1, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in head.state_dict().items()}
                marker = "  <- new best"
            else:
                patience_ctr += 1
                marker = f"  (patience {patience_ctr}/{patience})"
            history.append({"epoch": epoch + 1,
                            "train_loss": round(run_loss / max(run_n, 1), 5),
                            "train_acc": round(run_correct / max(run_n, 1), 4),
                            "val_loss": round(v_loss, 5),
                            "val_macro_f1": round(v_f1, 5), "val_acc": round(v_acc, 4)})
            print(f"  EPOCH {epoch + 1}/{epochs} done in {time.time() - t0:.0f}s "
                  f"| train_loss {run_loss / max(run_n, 1):.4f} "
                  f"| lr {sched.get_last_lr()[0]:.2e}")
            print(f"    VAL epoch end: macro_f1 {v_f1:.4f} | acc {v_acc:.4f} "
                  f"| val_loss {v_loss:.4f} | best {best_f1:.4f}{marker}", flush=True)
            t0 = time.time()
            _save_head_ckpt(ckpt_path, fingerprint, head, opt, sched,
                            step=step, next_epoch=epoch + 1, best_f1=best_f1,
                            patience_ctr=patience_ctr, best_state=best_state,
                            history=history)
            if epoch + 1 >= min_epochs and patience_ctr >= patience:
                LOG.info("early stopping after epoch %d -- no val improvement for %d "
                         "epochs (best val macro-F1 %.4f)", epoch + 1, patience, best_f1)
                break
    except KeyboardInterrupt:
        print()
        _save_head_ckpt(ckpt_path, fingerprint, head, opt, sched,
                        step=step, next_epoch=epoch + 1, best_f1=best_f1,
                        patience_ctr=patience_ctr, best_state=best_state, history=history)
        LOG.info("PAUSED mid-epoch %d at step %d. Run the same command again to resume "
                 "head training from epoch %d.", epoch + 1, step, epoch + 2)
        raise SystemExit(130)

    if best_state is not None:
        head.load_state_dict(best_state)
        head.eval()
    info = {"epochs_run": len(history), "steps": step, "lr": lr, "head_bs": bs,
            "warmup_steps": warmup, "label_smoothing": 0.05, "weight_decay": 1e-2,
            "best_val_macro_f1": round(best_f1, 5) if best_f1 > 0 else None,
            "early_stopped": bool(history) and len(history) < epochs
            and patience_ctr >= patience,
            "resumed_from_epoch": start_epoch + 1 if start_epoch else None}
    return head, history, info


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=None)
    ap.add_argument("--out", default="dino_probe")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--target-recall", type=float, default=0.98)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32,
                    help="feature-extraction batch; auto-halves on CUDA OOM")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--head", choices=("torch", "sklearn"), default="torch",
                    help="torch head = trained (AdamW, cosine, early stop); "
                         "sklearn = the original logistic-regression C grid")
    ap.add_argument("--epochs", type=int, default=30,
                    help="head-training epoch cap; early stopping usually fires first")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--min-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--head-bs", type=int, default=256)
    ap.add_argument("--log-every", type=int, default=400,
                    help="progress line + checkpoint every N head-training steps")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore the feature cache and any checkpoint; start over")
    a = ap.parse_args(argv)

    from ml_utils import get_paths, resolve_amp_dtype

    t0 = time.time()
    paths = get_paths()
    index = Path(a.index) if a.index else paths.cv_data / "index.csv"
    out = paths.runs / a.out if not Path(a.out).is_absolute() else Path(a.out)
    if not index.exists():
        raise SystemExit(f"no index at {index}; run ml_cv_prepare_data.py first")

    dev, dev_name = resolve_device(a.device)
    amp_dtype = resolve_amp_dtype(dev)
    if dev_name == "cpu":
        LOG.warning("running on CPU: a ViT-S/14 over a few thousand photos takes minutes, "
                    "not seconds. That is fine for a smoke test.")
    LOG.info("device %s | amp %s | %s", dev_name, amp_dtype, cuda_report().get("name", ""))

    fp = _fingerprint(index, backbone=TEACHER, size=a.size, seed=a.seed, kind=a.head)
    cache_dir = out / "features_cache"
    ckpt_path = out / "head_checkpoint.pt"

    df = pd.read_csv(index)
    model = timm.create_model(TEACHER, pretrained=True, num_classes=0, img_size=a.size)
    model = model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)          # frozen: no grads, no optimizer state
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    policy = PreprocessPolicy.from_data_config(
        timm.data.resolve_data_config({}, model=model), a.size)
    mean, std = norm_tensors(policy.normalise_mean, policy.normalise_std, dev)

    data: dict = {}
    for split in ("train", "val", "test"):
        part = df[df.split == split]
        if len(part):
            data[split] = extract_resumable(model, part, split, cache_dir,
                                            size=a.size, dev=dev, mean=mean, std=std,
                                            bs=a.bs, workers=a.workers,
                                            amp_dtype=amp_dtype, fingerprint=fp,
                                            fresh=a.fresh)
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()         # free the backbone before head training
    if "train" not in data or "val" not in data:
        raise SystemExit("the probe needs both a train and a val split; check the index")

    xtr = torch.from_numpy(data["train"][0]).to(dev)
    ytr = torch.from_numpy(data["train"][1]).to(dev)
    xva = torch.from_numpy(data["val"][0]).to(dev)
    yva = torch.from_numpy(data["val"][1]).to(dev)

    absent: list = []
    if a.head == "torch":
        head, history, hinfo = train_head_torch(
            xtr, ytr, xva, yva, n_classes=len(CLASSES), epochs=a.epochs,
            patience=a.patience, min_epochs=a.min_epochs, lr=a.lr, bs=a.head_bs,
            log_every=a.log_every, seed=a.seed, ckpt_path=ckpt_path, fingerprint=fp)
        absent = _pin_absent_classes(head, ytr)
        W = head.weight.detach().cpu().numpy().astype(np.float32)
        b = head.bias.detach().cpu().numpy().astype(np.float32)
        C = scores = None
        head_desc = (f"torch-linear AdamW lr={a.lr} wd=1e-2 ls=0.05 "
                     f"epochs={hinfo['epochs_run']}/{a.epochs}")
    else:
        W, b, C, scores = fit_head(data["train"][0], data["train"][1],
                                   data["val"][0], data["val"][1],
                                   n_classes=len(CLASSES), seed=a.seed)
        history, hinfo = [], {}
        head_desc = f"LogisticRegression C={C}"

    yva_np = data["val"][1]
    logits_va = xva.cpu().numpy() @ W.T + b
    T = float(temperature_scale(logits_va, yva_np))
    probs_va = softmax_np(logits_va / T)
    try:
        tau = float(pick_threshold(probs_va, yva_np, a.target_recall))
    except ValueError as exc:
        LOG.warning("%s -- tau=0.5 fallback (every sample goes to a human)", exc)
        tau = 0.5
    try:
        tau_reject = float(pick_reject_threshold(probs_va, yva_np))
    except ValueError as exc:
        LOG.warning("%s -- no REJECT zone; every HOLD stays risk_hold", exc)
        tau_reject = None

    metrics: dict = {
        "model": TEACHER, "head": head_desc, "temperature": T,
        "tau_from_val": tau, "tau_reject_from_val": tau_reject,
        "target_unsafe_recall": a.target_recall,
        "arch": f"{TEACHER}+linear-probe", "size": a.size,
        "absent_class_bias": ABSENT_CLASS_BIAS,
        "classes_absent_from_train": [CLASSES[i] for i in absent],
        "val": evaluate(probs_va, yva_np, tau, tau_reject=tau_reject),
        "cuda": cuda_report(),
        "fingerprint": fp,
    }
    if scores is not None:
        metrics["c_grid"] = scores
    if history:
        metrics["head_training"] = {**hinfo, "history": history}
    if "test" in data:
        metrics["test_unseen_sources"] = evaluate(
            softmax_np((data["test"][0] @ W.T + b) / T), data["test"][1], tau,
            tau_reject=tau_reject)
        metrics["test_sources"] = (
            sorted(df.loc[df["split"] == "test", "source"].dropna().unique().tolist())
            if "source" in df.columns else []
        )
    else:
        metrics["test_unseen_sources"] = None
        LOG.warning("no test split (no held-out sources). Cross-domain accuracy is "
                    "UNMEASURED -- do not quote a number.")

    out.mkdir(parents=True, exist_ok=True)
    atomic_joblib_dump({"W": W, "b": b, "T": T, "tau": tau, "tau_reject": tau_reject,
                        "classes": CLASSES,
                        "size": a.size, "backbone": TEACHER, "head": a.head,
                        "mean": list(policy.normalise_mean),
                        "std": list(policy.normalise_std),
                        "policy": policy.as_dict(),
                        "val_probs": probs_va,
                        "best_weights_path": (out / "best_head.pt").as_posix()},
                       out / "probe.joblib")
    if a.head == "torch":
        atomic_torch_save(head.state_dict(), out / "best_head.pt")
    else:
        atomic_torch_save({"weight": torch.from_numpy(W.T).float(),
                           "bias": torch.from_numpy(b).float()}, out / "best_head.pt")
    save_json(out / "metrics.json", metrics)
    save_json(out / "policy.json", policy.as_dict())
    save_json(out / "factor.json", {
        "classes_absent_from_train": [CLASSES[i] for i in absent],
        "interpretation": ("On this run, RISK and NOT_FOOD were absent from the training "
                           "split, so their head rows were pinned to bias -30 and can never "
                           "win the argmax. The shipped head is therefore a strong GOOD vs "
                           "REJECTED core; RISK here is a decision zone over that core's "
                           "release score, not a learned RISK class on this data."),
        "recommendation": ("To make RISK a learned class, add real RISK and NOT_FOOD rows to "
                           "the training split (source images + labels, mapped into the "
                           "existing index) and re-run the probe. Until then, treat the RISK "
                           "band as human-review-on-uncertainty, not as evidence of early "
                           "spoilage."),
        "run_is_reusable": True,
    })

    for split in ("val", "test_unseen_sources"):
        m = metrics.get(split)
        if m:
            LOG.info("%-20s macro-F1 %.3f  unsafe recall %s  false accepts %s  "
                     "good auto-accept %s", split, m["macro_f1"], m["unsafe_recall"],
                     m["false_accepts"], m["good_auto_accept_rate"])
    LOG.info("saved %s  (%.0fs) -- rerun the same command any time; caches and "
             "checkpoints are reused, so a rerun only recomputes what is missing",
             out, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
