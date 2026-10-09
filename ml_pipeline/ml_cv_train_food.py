from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ml_cv_common import (
    C2I,
    CLASSES,
    NUM_CLASSES,
    ModelEma,
    PreprocessPolicy,
    autocast_ctx,
    build_model,
    class_weights_from,
    dataset_balance,
    dataset_fingerprint,
    evaluate,
    logits_of,
    mixup_batch,
    norm_tensors,
    normalise,
    one_hot,
    pick_best,
    pick_reject_threshold,
    pick_threshold,
    resolve_worker_count,
    save_checkpoint,
    selection_score,
    softmax_np,
    temperature_scale,
)
from ml_cv_train_cls import OomRetry, backbone_prefixes, fit_batch_size, load_checkpoint, set_backbone_trainable
from ml_utils import (
    StageTimer,
    atomic_joblib_dump,
    atomic_torch_save,
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
    torch_load,
)

LOG = get_logger("cv.train_food")
SOURCE_COL = "source"
DEFAULT_MAX_TEMP_C = 80.0


class FoodImageDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        size: int,
        train: bool,
        policy: PreprocessPolicy,
        aug_strength: str = "light",
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.size = int(size)
        self.train = bool(train)
        self.mean = tuple(float(x) for x in policy.normalise_mean)
        self.std = tuple(float(x) for x in policy.normalise_std)
        self.transform = self._build_transform(aug_strength)

    def _build_transform(self, aug_strength: str):
        resize_size = max(self.size, int(round(self.size * 256 / 224)))
        ops: list[Any] = []
        if self.train:
            scale = {
                "light": (0.8, 1.0),
                "medium": (0.7, 1.0),
                "strong": (0.6, 1.0),
            }.get(aug_strength, (0.8, 1.0))
            ops.extend([
                transforms.RandomResizedCrop(self.size, scale=scale),
                transforms.RandomHorizontalFlip(),
            ])
            if aug_strength in {"medium", "strong"}:
                ops.append(
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.04)],
                        p=0.5,
                    )
                )
            if aug_strength == "strong":
                ops.extend([
                    transforms.RandomApply([transforms.RandomRotation(12)], p=0.3),
                    transforms.RandomGrayscale(p=0.05),
                ])
        else:
            ops.extend([
                transforms.Resize(resize_size),
                transforms.CenterCrop(self.size),
            ])
        ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        return transforms.Compose(ops)

    @staticmethod
    def _label_to_index(value: Any) -> int:
        if isinstance(value, (int, np.integer)):
            idx = int(value)
        elif isinstance(value, (float, np.floating)) and float(value).is_integer():
            idx = int(value)
        else:
            text_value = str(value).strip()
            if text_value in C2I:
                idx = int(C2I[text_value])
            else:
                idx = int(float(text_value))
        if not 0 <= idx < NUM_CLASSES:
            raise ValueError(f"label index out of range: {idx}")
        return idx

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = Path(str(row["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {path}")
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self._label_to_index(row["label"])


def _seed_worker(worker_id: int, base_seed: int) -> None:
    seed = (int(base_seed) + int(worker_id)) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader_food(
    frame: pd.DataFrame,
    size: int,
    train: bool,
    *,
    bs: int,
    workers: int,
    policy: PreprocessPolicy,
    aug_strength: str = "light",
    seed: int = 42,
):
    if bs <= 0:
        raise ValueError("bs must be positive")
    if workers < 0:
        raise ValueError("workers must be non-negative")
    dataset = FoodImageDataset(frame, size, train, policy, aug_strength)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    kwargs: dict[str, Any] = {
        "batch_size": int(bs),
        "shuffle": bool(train),
        "num_workers": int(workers),
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
        "generator": generator,
        "worker_init_fn": partial(_seed_worker, base_seed=int(seed)),
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = [x.cpu() for x in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _model_card(args, metrics: dict, policy: PreprocessPolicy, config: dict) -> dict:
    return {
        "task": "visible food quality classification from images",
        "claim": "Estimates visible food quality; does not certify safety or detect pathogens.",
        "classes": list(CLASSES),
        "architecture": args.arch,
        "input_size": int(args.size),
        "normalisation": policy.as_dict(),
        "best_epoch": metrics.get("best_epoch"),
        "best_weights": metrics.get("best_weights"),
        "validation": metrics.get("val"),
        "unseen_test": metrics.get("test_unseen_sources"),
        "test_sources": metrics.get("test_sources", []),
        "step_evaluations": metrics.get("step_evaluations", []),
        "configuration": config,
        "caveats": metrics.get("caveats", []),
    }


def gpu_health(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {}
    out: dict[str, Any] = {}
    try:
        out["vram_allocated_mb"] = round(torch.cuda.memory_allocated(device) / 1024**2, 1)
        out["vram_reserved_mb"] = round(torch.cuda.memory_reserved(device) / 1024**2, 1)
        props = torch.cuda.get_device_properties(device)
        out["total_vram_mb"] = round(props.total_memory / 1024**2, 1)
    except Exception:
        pass
    try:
        idx = torch.cuda.current_device()
        if hasattr(torch.cuda, "get_device_temperature"):
            out["temperature_c"] = float(torch.cuda.get_device_temperature(idx))
    except Exception:
        pass
    try:
        if hasattr(torch.cuda, "get_device_power_draw"):
            out["power_w"] = float(torch.cuda.get_device_power_draw(torch.cuda.current_device()))
    except Exception:
        pass
    return out


def log_gpu_health(device: torch.device, prefix: str = "") -> None:
    health = gpu_health(device)
    if not health:
        return
    LOG.info("%sGPU health: %s", prefix, " ".join(f"{k}={v}" for k, v in health.items()))
    temperature = health.get("temperature_c")
    if temperature is not None and temperature >= DEFAULT_MAX_TEMP_C:
        LOG.warning("%stemperature %.1fC >= %.1fC", prefix, temperature, DEFAULT_MAX_TEMP_C)


def _save_step_progress(
    path,
    *,
    epoch: int,
    global_step: int,
    batch_in_epoch: int,
    last_loss: float | None,
    last_lr: float | None,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    scheduler,
    scaler,
    ema,
    mean: torch.Tensor,
    std: torch.Tensor,
    loader_state: dict[str, Any],
) -> None:
    ckpt: dict[str, Any] = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "batch_in_epoch": int(batch_in_epoch),
        "last_loss": last_loss,
        "last_lr": last_lr,
        "mean": mean.detach().cpu(),
        "std": std.detach().cpu(),
        "loader_state": loader_state,
        "rng_state": _capture_rng_state(),
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": {"amp": scaler is not None},
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    atomic_joblib_dump(ckpt, str(path))


def _load_step_progress_joblib(path):
    import joblib
    with open(path, "rb") as fh:
        return joblib.load(fh)


def load_progress(path, *, model, opt, scheduler, scaler, ema, mean, std):
    ckpt = _load_step_progress_joblib(path)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if ema is not None and ckpt.get("ema") is not None:
        ema.load_state_dict(ckpt["ema"])
    if ckpt.get("mean") is not None:
        mean = ckpt["mean"].to(device=mean.device, dtype=mean.dtype)
    if ckpt.get("std") is not None:
        std = ckpt["std"].to(device=std.device, dtype=std.dtype)
    _restore_rng_state(ckpt.get("rng_state"))
    return {
        "epoch": int(ckpt["epoch"]),
        "global_step": int(ckpt["global_step"]),
        "batch_in_epoch": int(ckpt["batch_in_epoch"]),
        "last_loss": ckpt.get("last_loss"),
        "last_lr": ckpt.get("last_lr"),
        "loader_state": ckpt.get("loader_state"),
        "mean": mean,
        "std": std,
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--index", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="tf_efficientnet_lite0")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--size-eval", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--head-epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
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
    ap.add_argument("--aug", choices=["auto", "light", "medium", "strong"], default="auto")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--amp", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--resume", nargs="?", const="auto", default=None)
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=0.0)
    ap.add_argument("--max-temp", type=float, default=0.0)
    ap.add_argument("--max-train-time-seconds", type=float, default=0.0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--health-every", type=int, default=200)
    ap.add_argument("--target-unsafe-recall", type=float, default=0.98)
    ap.add_argument("--val-target-recall", type=float, default=0.98)
    ap.add_argument("--progress-every", type=int, default=100)
    ap.add_argument("--eval-every", dest="eval_every", type=int, default=500)
    ap.add_argument("--metrics-every", dest="eval_every", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--resume-state", default=None)
    ap.add_argument("--no-progress-save", action="store_true")
    return ap


def _format_value(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _validation_log(prefix: str, step: int, epoch: int, result: dict[str, Any]) -> None:
    LOG.info(
        "%s step=%d epoch=%d weights=%s macro_f1=%s accuracy=%s unsafe_recall=%s false_accepts=%s good_auto_accept=%s ece=%s tau=%s temperature=%s",
        prefix,
        step,
        epoch + 1,
        result.get("which", "unknown"),
        _format_value(result.get("macro_f1")),
        _format_value(result.get("accuracy")),
        _format_value(result.get("unsafe_recall")),
        _format_value(result.get("false_accepts")),
        _format_value(result.get("good_auto_accept_rate")),
        _format_value(result.get("ece")),
        _format_value(result.get("tau")),
        _format_value(result.get("temperature")),
    )


def _evaluate_candidate(model, val_df, size, policy, device, mean, std, bs, workers, amp_dtype, target_recall):
    previous_training = model.training
    model.eval()
    dl = make_loader_food(val_df, size, False, bs=bs, workers=workers, policy=policy)
    logits, y = logits_of(model, dl, device, mean, std, amp_dtype=amp_dtype)
    try:
        temperature = float(temperature_scale(logits, y))
    except Exception as exc:
        LOG.warning("temperature scaling failed: %s", exc)
        temperature = 1.0
    probabilities = softmax_np(logits / temperature)
    try:
        tau = float(pick_threshold(probabilities, y, target_recall))
    except ValueError as exc:
        LOG.warning("threshold selection failed: %s", exc)
        tau = 0.5
    try:
        tau_reject: float | None = float(pick_reject_threshold(probabilities, y))
    except ValueError as exc:
        LOG.warning("reject-threshold selection failed: %s -- every HOLD stays risk_hold", exc)
        tau_reject = None
    metrics = evaluate(probabilities, y, tau, tau_reject=tau_reject)
    metrics.update({"temperature": temperature, "tau": tau, "tau_reject": tau_reject,
                    "selection": selection_score(metrics)})
    if previous_training:
        model.train()
    return metrics


def evaluate_validation_pair(
    model,
    ema,
    val_df,
    size,
    policy,
    device,
    mean,
    std,
    bs,
    workers,
    amp_dtype,
    target_recall,
    step,
    epoch,
    reason,
    eval_history: list[dict[str, Any]],
):
    rng_state = _capture_rng_state()
    try:
        raw_metrics = _evaluate_candidate(
            model, val_df, size, policy, device, mean, std, bs, workers, amp_dtype, target_recall
        )
        ema_metrics = _evaluate_candidate(
            ema.module, val_df, size, policy, device, mean, std, bs, workers, amp_dtype, target_recall
        )
        candidate = pick_best([
            {"which": "raw", **raw_metrics},
            {"which": "ema", **ema_metrics},
        ])
        which = candidate.pop("which")
        record = {
            "step": int(step),
            "epoch": int(epoch),
            "reason": str(reason),
            "which": which,
            "val": candidate,
        }
        eval_history.append(record)
        return record
    finally:
        _restore_rng_state(rng_state)


def _save_progress_from_train(
    progress_path,
    epoch,
    global_step,
    batch_in_epoch,
    last_loss,
    last_lr,
    model,
    opt,
    scheduler,
    scaler,
    ema,
    mean,
    std,
    save_progress,
):
    if not save_progress or not progress_path:
        return
    try:
        _save_step_progress(
            progress_path,
            epoch=epoch,
            global_step=global_step,
            batch_in_epoch=batch_in_epoch,
            last_loss=last_loss,
            last_lr=last_lr,
            model=model,
            opt=opt,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            mean=mean,
            std=std,
            loader_state={"epoch": epoch},
        )
    except Exception as exc:
        LOG.warning("progress save failed: %s", exc)


def train_one_epoch(
    model: nn.Module,
    loader,
    opt,
    scheduler,
    scaler,
    *,
    ema,
    device,
    amp_dtype,
    mean,
    std,
    class_w,
    mix_prob,
    alpha_mix,
    alpha_cut,
    switch_prob,
    smoothing,
    clip_grad,
    epoch,
    steps_per_epoch,
    progress_every,
    eval_every,
    resume_state,
    total_steps,
    progress_path,
    save_progress,
    val_df,
    size_eval,
    policy,
    workers,
    val_bs,
    target_recall,
    eval_history,
    eval_callback: Callable[[dict[str, Any]], None],
):
    model.train()
    n = len(loader)
    skip_until = int(resume_state["batch_in_epoch"] + 1) if resume_state is not None else 0
    last_loss: float | None = None
    last_lr: float | None = None
    run_loss = 0.0
    run_correct = 0
    run_n = 0
    last_eval_step = 0
    step = -1

    def prepare_soft_targets(yb):
        if yb.dim() == 1:
            return one_hot(yb, NUM_CLASSES, smoothing)
        return yb.to(device=device, dtype=torch.float32)

    def soft_cross_entropy(logits, yb):
        logp = nn.functional.log_softmax(logits, dim=-1)
        if class_w is None:
            return -(yb.detach() * logp).sum(dim=-1).mean()
        weights = torch.as_tensor(class_w, device=logits.device, dtype=logits.dtype)
        return -((yb.detach() * logp) * weights[None, :]).sum(dim=-1).mean()

    try:
        for step, (x, y) in enumerate(loader):
            if step < skip_until:
                continue

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            use_amp = scaler is not None or (amp_dtype is not None and device.type == "cuda")
            with autocast_ctx(device, amp_dtype, use_amp):
                xb, yb = mixup_batch(
                    x,
                    y,
                    alpha_mix=alpha_mix,
                    alpha_cut=alpha_cut,
                    prob=mix_prob,
                    switch_prob=switch_prob,
                    smoothing=smoothing,
                )
                logits = model(normalise(xb, mean, std))
            logits = logits.float()
            yb = prepare_soft_targets(yb)
            loss = soft_cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
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

            batch_size = int(y.shape[0])
            last_loss = float(loss.detach())
            last_lr = float(scheduler.get_last_lr()[0])
            run_loss += last_loss * batch_size
            run_correct += int((logits.detach().argmax(1) == y).sum())
            run_n += batch_size

            completed_step = epoch * steps_per_epoch + step + 1
            remaining = max(0, int(total_steps) - completed_step)

            if progress_every and completed_step % progress_every == 0:
                LOG.info(
                    "progress step=%d/%d epoch=%d batch=%d/%d remaining=%d loss=%s acc=%s lr=%s",
                    completed_step,
                    total_steps,
                    epoch + 1,
                    step + 1,
                    n,
                    remaining,
                    _format_value(last_loss),
                    _format_value(run_correct / max(run_n, 1)),
                    _format_value(last_lr, 6),
                )
                _save_progress_from_train(
                    progress_path,
                    epoch,
                    completed_step,
                    step,
                    last_loss,
                    last_lr,
                    model,
                    opt,
                    scheduler,
                    scaler,
                    ema,
                    mean,
                    std,
                    save_progress,
                )

            if eval_every and completed_step % eval_every == 0:
                _save_progress_from_train(
                    progress_path,
                    epoch,
                    completed_step,
                    step,
                    last_loss,
                    last_lr,
                    model,
                    opt,
                    scheduler,
                    scaler,
                    ema,
                    mean,
                    std,
                    save_progress,
                )
                record = evaluate_validation_pair(
                    model,
                    ema,
                    val_df,
                    size_eval,
                    policy,
                    device,
                    mean,
                    std,
                    val_bs,
                    workers,
                    amp_dtype,
                    target_recall,
                    completed_step,
                    epoch,
                    "step",
                    eval_history,
                )
                record["train_loss"] = last_loss
                record["train_acc"] = run_correct / max(run_n, 1)
                record["lr"] = last_lr
                eval_callback(record)
                _validation_log("validation", completed_step, epoch, {**record["val"], "which": record["which"]})
                last_eval_step = completed_step
                save_json(progress_path.with_suffix(".eval_history.json") if isinstance(progress_path, Path) else Path(str(progress_path) + ".eval_history.json"), eval_history)

        return {
            "loss": run_loss / max(run_n, 1),
            "train_acc": run_correct / max(run_n, 1),
            "n": run_n,
            "last_loss": last_loss,
            "last_lr": last_lr,
            "last_eval_step": last_eval_step,
            "resumed_from_step": int(resume_state["global_step"]) if resume_state else None,
            "resumed_from_epoch": int(resume_state["epoch"]) if resume_state else None,
            "resumed_from_batch": int(resume_state["batch_in_epoch"]) if resume_state else None,
        }
    except KeyboardInterrupt:
        if step >= 0:
            completed_step = epoch * steps_per_epoch + step + 1
            _save_progress_from_train(
                progress_path,
                epoch,
                completed_step,
                step,
                last_loss,
                last_lr,
                model,
                opt,
                scheduler,
                scaler,
                ema,
                mean,
                std,
                save_progress,
            )
            LOG.info("training paused at step %d; progress saved", completed_step)
        raise


def _default_index() -> Path:
    return get_paths().cv_data / "index.csv"


def _load_checkpoint_meta(ckpt: dict[str, Any]) -> dict[str, Any]:
    return {
        "epoch": int(ckpt.get("epoch", -1)),
        "best_score": float(ckpt.get("best_score", -math.inf)),
        "global_step": int(ckpt.get("global_step", 0)),
        "config": ckpt.get("config", {}),
        "history": ckpt.get("history", []),
    }


def _best_which_from_history(history: list[dict[str, Any]], fallback: str = "raw") -> str:
    for record in reversed(history):
        if isinstance(record, dict) and record.get("which") in {"raw", "ema"}:
            return str(record["which"])
    return fallback


def _best_epoch_from_history(history: list[dict[str, Any]]) -> int:
    best_epoch = -1
    best_score = -math.inf
    for record in history:
        if not isinstance(record, dict):
            continue
        val = record.get("val")
        if not isinstance(val, dict):
            continue
        score = val.get("selection")
        if score is None:
            continue
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            continue
        if score_f > best_score:
            best_score = score_f
            best_epoch = int(record.get("epoch", -1))
    return best_epoch


def test_sources_from_df(df: pd.DataFrame) -> list[str]:
    if SOURCE_COL not in df.columns or "split" not in df.columns:
        return []
    return sorted(df.loc[df["split"] == "test", SOURCE_COL].dropna().astype(str).unique().tolist())


def class_w_map_food(tr: pd.DataFrame, device: torch.device):
    _, mapping = class_weights_from(tr, device=device)
    return mapping


def _selection_update(
    record: dict[str, Any],
    *,
    best_score: float,
    best_epoch: int,
    best_which: str,
    model,
    ema,
    out: Path,
    ck_best: Path,
    ck_last: Path,
    opt,
    scheduler,
    scaler,
    config: dict[str, Any],
    history: list[dict[str, Any]],
    epoch: int,
    steps_per_epoch: int,
) -> tuple[float, int, str]:
    score = float(record["val"].get("selection", -math.inf))
    if score <= best_score:
        return best_score, best_epoch, best_which
    best_score = score
    best_epoch = int(record["epoch"])
    best_which = str(record["which"])
    config["best_which"] = best_which
    best_state = ema.module.state_dict() if best_which == "ema" else model.state_dict()
    save_checkpoint(
        ck_best,
        model=model,
        optimizer=opt,
        scheduler=scheduler,
        scaler=scaler,
        ema=ema,
        epoch=epoch,
        best_score=best_score,
        global_step=int(record["step"]),
        config=config,
        history=history,
        model_state=best_state,
    )
    LOG.info(
        "best checkpoint updated at step=%d epoch=%d weights=%s selection=%s",
        int(record["step"]),
        int(record["epoch"]) + 1,
        best_which,
        _format_value(score),
    )
    return best_score, best_epoch, best_which


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.eval_every <= 0:
        raise ValueError("--eval-every must be positive")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.bs <= 0:
        raise ValueError("--bs must be positive")

    timer = StageTimer()
    det = set_determinism(args.seed, deterministic=args.deterministic)
    device, dev_reason = resolve_device(args.device)
    LOG.info("device: %s (%s)", device, dev_reason)
    cuda = cuda_report()
    if device.type == "cuda" and not cuda.get("available"):
        LOG.warning("CUDA unavailable: %s", dev_reason)
    perf = configure_torch_performance(device)
    amp_dtype = resolve_amp_dtype(device)
    amp_on = args.amp != "off" and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type) if amp_on and amp_dtype is torch.float16 else None

    size = int(args.size)
    size_eval = int(args.size_eval) if args.size_eval else size
    index = Path(args.index) if args.index else _default_index()
    if not index.exists():
        LOG.error("index csv not found: %s", index)
        return 2

    df = pd.read_csv(index)
    required_columns = {"path", "label", "split"}
    missing = required_columns.difference(df.columns)
    if missing:
        LOG.error("index csv missing columns: %s", sorted(missing))
        return 2
    if SOURCE_COL not in df.columns:
        LOG.warning("index csv has no source column")

    tr = df.loc[df["split"] == "train"].copy()
    va = df.loc[df["split"] == "val"].copy()
    te = df.loc[df["split"] == "test"].copy()
    for name, part in (("train", tr), ("val", va), ("test", te)):
        LOG.info("%-6s n=%-8d %s", name, len(part), dataset_balance(part)["counts"] if len(part) else "-")
    if not len(tr) or not len(va):
        LOG.error("empty train or val split")
        return 2

    train_sources = sorted(tr[SOURCE_COL].dropna().astype(str).unique().tolist()) if SOURCE_COL in tr else []
    test_sources = test_sources_from_df(df)
    if test_sources and set(train_sources) & set(test_sources):
        LOG.warning("test source overlap detected: train=%s test=%s", train_sources, test_sources)
    else:
        LOG.info("test sources=%s train sources=%s", test_sources, train_sources)

    out = Path(args.out)
    if not out.is_absolute():
        run_root = Path(get_paths().runs)
        out = out if out.parts and out.parts[0] == run_root.name else run_root / out
    out.mkdir(parents=True, exist_ok=True)

    aug = args.aug
    if aug == "auto":
        aug = "strong" if len(tr) < 2000 else "medium" if len(tr) < 10000 else "light"

    bs = fit_batch_size(args.arch, size, args.bs, vram_gb=cuda.get("total_vram_gb"), device=device)
    if bs != args.bs:
        LOG.info("batch size fitted %d -> %d", args.bs, bs)
    workers = resolve_worker_count(args.workers, batch_size=bs)

    model, policy = build_model(
        args.arch,
        pretrained=True,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path,
        size=size,
    )
    model.to(device)
    mean, std = norm_tensors(policy.normalise_mean, policy.normalise_std, device)
    prefixes = backbone_prefixes(args.arch)
    class_w, class_w_map = class_weights_from(tr, device=device)

    opt = torch.optim.AdamW(
        param_groups(
            model,
            base_lr=args.lr,
            backbone_prefixes=prefixes,
            head_lr_mult=args.head_lr_mult,
            weight_decay=args.wd,
        )
    )

    probe_loader = make_loader_food(
        tr,
        size,
        True,
        bs=bs,
        workers=workers,
        policy=policy,
        aug_strength=aug,
        seed=args.seed,
    )
    steps_per_epoch = max(1, len(probe_loader))
    del probe_loader
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_frac)
    head_steps = steps_per_epoch * min(args.head_epochs, args.epochs)

    def lr_lambda(step: int) -> float:
        if step < head_steps:
            return max(1e-3, (step + 1) / max(1, head_steps)) * 0.5
        return linear_warmup_cosine(step, total_steps, warmup_steps, args.min_lr_ratio)

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ema = ModelEma(model, decay=args.ema_decay, device=device)

    config: dict[str, Any] = {
        "arch": args.arch,
        "size": size,
        "size_eval": size_eval,
        "bs": bs,
        "epochs": args.epochs,
        "head_epochs": args.head_epochs,
        "lr": args.lr,
        "head_lr_mult": args.head_lr_mult,
        "wd": args.wd,
        "warmup_frac": args.warmup_frac,
        "min_lr_ratio": args.min_lr_ratio,
        "smoothing": args.smoothing,
        "mix_prob": args.mix_prob,
        "alpha_mix": args.alpha_mix,
        "alpha_cut": args.alpha_cut,
        "switch_prob": args.switch_prob,
        "clip_grad": args.clip_grad,
        "drop_rate": args.drop_rate,
        "drop_path": args.drop_path,
        "aug": aug,
        "ema_decay": args.ema_decay,
        "seed": args.seed,
        "amp": amp_on,
        "amp_dtype": str(amp_dtype),
        "policy": policy.as_dict(),
        "index": str(index),
        "dataset_sha256": dataset_fingerprint(df),
        "determinism": det,
        "torch": str(require("torch").__version__),
        "task": "food quality classification from multiple sources",
        "classes": CLASSES,
        "test_sources": test_sources,
        "train_sources": train_sources,
        "source_col": SOURCE_COL,
        "eval_every": args.eval_every,
        "progress_every": args.progress_every,
        "best_which": "raw",
    }

    ck_last = out / "last.pt"
    ck_best = out / "best.pt"
    progress_path = Path(args.resume_state) if args.resume_state else out / "progress.pt"
    history_path = out / "history.json"
    eval_history_path = out / "eval_history.json"
    start_epoch = 0
    best_score = -math.inf
    best_epoch = -1
    best_which = "raw"
    history: list[dict[str, Any]] = []
    eval_history: list[dict[str, Any]] = []
    resume_state = None
    train_t0 = time.time()

    if args.resume:
        explicit_checkpoint = args.resume != "auto"
        ck_path = ck_last if not explicit_checkpoint else Path(args.resume)
        if explicit_checkpoint and not ck_path.exists():
            LOG.error("checkpoint not found: %s", ck_path)
            return 2
        resumed_meta: dict[str, Any] = {}
        if ck_path.exists():
            resumed_meta = torch_load(ck_path, map_location="cpu")
            meta = _load_checkpoint_meta(resumed_meta)
            load_checkpoint(ck_path, model=model, optimizer=opt, scheduler=scheduler, scaler=scaler, ema=ema)
            _restore_rng_state(resumed_meta.get("rng_state"))
            start_epoch = int(meta["epoch"]) + 1
            best_score = float(meta["best_score"])
            history = list(meta.get("history", []))
            best_which = str(meta.get("config", {}).get("best_which", _best_which_from_history(history)))
            best_epoch = _best_epoch_from_history(history)
        elif not progress_path.exists():
            LOG.error("no checkpoint or progress state found for resume")
            return 2
        if history_path.exists():
            try:
                with open(history_path, "r", encoding="utf-8") as fh:
                    history = json.load(fh)
            except Exception:
                history = list(meta.get("history", []))
        if eval_history_path.exists():
            try:
                with open(eval_history_path, "r", encoding="utf-8") as fh:
                    eval_history = json.load(fh)
            except Exception:
                eval_history = []
        run_state_path = out / "run_state.json"
        if run_state_path.exists():
            try:
                with open(run_state_path, "r", encoding="utf-8") as fh:
                    run_state = json.load(fh)
                best_score = float(run_state.get("best_score", best_score))
                best_epoch = int(run_state.get("best_epoch", best_epoch))
                best_which = str(run_state.get("best_which", best_which))
            except Exception:
                pass
        state_candidate = progress_path
        if state_candidate.exists():
            try:
                candidate = load_progress(
                    state_candidate,
                    model=model,
                    opt=opt,
                    scheduler=scheduler,
                    scaler=scaler,
                    ema=ema,
                    mean=mean,
                    std=std,
                )
                candidate_epoch = int(candidate["epoch"])
                if candidate_epoch >= start_epoch:
                    resume_state = candidate
                    start_epoch = candidate_epoch
                    LOG.info(
                        "resuming epoch=%d step=%d batch=%d",
                        candidate_epoch + 1,
                        int(candidate["global_step"]),
                        int(candidate["batch_in_epoch"]) + 1,
                    )
                else:
                    LOG.info("ignoring stale progress checkpoint from epoch=%d", candidate_epoch + 1)
            except Exception as exc:
                LOG.warning("progress checkpoint load failed: %s", exc)

    if start_epoch >= args.epochs:
        LOG.info("training already complete")
        return _finalise_from_best(
            out,
            ck_best if ck_best.exists() else ck_last,
            args,
            device,
            tr,
            va,
            te,
            df,
            size,
            size_eval,
            policy,
            mean,
            std,
            bs,
            workers,
            amp_dtype,
            config,
            history,
            eval_history,
            timer,
            train_t0,
            aug,
            cuda,
            perf,
            det,
        )

    oom = OomRetry(device)

    def on_eval(record: dict[str, Any]) -> None:
        nonlocal best_score, best_epoch, best_which
        best_score, best_epoch, best_which = _selection_update(
            record,
            best_score=best_score,
            best_epoch=best_epoch,
            best_which=best_which,
            model=model,
            ema=ema,
            out=out,
            ck_best=ck_best,
            ck_last=ck_last,
            opt=opt,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            history=history,
            epoch=int(record["epoch"]),
            steps_per_epoch=steps_per_epoch,
        )
        config["best_which"] = best_which
        save_json(eval_history_path, eval_history)
        save_json(out / "run_state.json", {
            "best_score": best_score,
            "best_epoch": best_epoch,
            "best_which": best_which,
            "global_step": int(record["step"]),
        })

    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_t0 = time.time()
            global_step_start = epoch * steps_per_epoch
            trainable = set_backbone_trainable(model, args.arch, global_step_start >= head_steps)
            LOG.info(
                "epoch %02d/%d stage=%s trainable_m=%.2f lr=%s elapsed_min=%.1f bs=%d",
                epoch + 1,
                args.epochs,
                "full_finetune" if global_step_start >= head_steps else "head_only",
                trainable / 1e6,
                _format_value(scheduler.get_last_lr()[0], 6),
                (time.time() - train_t0) / 60,
                bs,
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                log_gpu_health(device, prefix=f"[epoch {epoch + 1}] ")

            loader = make_loader_food(
                tr,
                size,
                True,
                bs=bs,
                workers=workers,
                policy=policy,
                aug_strength=aug,
                seed=args.seed + epoch,
            )
            with oom.guard(bs) as effective_bs:
                if effective_bs != bs:
                    loader = make_loader_food(
                        tr,
                        size,
                        True,
                        bs=effective_bs,
                        workers=workers,
                        policy=policy,
                        aug_strength=aug,
                        seed=args.seed + epoch,
                    )
                    bs = effective_bs
                with timer.stage("train"):
                    trn = train_one_epoch(
                        model,
                        loader,
                        opt,
                        scheduler,
                        scaler,
                        ema=ema,
                        device=device,
                        amp_dtype=amp_dtype,
                        mean=mean,
                        std=std,
                        class_w=class_w,
                        mix_prob=args.mix_prob,
                        alpha_mix=args.alpha_mix,
                        alpha_cut=args.alpha_cut,
                        switch_prob=args.switch_prob,
                        smoothing=args.smoothing,
                        clip_grad=args.clip_grad,
                        epoch=epoch,
                        steps_per_epoch=steps_per_epoch,
                        progress_every=args.progress_every,
                        eval_every=args.eval_every,
                        resume_state=resume_state,
                        total_steps=total_steps,
                        progress_path=progress_path,
                        save_progress=not args.no_progress_save,
                        val_df=va,
                        size_eval=size_eval,
                        policy=policy,
                        workers=workers,
                        val_bs=bs,
                        target_recall=args.val_target_recall,
                        eval_history=eval_history,
                        eval_callback=on_eval,
                    )
                    resume_state = None
            del loader

            with timer.stage("epoch_end_eval"):
                epoch_completed_step = (epoch + 1) * steps_per_epoch
                if not eval_history or int(eval_history[-1].get("step", -1)) != epoch_completed_step:
                    record = evaluate_validation_pair(
                        model,
                        ema,
                        va,
                        size_eval,
                        policy,
                        device,
                        mean,
                        std,
                        bs,
                        workers,
                        amp_dtype,
                        args.val_target_recall,
                        epoch_completed_step,
                        epoch,
                        "epoch_end",
                        eval_history,
                    )
                    record["train_loss"] = trn["loss"]
                    record["train_acc"] = trn["train_acc"]
                    record["lr"] = trn["last_lr"]
                    on_eval(record)
                    _validation_log("epoch_end_validation", epoch_completed_step, epoch, {**record["val"], "which": record["which"]})

            last_eval = eval_history[-1]
            last_eval["train_loss"] = trn["loss"]
            last_eval["train_acc"] = trn["train_acc"]
            last_eval["lr"] = trn["last_lr"]
            epoch_record = {
                "epoch": epoch,
                "global_step": epoch_completed_step,
                "which": str(last_eval["which"]),
                "train": trn,
                "val": last_eval["val"],
                "peak_mem_mb": (torch.cuda.max_memory_allocated() / 1024**2) if device.type == "cuda" else None,
                "elapsed_min": (time.time() - train_t0) / 60,
                "epoch_seconds": time.time() - epoch_t0,
                "test_sources": test_sources,
                "lr": trn["last_lr"],
            }
            history.append(epoch_record)
            save_json(history_path, history)
            save_json(eval_history_path, eval_history)

            config["best_which"] = best_which
            save_checkpoint(
                ck_last,
                model=model,
                optimizer=opt,
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
                epoch=epoch,
                best_score=best_score,
                global_step=epoch_completed_step,
                config=config,
                history=history,
            )
            if progress_path.exists():
                try:
                    progress_path.unlink()
                except OSError:
                    pass

            stale = epoch - best_epoch
            LOG.info(
                "epoch %02d summary loss=%s train_acc=%s val_macro_f1=%s val_unsafe_recall=%s false_accepts=%s good_auto_accept=%s ece=%s weights=%s best_epoch=%d best_selection=%s",
                epoch + 1,
                _format_value(trn["loss"]),
                _format_value(trn["train_acc"]),
                _format_value(eval_history[-1]["val"].get("macro_f1")),
                _format_value(eval_history[-1]["val"].get("unsafe_recall")),
                _format_value(eval_history[-1]["val"].get("false_accepts")),
                _format_value(eval_history[-1]["val"].get("good_auto_accept_rate")),
                _format_value(eval_history[-1]["val"].get("ece")),
                eval_history[-1]["which"],
                best_epoch + 1,
                _format_value(best_score),
            )

            if stale >= args.patience:
                LOG.info("early stopping after %d stagnant epochs", stale)
                break
            if args.max_minutes and (time.time() - train_t0) / 60 >= args.max_minutes:
                LOG.info("max minutes reached; last checkpoint is resumable")
                break
            if args.max_temp:
                temperature = gpu_health(device).get("temperature_c")
                if temperature is not None and temperature >= args.max_temp:
                    LOG.info("temperature limit reached at %.1fC", temperature)
                    break
            if args.max_train_time_seconds and (time.time() - epoch_t0) >= args.max_train_time_seconds:
                LOG.info("epoch time limit reached")
                break
    except KeyboardInterrupt:
        LOG.info("training interrupted")
        return 130

    return _finalise_from_best(
        out,
        ck_best if ck_best.exists() else ck_last,
        args,
        device,
        tr,
        va,
        te,
        df,
        size,
        size_eval,
        policy,
        mean,
        std,
        bs,
        workers,
        amp_dtype,
        config,
        history,
        eval_history,
        timer,
        train_t0,
        aug,
        cuda,
        perf,
        det,
    )


def evaluate_candidate_food(
    model,
    val_df,
    size,
    policy,
    device,
    mean,
    std,
    bs,
    workers,
    amp_dtype,
    *,
    target_recall: float,
) -> dict[str, Any]:
    return _evaluate_candidate(
        model,
        val_df,
        size,
        policy,
        device,
        mean,
        std,
        bs,
        workers,
        amp_dtype,
        target_recall,
    )


def _store_predictions(probs, y, df, path, *, limit=20000):
    probabilities = np.asarray(probs, dtype=float)
    targets = np.asarray(y)
    paths = df["path"].tolist() if "path" in df.columns else None
    if len(probabilities) > limit:
        indices = np.linspace(0, len(probabilities) - 1, limit).astype(int)
        probabilities = probabilities[indices]
        targets = targets[indices]
        if paths is not None:
            paths = [paths[i] for i in indices]
    payload = {
        "n": int(len(targets)),
        "classes": CLASSES,
        "probs": np.round(probabilities, 6).tolist(),
        "y_true": targets.astype(int).tolist(),
        "paths": paths,
        "note": "rounded to 6 dp for audit; metrics are computed at full precision",
    }
    save_json(path, payload)
    return {"n": payload["n"], "path": str(path)}


def _finalise_from_best(
    out,
    ck_path,
    args,
    device,
    tr,
    va,
    te,
    df,
    size,
    size_eval,
    policy,
    mean,
    std,
    bs,
    workers,
    amp_dtype,
    config,
    history,
    eval_history,
    timer,
    t0,
    aug,
    cuda,
    perf,
    det,
):
    model, _ = build_model(
        args.arch,
        pretrained=False,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path,
        size=size,
    )
    model.to(device)
    meta = load_checkpoint(ck_path, model=model, optimizer=None, scheduler=None, scaler=None, ema=None)
    ckpt = torch_load(ck_path, map_location="cpu")
    best_which = str(ckpt.get("config", {}).get("best_which", config.get("best_which", "raw")))
    if ckpt.get("ema"):
        ema = ModelEma(model, decay=float(config.get("ema_decay", 0.999)), device=device)
        ema.load_state_dict(ckpt["ema"])
        if best_which == "ema":
            model.load_state_dict(ema.module.state_dict())
    model.eval()

    m_val = {"n": 0}
    temperature = 1.0
    tau = 0.5
    if len(va):
        with timer.stage("final_val"):
            m_val = evaluate_candidate_food(
                model,
                va,
                size_eval,
                policy,
                device,
                mean,
                std,
                bs,
                workers,
                amp_dtype,
                target_recall=args.val_target_recall,
            )
        temperature = float(m_val.get("temperature", 1.0))
        tau = float(m_val.get("tau", 0.5))
        tau_reject = m_val.get("tau_reject")

        with timer.stage("store_val_preds"):
            val_loader = make_loader_food(va, size_eval, False, bs=bs, workers=workers, policy=policy)
            val_logits, val_y = logits_of(model, val_loader, device, mean, std, amp_dtype=amp_dtype)
            _store_predictions(softmax_np(val_logits / temperature), val_y, va, out / "preds_val.json")

    metrics: dict[str, Any] = {
        "arch": args.arch,
        "size": size,
        "size_eval": size_eval,
        "batch_size": bs,
        "best_epoch": int(ckpt.get("epoch", -1)) + 1,
        "best_weights": best_which,
        "epochs_run": len(history),
        "temperature": temperature,
        "tau_from_val": tau,
        "target_unsafe_recall": args.target_unsafe_recall,
        "aug": aug,
        "seed": args.seed,
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
        "class_weights": {k: round(v, 4) for k, v in class_w_map_food(tr, device).items()},
        "dataset": {split: dataset_balance(df[df.split == split]) for split in ("train", "val", "test") if (df["split"] == split).any()},
        "val": m_val,
        "test_unseen_sources": None,
        "test_sources": test_sources_from_df(df),
        "step_evaluations": eval_history,
        "cuda": cuda,
        "perf": perf,
        "determinism": det,
        "stage_seconds": timer.as_dict(),
        "train_minutes": round((time.time() - t0) / 60, 2),
        "config": config,
        "caveats": [
            "This model estimates visible food quality from photos and does not certify food safety.",
            "Unsafe recall is not measured on an unseen test split without unsafe classes.",
        ],
    }

    if not args.no_test and len(te):
        test_labels = te["label"].tolist()
        has_unsafe = any(label in (CLASSES[1], CLASSES[2]) for label in test_labels)
        if has_unsafe:
            with timer.stage("final_test"):
                test_loader = make_loader_food(te, size_eval, False, bs=bs, workers=workers, policy=policy)
                test_logits, test_y = logits_of(model, test_loader, device, mean, std, amp_dtype=amp_dtype)
                test_probs = softmax_np(test_logits / temperature)
                metrics["test_unseen_sources"] = evaluate(test_probs, test_y, tau,
                                                          tau_reject=tau_reject)
                _store_predictions(test_probs, test_y, te, out / "preds_test.json")
        else:
            LOG.warning("test split has no unsafe classes; unsafe recall is not measured")
            metrics["test_unseen_sources"] = {
                "n": int(len(te)),
                "note": "no unsafe class in test split; unsafe recall unmeasured",
                "unsafe_recall": None,
                "meets_recall_target": None,
            }
    else:
        LOG.warning("no test split; cross-domain accuracy is unmeasured")

    atomic_torch_save(model.state_dict(), out / "model.pt")
    atomic_joblib_dump(
        {
            "arch": args.arch,
            "size": size_eval,
            "mean": list(policy.normalise_mean),
            "std": list(policy.normalise_std),
            "T": temperature,
            "tau": tau,
            "tau_reject": tau_reject,
            "classes": CLASSES,
            "policy": policy.as_dict(),
        },
        out / "calib.joblib",
    )
    save_json(out / "preprocess.json", policy.as_dict())
    save_json(out / "metrics.json", metrics)
    save_json(
        out / "run.json",
        {
            "config": config,
            "env": env_fingerprint(),
            "history": history,
            "eval_history": eval_history,
            "best_epoch": metrics["best_epoch"],
            "best_weights": best_which,
        },
    )
    save_json(out / "card.json", _model_card(args, metrics, policy, config))

    rows = []
    for split in ("val", "test_unseen_sources"):
        metric = metrics.get(split)
        if not metric:
            continue
        rows.append(
            {
                "split": split,
                "n": metric.get("n"),
                "macro_f1": round(metric["macro_f1"], 4) if metric.get("macro_f1") is not None else None,
                "accuracy": round(metric["accuracy"], 4) if metric.get("accuracy") is not None else None,
                "unsafe_recall": metric.get("unsafe_recall"),
                "false_accepts": metric.get("false_accepts"),
                "good_auto_accept": metric.get("good_auto_accept_rate"),
                "ece": metric.get("ece"),
            }
        )
    print()
    print(format_table(rows))
    print(
        f"best checkpoint epoch {metrics['best_epoch']} ({best_which}) "
        f"T={temperature:.3f} tau={tau:.4f} train_minutes={metrics['train_minutes']:.1f}"
    )
    print(f"artifacts -> {out}")
    test_metric = metrics.get("test_unseen_sources")
    if isinstance(test_metric, dict) and test_metric.get("unsafe_recall") is not None and test_metric["unsafe_recall"] < args.target_unsafe_recall:
        LOG.warning(
            "test unsafe recall %.4f < target %.4f",
            float(test_metric["unsafe_recall"]),
            args.target_unsafe_recall,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
