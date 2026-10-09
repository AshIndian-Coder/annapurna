from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFile
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from ml_utils import get_logger, get_paths, sha256_json

ImageFile.LOAD_TRUNCATED_IMAGES = True

LOG = get_logger("cv_common")

CLASSES: list[str] = ["GOOD", "RISK", "REJECTED", "NOT_FOOD"]
C2I: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
I2C: dict[int, str] = {i: c for i, c in enumerate(CLASSES)}
GOOD = C2I["GOOD"]
NOT_FOOD = C2I["NOT_FOOD"]

UNSAFE_IDX: list[int] = [C2I["RISK"], C2I["REJECTED"]]
NUM_CLASSES = len(CLASSES)

CROP_PCT = 0.875

@dataclass(frozen=True)
class PreprocessPolicy:
    """Serialisable description of the image front-end.

    Written to ``bundle/preprocess.json`` and read by the Flutter app, so adding a
    transform here is a breaking change unless the policy object travels with it.
    """

    size: int = 224
    crop_pct: float = CROP_PCT
    interpolation: str = "bilinear"
    #: shortest-side target actually fed to the resize call
    resize_short: int = 256
    color_space: str = "RGB"
    layout: str = "NCHW"
    value_range: str = "0..1"
    normalise_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalise_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    # int8 quantisation constants, filled in by ml_cv_export from the TFLite
    # converter so the device quantises with exactly the same scale/zero-point.
    quant_scale: float | None = None
    quant_zero_point: int | None = None

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_data_config(cls, cfg: dict, size: int) -> "PreprocessPolicy":
        mean = tuple(float(v) for v in cfg.get("mean", (0.485, 0.456, 0.406)))
        std = tuple(float(v) for v in cfg.get("std", (0.229, 0.224, 0.225)))
        return cls(size=size, crop_pct=CROP_PCT,
                   resize_short=resize_short_for(size),
                   normalise_mean=mean, normalise_std=std)


def resize_short_for(size: int, crop_pct: float = CROP_PCT) -> int:
    """Shortest-side target that yields exactly ``size`` after the centre crop."""
    return int(round(size / crop_pct))


def eval_policy(size: int = 224, crop_pct: float = CROP_PCT) -> PreprocessPolicy:
    return PreprocessPolicy(size=size, crop_pct=crop_pct, resize_short=resize_short_for(size, crop_pct))


def policy_from_manifest(data: dict) -> PreprocessPolicy:
    """Rebuild a policy from ``preprocess.json`` (used by export and the parity gate)."""
    fields = {f for f in PreprocessPolicy.__dataclass_fields__}
    return PreprocessPolicy(**{k: v for k, v in data.items() if k in fields})


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def train_tf(size: int = 224, *, strength: str = "medium") -> T.Compose:
    """Training augmentation, tuned for cooked-food spoilage.

    ``strength`` maps to the data budget: ``light`` for >=10k photos, ``medium``
    (default) for 2-10k, ``strong`` for the <2k regime a first pilot produces.
    """
    heavy = strength == "strong"
    med = strength == "medium"
    ops: list[Any] = [
        # scale=(0.6, 1.0) per spec: mild crop jitter, not the aggressive 0.08 used by
        # ImageNet recipes that assume 100k images.
        T.RandomResizedCrop(size, scale=(0.35 if heavy else 0.6, 1.0),
                            ratio=(0.75, 1.3333), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.RandomRotation(degrees=15, interpolation=T.InterpolationMode.BILINEAR)],
                      p=0.5 if med else 0.3),
        # brightness/contrast +-25 % (spec), hue <= 0.03 (spec), saturation 0.0.
        T.ColorJitter(brightness=(0.75, 1.25) if med else (0.85, 1.15),
                      contrast=(0.75, 1.25) if med else (0.85, 1.15),
                      saturation=(1.0, 1.0),
                      hue=(-0.03, 0.03)),
        T.RandomGrayscale(p=0.02),
    ]
    ops += [
        # Camera realism: the app re-encodes at JPEG q80 (D25) and phones apply noise
        # reduction.  Training through a compression+noise chain closes that gap.
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.2))],
                      p=0.25 if med else 0.15),
        T.Lambda(_jpeg_jitter),
    ]
    if heavy:
        ops.append(T.RandomGrayscale(p=0.05))
    ops += [T.ToTensor()]
    if med or heavy:
        ops.append(T.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.3, 3.3),
                                  value="random"))
    return T.Compose(ops)


def _jpeg_jitter(img: Image.Image) -> Image.Image:
    """Re-encode at a random quality in the 40-95 band (spec 06 section 8)."""
    try:
        buf = _jpeg_buffer(img, quality=random.randint(40, 95))
        return Image.open(buf).convert("RGB")
    except Exception:  # noqa: BLE001 - augmentation must never break a batch
        return img


def _jpeg_buffer(img: Image.Image, quality: int):
    import io

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return buf


def eval_tf(size: int = 224, crop_pct: float = CROP_PCT) -> T.Compose:
    """Deterministic eval transform -- identical to the exported model / device."""
    short = resize_short_for(size, crop_pct)
    return T.Compose([
        T.Resize(short, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.CenterCrop(size),
        T.ToTensor(),
    ])


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def _labels_to_idx(series: pd.Series) -> np.ndarray:
    """Map a label column (class names *or* indices) to ``int64`` indices.

    Accepts either representation because ``pandas>=3`` infers a dedicated Arrow
    string dtype, so the old ``series.dtype == object`` check silently took the
    integer branch and raised ``ValueError: invalid literal for int()`` on every
    string-labelled dataset.
    """
    s = pd.Series(series)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("int64").to_numpy()
    mapped = s.astype(str).str.strip().str.upper().map(C2I)
    if mapped.isna().any():
        bad = sorted(set(s.astype(str)[mapped.isna()]))
        raise ValueError(f"unknown class labels {bad}; expected {CLASSES}. "
                         f"Remap with ml_cv_prepare_data --map")
    return mapped.astype("int64").to_numpy()


class ImgDS(Dataset):
    """Yields **un-normalised** ``[0,1]`` float32 NCHW tensors.

    Normalisation happens on the GPU because the teacher and student backbones use
    different ImageNet statistics; baking it into the dataset would duplicate data on
    disk and make the two paths drift apart.
    """

    def __init__(self, df: pd.DataFrame, size: int = 224, train: bool = False, *,
                 policy: PreprocessPolicy | None = None, aug_strength: str = "medium",
                 path_col: str = "path", label_col: str = "label",
                 return_index: bool = False):
        if path_col not in df.columns:
            raise KeyError(f"index csv must contain a '{path_col}' column; got {list(df.columns)}")
        self.paths = df[path_col].astype(str).tolist()
        if label_col in df.columns:
            self.y = _labels_to_idx(df[label_col])
        else:  # inference set
            self.y = np.zeros(len(self.paths), dtype=np.int64)
        out_of_range = sorted(set(self.y.tolist()) - set(range(NUM_CLASSES)))
        if out_of_range:
            raise ValueError(f"labels outside CLASSES: {out_of_range}; check prepare_data --map")
        self.policy = policy or eval_policy(size)
        self.tf = train_tf(size, strength=aug_strength) if train else eval_tf(size, self.policy.crop_pct)
        self.size = size
        self.return_index = return_index
        self.train = train

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, i: int) -> Image.Image:
        p = self.paths[i]
        try:
            img = Image.open(p)
            # JPEG DCT-scaled decode: a 1600 px photo is decoded straight to roughly
            # `size`, which is a 10-25x bandwidth saving on the dataloader workers.
            if img.mode in ("RGB", "L", "CMYK"):
                img.draft("RGB", (self.size * 2, self.size * 2))
            img = img.convert("RGB")
        except Exception as exc:  # noqa: BLE001
            LOG.warning("unreadable image %s (%s); substituting grey", p, exc)
            img = Image.new("RGB", (self.size, self.size), (128, 128, 128))
        return img

    def __getitem__(self, i: int):
        img = self._load(i)
        try:
            x = self.tf(img)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("transform failed for %s (%s); substituting grey", self.paths[i], exc)
            x = eval_tf(self.size, self.policy.crop_pct)(
                Image.new("RGB", (self.size, self.size), (128, 128, 128)))
        out = (x, int(self.y[i]))
        return (*out, i) if self.return_index else out


def make_loader(df: pd.DataFrame, size: int = 224, train: bool = False, *, bs: int = 64,
                workers: int = 4, seed: int = 42, policy: PreprocessPolicy | None = None,
                aug_strength: str = "medium", return_index: bool = False,
                drop_last: bool | None = None, prefetch_factor: int | None = None) -> DataLoader:
    """Seeded, memory-pinned loader.

    ``drop_last`` defaults to True for training: with class-weighted loss the final
    short batch skews the BatchNorm statistics and the weighted gradient.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    ds = ImgDS(df, size, train, policy=policy, aug_strength=aug_strength,
               return_index=return_index)
    if workers <= 0:
        prefetch_factor = None
    elif prefetch_factor is None:
        prefetch_factor = 4 if train else 2
    return DataLoader(
        ds, batch_size=bs, shuffle=train, num_workers=max(0, workers),
        pin_memory=True, drop_last=bool(train) if drop_last is None else drop_last,
        persistent_workers=workers > 0, generator=g if train else None,
        prefetch_factor=prefetch_factor,
    )


def norm_tensors(mean: Sequence[float], std: Sequence[float], dev: torch.device
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    m = torch.tensor(list(mean), dtype=torch.float32, device=dev).view(1, 3, 1, 1)
    s = torch.tensor(list(std), dtype=torch.float32, device=dev).view(1, 3, 1, 1)
    return m, s


def normalise(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


# --------------------------------------------------------------------------- #
# Mixed precision helpers (safe on CPU and on GPUs without bf16)
# --------------------------------------------------------------------------- #
def autocast_ctx(device: torch.device, dtype: torch.dtype | None, enabled: bool = True):
    """``torch.autocast`` that degrades to a no-op on CPU or when dtype is None."""
    import contextlib

    if not enabled or dtype is None or getattr(device, "type", "cpu") != "cuda":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=dtype)


class ModelEma:
    """Exponential moving average of model weights **and** buffers.

    Buffers (BatchNorm running stats) are copied rather than averaged -- averaging a
    running variance produces NaNs and is a common EMA bug.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998, device: torch.device | None = None):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0,1)")
        self.decay = float(decay)
        self.module = _clone_eval(model, device)
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / 2000))  # warm-up ramp
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            src = msd[k].detach()
            if v.dtype.is_floating_point and v.is_cuda == src.is_cuda:
                v.mul_(d).add_(src.to(v.device), alpha=1 - d)
            else:
                v.copy_(src.to(v.device))

    def state_dict(self) -> dict:
        return self.module.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.module.load_state_dict(sd)

    def parameters_m(self) -> float:
        return sum(p.numel() for p in self.module.parameters()) / 1e6


def _clone_eval(model: nn.Module, device: torch.device | None) -> nn.Module:
    import copy

    m = copy.deepcopy(model).eval()
    if device is not None:
        m.to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


# --------------------------------------------------------------------------- #
# Mixup / CutMix + soft-target loss
# --------------------------------------------------------------------------- #
def one_hot(y: torch.Tensor, num_classes: int, smoothing: float = 0.0) -> torch.Tensor:
    """``F.one_hot`` with optional label smoothing (train mode only)."""
    off = smoothing / max(1, num_classes - 1) if smoothing > 0 else 0.0
    oh = F.one_hot(y.clamp(0, num_classes - 1), num_classes).float()
    if off:
        oh = oh * (1 - smoothing) + off
    return oh


def mixup_batch(x: torch.Tensor, y: torch.Tensor, *, alpha_mix: float = 0.2, alpha_cut: float = 1.0,
                prob: float = 0.5, switch_prob: float = 0.5, smoothing: float = 0.0,
                generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch-level mixup or CutMix.  Returns ``(x, soft_targets)``.

    * **mixup** regularises by blending two photos; the blended dish has a valid
      label (both parents are in the same food family most of the time).
    * **CutMix** pastes a patch, which keeps spatial statistics intact -- on a spoilage
      task where a small mould patch matters, that is usually the better half.

    CutMix is applied with ``prob`` and mixup with ``1-prob``; ``lam`` comes from a
    Beta draw so the expected mix ratio is 0.5 in both cases.
    """
    num_classes = y.shape[1] if y.dim() > 1 else NUM_CLASSES
    y2 = y if y.dim() > 1 else one_hot(y, num_classes, smoothing)
    if prob <= 0 or x.shape[0] < 2:
        return x, y2.float()

    def draw(p: float, a: float) -> float:
        r = torch.rand((), generator=generator).item()
        return float(np.random.beta(p, a)) if r < p else 0.0

    use_cut = torch.rand((), generator=generator).item() < switch_prob
    lam = draw(alpha_cut if use_cut else alpha_mix,
               alpha_cut if use_cut else alpha_mix)
    if lam <= 0:
        return x, y2.float()

    perm = torch.randperm(x.shape[0], device=x.device)
    if use_cut:
        _, _, h, w = x.shape
        rh, rw = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
        if rh < 1 or rw < 1:
            return x, y2.float()
        cy, cx = int(torch.randint(0, h - rh + 1, (1,), generator=generator).item()), \
            int(torch.randint(0, w - rw + 1, (1,), generator=generator).item())
        x = x.clone()
        x[:, :, cy:cy + rh, cx:cx + rw] = x[perm][:, :, cy:cy + rh, cx:cx + rw]
        lam = 1 - (rh * rw) / float(h * w)
    else:
        x = x * lam + x[perm] * (1 - lam)
    y_mix = y2 * lam + y2[perm] * (1 - lam)
    return x, y_mix


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, *,
                       class_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Cross entropy against soft targets, with the same class weighting as hard CE.

    Weights are taken from the *dominant* (argmax) class of the mixed target so the
    weighting stays interpretable.
    """
    logp = F.log_softmax(logits, dim=-1)
    loss = -(targets * logp).sum(dim=-1)
    if class_weight is not None:
        w = class_weight.to(logits.device)[targets.argmax(dim=-1)]
        loss = loss * w
    return loss.mean()


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def logits_of(model: nn.Module, dl: DataLoader, dev: torch.device,
              mean: torch.Tensor, std: torch.Tensor, *,
              amp_dtype: torch.dtype | None = None, amp: bool = True,
              return_index: bool = False) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a loader through a model and collect fp32 logits.

    The original version hard-coded ``torch.autocast("cuda", dtype=bfloat16)``, which
    either warns or fails outright on a CPU-only wheel -- and silently disables AMP
    when bf16 is unsupported, so "we trained with mixed precision" was not always true.
    """
    model.eval()
    out, ys, idxs = [], [], []
    for batch in dl:
        x, y = batch[0], batch[1]
        x = x.to(dev, non_blocking=True)
        with autocast_ctx(dev, amp_dtype, amp and getattr(dev, "type", "cpu") == "cuda"):
            lg = model(normalise(x, mean, std))
        out.append(lg.float().cpu())
        ys.append(y)
        if return_index:
            idxs.append(batch[2])
    logits = torch.cat(out).numpy()
    y = torch.cat(ys).numpy()
    if return_index:
        return logits, y, torch.cat(idxs).numpy()
    return logits, y


def softmax_np(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - np.nanmax(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# --------------------------------------------------------------------------- #
# Calibration + operating point
# --------------------------------------------------------------------------- #
def temperature_scale(logits: np.ndarray, y: np.ndarray, *,
                      max_iter: int = 100, lr: float = 0.1) -> float:
    """Single-parameter temperature scaling fitted with LBFGS on NLL.

    Clamps ``log_T`` so a degenerate fit cannot explode to inf and produce NaNs at
    inference (the original version had no clamp).
    """
    lg = torch.tensor(np.asarray(logits), dtype=torch.float32)
    yy = torch.tensor(np.asarray(y), dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = F.cross_entropy(lg / log_t.clamp(-4.0, 4.0).exp(), yy)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().clamp(-4.0, 4.0).exp())


def pick_threshold(probs: np.ndarray, y: np.ndarray, target: float = 0.98) -> float:
    """Threshold on ``s = 1 - P(GOOD)`` giving >= ``target`` recall of truly unsafe rows.

    ``s >= tau`` means "not auto-accepted".  ``method="lower"`` deliberately rounds the
    quantile *down*, which can only increase recall -- rounding up would silently break
    the 0.98 contract on ties.

    Raises when the split contains no unsafe rows: silently returning 1.0 would make
    every downstream report claim 100 % recall on a degenerate set.
    """
    probs, y = np.asarray(probs), np.asarray(y)
    if probs.ndim == 1:
        probs = np.column_stack([1 - probs, probs])
    s = 1 - probs[:, GOOD]
    pos = np.isin(y, UNSAFE_IDX)
    if not pos.any():
        raise ValueError("no unsafe samples to calibrate on; refusing to invent a threshold")
    return float(np.quantile(s[pos], 1 - target, method="lower"))


def pick_reject_threshold(probs: np.ndarray, y: np.ndarray, target: float = 0.90) -> float:
    """Upper edge of the RISK zone: ``s = 1 - P(GOOD)`` at or above which we REJECT.

    Calibrated like :func:`pick_threshold`, but on the rows that are truly ``REJECTED``
    (fully spoiled): ``tau_reject`` is the ``1 - target`` quantile of their scores, so at
    least ``target`` of them land in the hard-reject zone.  ``method="lower"`` rounds
    down, which can only widen the reject zone -- the safe direction, same reasoning as
    in :func:`pick_threshold`.

    Raises when the split contains no REJECTED rows: without them there is no evidence
    for where "clearly bad" starts, and inventing a number would silently convert every
    HOLD into a REJECT (or the reverse) on the serving path.
    """
    probs, y = np.asarray(probs), np.asarray(y)
    if probs.ndim == 1:
        probs = np.column_stack([1 - probs, probs])
    s = 1 - probs[:, GOOD]
    rej = y == C2I["REJECTED"]
    if not rej.any():
        raise ValueError("no REJECTED samples to calibrate on; refusing to invent tau_reject")
    return float(np.quantile(s[rej], 1 - target, method="lower"))


def decide(probs: np.ndarray, tau: float, tau_reject: float | None = None) -> np.ndarray:
    """Three-way serving decision from the model's probabilities: the RISK middle zone.

    The model is trained on four classes, but the *serving* rule the pipeline ships is
    built on the release score ``s = 1 - P(GOOD)`` (see :func:`pick_threshold`).  This
    adds the middle band the product asked for: a batch whose score is neither
    confidently releasable nor confidently rejectable is **RISK -- hold for a human**,
    not silently forced into GOOD or REJECTED.

    Per row, in order:

    * ``s < tau``                                    -> ``auto_accept`` (GOOD, eligible)
    * ``tau_reject is not None and s >= tau_reject`` -> ``reject``       (clearly bad)
    * otherwise                                      -> ``risk_hold``    (RISK, review)

    With ``tau_reject=None`` (or ``tau_reject <= tau``) the middle band is empty and the
    decision collapses to the existing binary hold/release rule -- so every caller that
    has not opted in keeps its exact previous behaviour.  The binary hold flag is simply
    ``action != "auto_accept"``; the safety contract never depends on the band.
    """
    probs = np.asarray(probs)
    if probs.ndim == 1:
        probs = np.column_stack([1 - probs, probs])
    s = 1 - probs[:, GOOD]
    action = np.full(len(s), "risk_hold", dtype=object)
    if tau_reject is not None and float(tau_reject) > float(tau):
        action[s >= float(tau_reject)] = "reject"
    action[s < float(tau)] = "auto_accept"
    return action


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, *, bins: int = 15) -> float:
    """Binary ECE on the max predicted class probability."""
    p = np.asarray(probs)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == np.asarray(y)).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi) if lo > 0 else (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(ece)


def evaluate(probs: np.ndarray, y: np.ndarray, tau: float, *,
             tau_reject: float | None = None,
             classes: Sequence[str] = tuple(CLASSES)) -> dict:
    """Full metric block for one split.

    Reported first-class, because the other numbers do not protect a diner:
    ``unsafe_recall`` (sensitivity), ``false_accepts`` (absolute count of unsafe batches
    that would have been released) and ``good_auto_accept_rate`` (how much good food we
    waste holding for a human).  ``macro_f1`` is secondary.
    """
    probs = np.asarray(probs)
    y = np.asarray(y)
    if probs.ndim == 1:
        probs = np.column_stack([1 - probs, probs])
    s = 1 - probs[:, GOOD]
    flagged = s >= tau
    action = decide(probs, tau, tau_reject)
    pred = probs.argmax(axis=1)
    pos = np.isin(y, UNSAFE_IDX)
    good = y == GOOD
    nf = y == C2I["NOT_FOOD"]
    risk = y == C2I["RISK"]
    out = {
        "n": int(len(y)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "accuracy": float((pred == y).mean()) if len(y) else None,
        "unsafe_recall": float(flagged[pos].mean()) if pos.any() else None,
        "false_accepts": int((~flagged[pos]).sum()) if pos.any() else None,
        "good_auto_accept_rate": float((~flagged[good]).mean()) if good.any() else None,
        "not_food_recall": float((pred[nf] == NOT_FOOD).mean()) if nf.any() else None,
        "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean()) if len(y) else None,
        "ece": expected_calibration_error(probs, y) if len(y) else None,
        "tau": float(tau),
        "tau_reject": None if tau_reject is None else float(tau_reject),
        "actions": {a: int((action == a).sum())
                    for a in ("auto_accept", "risk_hold", "reject")},
        "risk_hold_rate": float((action == "risk_hold").mean()) if len(y) else None,
        "risk_zone_recall": float((action[risk] == "risk_hold").mean()) if risk.any() else None,
        "reject_recall": (float((action[y == C2I["REJECTED"]] == "reject").mean())
                          if (y == C2I["REJECTED"]).any() else None),
        "per_class_recall": {
            classes[i]: float((pred[y == i] == i).mean()) if (y == i).any() else None
            for i in range(len(classes))
        },
        "confusion": _confusion(y, pred, classes),
    }
    out["meets_recall_target"] = (
        None if out["unsafe_recall"] is None else bool(out["unsafe_recall"] >= 0.98)
    )
    return out


def _confusion(y: np.ndarray, pred: np.ndarray, classes: Sequence[str]) -> dict:
    """Dense confusion matrix as a dict-of-dicts.

    The previous implementation used ``pd.crosstab(...).to_dict()``, which introduces
    NaN whenever a class is absent from one of the two vectors and, because
    ``json.dumps`` happily emits ``NaN``, produced a metrics.json that Go and Dart both
    fail to parse.
    """
    k = len(classes)
    mat = [[0 for _ in range(k)] for _ in range(k)]
    for t, p in zip(y.tolist(), pred.tolist()):
        if 0 <= t < k and 0 <= p < k:
            mat[t][p] += 1
    return {classes[t]: {classes[p]: mat[t][p] for p in range(k)} for t in range(k)}


def selection_score(m: dict, *, w_f1: float = 1.0, w_recall: float = 4.0,
                    w_accept: float = 1.0) -> float:
    """Single scalar for checkpoint selection.

    Macro-F1 alone is the wrong objective here: a checkpoint can hold F1 by starting
    to flag *everything*, which is safe but wastes good food.  This rewards unsafe
    recall hardest (it is the safety contract), then the good-food auto-accept rate,
    then F1.  Selecting on this is what stops the trainer from parking on a degenerate
    all-RISK solution.
    """
    r = m.get("unsafe_recall")
    a = m.get("good_auto_accept_rate")
    f1 = m.get("macro_f1")
    return float(w_f1 * (f1 or 0.0) + w_recall * (r if r is not None else 0.0)
                 + w_accept * (a if a is not None else 0.0))


def pick_best(candidates: Iterable[dict], **kw) -> dict | None:
    """Argmax over ``selection_score``; ``None`` for an empty iterable."""
    best, best_s = None, -math.inf
    for c in candidates:
        s = selection_score(c, **kw)
        if s > best_s:
            best, best_s = c, s
    return best


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
CHECKPOINT_KEYS = ("epoch", "best_score", "global_step", "model", "ema", "optimizer",
                   "scheduler", "scaler", "rng", "config", "history", "created")


def save_checkpoint(path, *, model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler=None, scaler=None, ema: "ModelEma | None" = None,
                    epoch: int = 0, best_score: float = -math.inf, global_step: int = 0,
                    config: dict | None = None, history: list | None = None,
                    model_state: dict | None = None) -> None:
    """Write a **complete** resumable checkpoint.

    Everything needed to continue bit-for-bit is in here: model + EMA weights,
    optimiser moments, LR schedule position, AMP gradient scaler, the global step,
    every RNG state (python / numpy / torch CPU / torch CUDA) and the run config.
    Saving only ``state_dict`` -- as the previous version did -- means a resumed run
    restarts the LR schedule and the Adam moments, and silently trains a different
    model than the one you interrupted.
    """
    from ml_utils import atomic_torch_save

    ckpt = {
        "format_version": 2,
        "epoch": int(epoch),
        "best_score": float(best_score),
        "global_step": int(global_step),
        "model": model_state if model_state is not None else model.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": capture_rng_state(),
        "config": dict(config or {}),
        "history": list(history or []),
    }
    atomic_torch_save(ckpt, path)


def capture_rng_state() -> dict:
    """Snapshot every RNG so ``--resume`` is a true continuation.

    NumPy's state is flattened to plain Python lists on purpose: a checkpoint is
    loaded with ``weights_only=True``, and the raw ``np.random.get_state()`` tuple
    contains a NumPy array, which makes the safe unpickler reject the whole file.
    Keeping the checkpoint free of pickled NumPy globals means it also survives a
    NumPy major-version bump.
    """
    import random as _random

    name, keys, pos, has_gauss, cached = np.random.get_state()
    state: dict[str, Any] = {
        "python": _random.getstate(),
        "numpy": {"name": name, "keys": [int(v) for v in keys], "pos": int(pos),
                  "has_gauss": int(has_gauss), "cached_gaussian": float(cached)},
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict | None) -> None:
    """Best-effort RNG restore.  Missing keys are tolerated (older checkpoints)."""
    if not state:
        return
    import random as _random

    with _suppress():
        _random.setstate(state["python"])
    ns = state.get("numpy")
    if isinstance(ns, dict):
        with _suppress():
            np.random.set_state((ns["name"], np.array(ns["keys"], dtype=np.uint32),
                                 ns["pos"], ns["has_gauss"], ns["cached_gaussian"]))
    elif ns is not None:  # legacy raw tuple
        with _suppress():
            np.random.set_state(ns)
    with _suppress():
        torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu") else state["torch"])
    with _suppress():
        if torch.cuda.is_available() and state.get("cuda"):
            torch.cuda.set_rng_state_all(state["cuda"])


class _suppress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True


def load_checkpoint(path, *, model: nn.Module, optimizer: torch.optim.Optimizer | None = None,
                    scheduler=None, scaler=None, ema: ModelEma | None = None,
                    strict: bool = True) -> dict:
    """Restore a checkpoint written by :func:`save_checkpoint`.

    Returns the metadata so the caller can resume the epoch loop at ``ckpt["epoch"] + 1``.
    Missing optimiser/scheduler state is not fatal -- it is logged, and the caller
    decides whether to continue (a first resume from a weights-only file still works).
    """
    from ml_utils import torch_load

    ckpt = torch_load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=strict)
    if ema is not None and ckpt.get("ema"):
        ema.load_state_dict(ckpt["ema"])
    if optimizer is not None and ckpt.get("optimizer"):
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except ValueError as exc:  # param-group mismatch after a CLI change
            LOG.warning("optimizer state not restored (%s); continuing with fresh moments", exc)
    if scheduler is not None and ckpt.get("scheduler"):
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except (ValueError, KeyError) as exc:
            LOG.warning("scheduler state not restored (%s); schedule restarts", exc)
    if scaler is not None and ckpt.get("scaler"):
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except ValueError as exc:
            LOG.warning("GradScaler state not restored (%s)", exc)
    restore_rng_state(ckpt.get("rng"))
    return {"epoch": int(ckpt.get("epoch", -1)),
            "best_score": float(ckpt.get("best_score", -math.inf)),
            "global_step": int(ckpt.get("global_step", 0)),
            "config": ckpt.get("config", {}),
            "history": ckpt.get("history", [])}


# --------------------------------------------------------------------------- #
# Class weighting
# --------------------------------------------------------------------------- #
def class_weights_from(df: pd.DataFrame, *, device: torch.device | str = "cpu",
                       power: float = 0.5, clip: tuple[float, float] = (0.25, 4.0)
                       ) -> tuple[torch.Tensor, dict]:
    """Inverse-sqrt-frequency weights, normalised to mean 1 and clipped.

    Plain inverse-frequency weighting on a 4-class problem with a rare ``NOT_FOOD``
    class produces a ~40x weight that the optimiser then over-fits; ``power=0.5`` plus a
    clip keeps the correction while leaving the common classes learnable.
    """
    counts = (df["label"].value_counts().reindex(CLASSES).fillna(0) + 1).to_numpy(float)
    w = (counts.sum() / (len(CLASSES) * counts)) ** power
    w = np.clip(w, *clip)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device), dict(zip(CLASSES, w.tolist()))


def dataset_balance(df: pd.DataFrame) -> dict:
    """Class histogram + imbalance ratio, logged before every run."""
    counts = df["label"].value_counts().reindex(CLASSES).fillna(0).astype(int)
    n = int(counts.sum())
    rare = int(counts.min()) if n else 0
    return {"n": n, "counts": {c: int(v) for c, v in counts.items()},
            "imbalance_ratio": (float(counts.max()) / max(rare, 1)) if n else None,
            "missing_classes": [c for c in CLASSES if counts[c] == 0]}


# --------------------------------------------------------------------------- #
# Convenience: build a model + its preprocessing contract
# --------------------------------------------------------------------------- #
def build_model(arch: str = "tf_efficientnet_lite0", *, num_classes: int = NUM_CLASSES,
                pretrained: bool = True, drop_rate: float = 0.2, drop_path_rate: float = 0.05,
                size: int = 224):
    """``timm`` backbone + linear head, with the preprocessing contract resolved."""
    timm = __import__("timm")
    model = timm.create_model(arch, pretrained=pretrained, num_classes=num_classes,
                              drop_rate=drop_rate, drop_path_rate=drop_path_rate)
    cfg = timm.data.resolve_data_config({}, model=model)
    policy = PreprocessPolicy.from_data_config(cfg, size)
    return model, policy


def describe_dataset(index_csv) -> pd.DataFrame:
    df = pd.read_csv(index_csv)
    if "split" not in df.columns:
        raise KeyError(f"{index_csv} has no 'split' column; run ml_cv_prepare_data first")
    return df


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Content hash of the split assignment (paths + labels + split)."""
    cols = [c for c in ("path", "label", "split", "source", "group") if c in df.columns]
    return sha256_json(df.sort_values(cols[0])[cols].to_dict(orient="list"))


def resolve_worker_count(requested: int, *, batch_size: int) -> int:
    """Clamp dataloader workers to something sane for a laptop.

    On Windows ``spawn`` re-imports the dataset per worker, so asking for 16 on a
    75 W laptop trades throughput for thrash.  Cap at 8 and at ``4 * batch_size / 32``.
    """
    import os as _os

    cpu = _os.cpu_count() or 4
    return int(max(0, min(requested if requested >= 0 else min(8, cpu - 1), 8, cpu - 1,
                          max(1, 4 * batch_size // 32))))


def default_output_dir(name: str) -> Any:
    return get_paths().runs / name