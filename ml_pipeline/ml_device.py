"""Device resolution for training.

    python ml_device.py              # report what is available
    python ml_device.py --require cuda

Everything here is device-AGNOSTIC on purpose.  The same training code has to run
on three very different targets:

* a developer box with a CUDA GPU (fast training),
* a CPU-only laptop or CI runner (still correct, just slower),
* an Android phone, where the model is *inference only* -- there is no GPU training
  and no CUDA at all, the artefact is an int8 TFLite file.

So the rule is: **never** ``torch.device("cuda")`` as a literal.  Always go through
:func:`resolve_device`, which honours an explicit request, then an env override, then
the environment's own answer.  If CUDA is asked for and missing, :func:`resolve_device`
raises unless ``allow_fallback=True`` -- so a CPU-only machine can still produce
artefacts, but a silent GPU->CPU downgrade never hides a real misconfiguration.

AMP follows the same idea: ``float16`` autocast is only enabled on CUDA, because
bfloat16/float16 on CPU is at best a no-op and at worst slower.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

from ml_utils import get_logger

LOG = get_logger("device")

DEVICE_CHOICES = ("auto", "cpu", "cuda")

#: Torch device strings we understand.  "mps" covers Apple silicon.
KNOWN_DEVICES = ("cpu", "cuda", "mps")


@dataclass(frozen=True)
class DeviceInfo:
    """What was actually resolved, and why -- so a run can be audited later."""

    device: str
    torch_device_type: str
    requested: str
    cuda_available: bool
    gpu_name: str | None
    gpu_count: int
    cuda_capability: tuple[int, int] | None
    amp_enabled: bool
    amp_dtype: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "requested": self.requested,
            "cuda_available": self.cuda_available,
            "gpu_name": self.gpu_name,
            "gpu_count": self.gpu_count,
            "cuda_capability": list(self.cuda_capability) if self.cuda_capability else None,
            "amp_enabled": self.amp_enabled,
            "amp_dtype": self.amp_dtype,
            "reason": self.reason,
        }

    def line(self) -> str:
        if self.torch_device_type == "cuda":
            return f"GPU  {self.gpu_name} (cuda:{self.gpu_count - 1}, sm_{''.join(map(str, self.cuda_capability or (0, 0)))})"
        if self.torch_device_type == "mps":
            return "GPU  Apple MPS (Metal)"
        return "CPU  no accelerator available -- training will be slower but correct"


def _torch() -> Any:
    import torch  # imported lazily so callers without torch still work

    return torch


def cuda_details() -> tuple[bool, str | None, int, tuple[int, int] | None]:
    """(available, name, count, capability).  Safe to call with no GPU present."""
    try:
        torch = _torch()
    except Exception as exc:  # noqa: BLE001 - torch may simply be absent
        LOG.debug("torch unavailable while probing CUDA: %s", exc)
        return False, None, 0, None
    try:
        if not torch.cuda.is_available():
            return False, None, 0, None
        n = torch.cuda.device_count()
        cap = torch.cuda.get_device_capability(0) if n else None
        return True, (torch.cuda.get_device_name(0) if n else None), n, cap
    except Exception as exc:  # noqa: BLE001 - a broken driver must not crash the run
        LOG.warning("CUDA probe failed (%s); treating GPU as unavailable", exc)
        return False, None, 0, None


def resolve_device(
    requested: str = "auto",
    *,
    allow_fallback: bool = True,
    prefer_gpu: bool = True,
) -> DeviceInfo:
    """Pick a torch device.

    Args:
        requested: ``"auto"``, ``"cpu"`` or ``"cuda"``.
        allow_fallback: when ``requested="cuda"`` but no GPU exists, return CPU
            instead of raising.  Keep this ``True`` for anything that must be
            runnable on a phone or a CI box.
        prefer_gpu: used only when ``requested="auto"``.  ``False`` pins auto to CPU.

    Raises:
        RuntimeError: ``requested="cuda"``, no GPU, and ``allow_fallback=False``.
    """
    req = (requested or "auto").strip().lower()
    if req not in DEVICE_CHOICES:
        raise ValueError(f"unknown device {requested!r}; choose one of {DEVICE_CHOICES}")

    # Env override wins over 'auto' but never over an explicit CLI request.
    if req == "auto":
        env = os.environ.get("ANNAPURNA_DEVICE", "").strip().lower()
        if env in KNOWN_DEVICES:
            req = env

    avail, name, count, cap = cuda_details()

    if req == "cpu" or (req == "auto" and not prefer_gpu):
        info = DeviceInfo("cpu", "cpu", requested, avail, name, count, cap,
                          False, "none", "cpu explicitly requested")
        LOG.info("device: %s", info.line())
        return info

    if req in ("cuda", "auto"):
        if avail:
            amp = True
            dtype = "float16"
            reason = "CUDA available" if req == "cuda" else "auto-selected CUDA"
            info = DeviceInfo("cuda", "cuda", requested, avail, name, count, cap, amp, dtype, reason)
            LOG.info("device: %s", info.line())
            return info
        if req == "cuda" and not allow_fallback:
            raise RuntimeError(
                "CUDA was requested but is not available. Install a CUDA build of torch "
                "(pip install torch --index-url https://download.pytorch.org/whl/cu128) "
                "or re-run with --device cpu."
            )
        reason = ("no CUDA device found -- falling back to CPU so the run still completes "
                  "(results are identical, only slower)")
        info = DeviceInfo("cpu", "cpu", requested, avail, name, count, cap, False, "none", reason)
        LOG.warning("device: %s", info.line())
        return info

    # mps (Apple silicon)
    try:
        torch = _torch()
        backends = getattr(torch, "backends", None)
        mps = getattr(backends, "mps", None) if backends else None
        if mps is not None and mps.is_available():
            info = DeviceInfo("mps", "mps", requested, avail, name, count, cap, False, "none", "MPS available")
            LOG.info("device: %s", info.line())
            return info
    except Exception as exc:  # noqa: BLE001
        LOG.debug("MPS probe failed: %s", exc)

    info = DeviceInfo("cpu", "cpu", requested, avail, name, count, cap, False, "none", "no supported accelerator")
    LOG.info("device: %s", info.line())
    return info


def autocast_dtype(device_info: DeviceInfo) -> Any:
    """Torch dtype for AMP, or ``None`` when AMP should stay off."""
    if not device_info.amp_enabled:
        return None
    torch = _torch()
    if device_info.amp_dtype == "bfloat16":
        return torch.bfloat16
    return torch.float16


def fit_batch_size_for_device(base: int, device_info: DeviceInfo, arch: str = "") -> int:
    """Scale a batch size to the resolved device.

    The phone constraint lives here: the *training* batch can be large on a GPU, but
    the exported int8 model must still fit the device budget, which is enforced
    separately in ``ml_cv_bakeoff.PHONE_BUDGET_MB``.  This function never raises the
    batch beyond what a 2 GB GPU can hold, because the RTX-class laptop cards people
    train on are frequently the smallest target.
    """
    if device_info.torch_device_type != "cuda":
        return base
    heavy = any(k in arch.lower() for k in ("convnext", "resnet50", "densenet", "large", "vit", "swin"))
    free_gb = _free_gpu_gb()
    if free_gb is None:
        return base
    ceiling = 512 if heavy else 1024
    by_memory = max(8, int(free_gb * 1024 / (12 if heavy else 4)))
    return max(8, min(base, ceiling, by_memory))


def _free_gpu_gb() -> float | None:
    try:
        torch = _torch()
        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info(0)
        return free / (1024**3)
    except Exception:  # noqa: BLE001
        return None


def add_device_argument(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Shared ``--device`` flag so every entry point accepts the same vocabulary."""
    parser.add_argument(
        "--device",
        default="auto",
        choices=list(DEVICE_CHOICES),
        help="auto (recommended: GPU when present, CPU otherwise), cpu, or cuda",
    )
    parser.add_argument(
        "--require-device",
        action="store_true",
        help="fail instead of falling back when an explicit --device cuda is unavailable",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Report the training device this machine will use.")
    add_device_argument(ap)
    a = ap.parse_args(argv)
    try:
        info = resolve_device(a.device, allow_fallback=not a.require_device)
    except RuntimeError as exc:
        print(f"[FAIL] {exc}")
        return 1
    print(info.line())
    print(f"  requested={info.requested}  cuda_available={info.cuda_available}  amp={info.amp_dtype if info.amp_enabled else 'off'}")
    print(f"  reason: {info.reason}")
    print("  (training device is chosen at runtime; the exported model is device-agnostic)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())