"""Shared infrastructure for every stage of the ML pipeline.

Design rules this module exists to enforce:

1.  **One root for every side effect.**  Nothing in the pipeline may write to the
    process CWD.  All data / registry / report paths resolve through :class:`Paths`,
    which is anchored on the repository root (``ml_pipeline/..``) and overridable with
    the ``ML_PIPELINE_ROOT`` environment variable.  This is what makes ``make eval``
    and ``make train`` reproducible regardless of where a developer happens to be.
2.  **Writes are atomic and JSON is always valid.**  ``json.dumps`` emits the bare
    tokens ``NaN`` / ``Infinity`` by default; Go's ``encoding/json`` and ``JSON.parse``
    both reject them.  Every metrics artefact is written with ``allow_nan=False`` after
    sanitising non-finite floats to ``null``, through a temp-file + ``os.replace`` so an
    interrupted run can never leave a truncated ``metrics.json`` behind.
3.  **Runs are reproducible.**  :func:`set_determinism` seeds Python, NumPy and Torch
    (CPU + CUDA), pins cuDNN autotune, and records the resulting fingerprint in the run
    metadata so "we got 94.1 % with seed 42" can actually be re-checked.
4.  **Torch is optional.**  CPU-only stages (LightGBM fusion, logistics, accounting)
    must never pay for importing PyTorch, and must never crash because it is absent.
    Every torch symbol is imported lazily inside the function that needs it.

The pipeline stores a run manifest (``run.json``) next to each model so an artefact
carries its own provenance: git SHA, dependency versions, seed, hyper-parameters and
the hash of the data it was trained on.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import functools
import hashlib
import importlib
import json
import logging
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "LOGGER_NAME",
    "Paths",
    "get_logger",
    "setup_logging",
    "set_determinism",
    "sanitize",
    "save_json",
    "load_json",
    "atomic_write_bytes",
    "atomic_torch_save",
    "atomic_joblib_dump",
    "next_version_dir",
    "latest_version_dir",
    "list_version_dirs",
    "sha256_file",
    "sha256_bytes",
    "sha256_json",
    "git_sha",
    "env_fingerprint",
    "count_parameters",
    "resolve_device",
    "resolve_amp_dtype",
    "cuda_report",
    "human_bytes",
    "try_import",
    "require",
    "configure_torch_performance",
    "linear_warmup_cosine",
    "param_groups",
    "Timer",
    "StageTimer",
]

LOGGER_NAME = "ml_pipeline"

_VERSION_RE = re.compile(r"^v(\d+)$")


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def _force_utf8_streams() -> None:
    """Make stdout/stderr UTF-8 with replacement on a legacy code page.

    Windows consoles still default to cp1252, and third-party code prints emoji and box
    characters without asking: ``torch.onnx`` prints a U+2705 on its verbose path, and a
    ``UnicodeEncodeError`` inside a library's *printer* kills the whole process -- the
    export dies after doing all the work, with a traceback about a checkmark.  Replacement
    characters in a log line are a cosmetic loss; losing the run is not.
    """
    for stream in (sys.stdout, sys.stderr):
        enc = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
        if enc in ("utf8", ""):
            continue
        with contextlib.suppress(Exception):
            stream.reconfigure(encoding="utf-8", errors="replace")


def setup_logging(level: int | str | None = None, *, force: bool = False) -> logging.Logger:
    """Install a single stderr handler on the pipeline logger (idempotent)."""
    if level is None:
        level = os.environ.get("ML_LOG_LEVEL", "INFO")
    _force_utf8_streams()
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers and not force:
        logger.setLevel(str(level).upper())
        return logger
    for h in list(logger.handlers):
        logger.removeHandler(h)
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.setLevel(str(level).upper())
    logger.propagate = False
    with contextlib.suppress(AttributeError):  # keep pytest output readable
        logging.getLogger("filelock").setLevel(logging.WARNING)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME if not name else f"{LOGGER_NAME}.{name}")
    if not logging.getLogger(LOGGER_NAME).handlers:
        setup_logging()
    else:
        _force_utf8_streams()
    return logger


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Paths:
    """Canonical locations for data, artefacts and reports.

    ``root`` is the repository root (``annapurna/``).  Everything the pipeline reads
    or writes hangs off it, so a run started from any working directory lands in the
    same place.
    """

    root: Path

    # -- factories ---------------------------------------------------------- #
    @classmethod
    def resolve(cls, root: str | os.PathLike[str] | None = None) -> "Paths":
        if root is not None:
            base = Path(root).expanduser().resolve()
        elif os.environ.get("ML_PIPELINE_ROOT"):
            base = Path(os.environ["ML_PIPELINE_ROOT"]).expanduser().resolve()
        else:
            base = Path(__file__).resolve().parents[1]
        return cls(root=base)

    # -- directories -------------------------------------------------------- #
    @property
    def pipeline_dir(self) -> Path:
        return Path(__file__).resolve().parent

    @property
    def configs(self) -> Path:
        return self.pipeline_dir

    @property
    def data(self) -> Path:
        return _env_path("ML_DATA_DIR", self.root / "ml_pipeline" / "data")

    @property
    def data_raw(self) -> Path:
        return self.data / "raw"

    @property
    def data_interim(self) -> Path:
        return self.data / "interim"

    @property
    def data_processed(self) -> Path:
        return self.data / "processed"

    @property
    def cv_data(self) -> Path:
        return self.data / "cv"

    @property
    def registry(self) -> Path:
        return _env_path("ML_REGISTRY_DIR", self.root / "registry")

    @property
    def runs(self) -> Path:
        return _env_path("ML_RUNS_DIR", self.pipeline_dir / "runs2")

    @property
    def reports(self) -> Path:
        return self.registry / "reports"

    @property
    def tests_dir(self) -> Path:
        return self.pipeline_dir

    # -- model registry ----------------------------------------------------- #
    def model_dir(self, name: str, version: str | int | None = None) -> Path:
        base = self.registry / "models" / name
        return base if version is None else base / _norm_version(version)

    def latest_model_dir(self, name: str) -> Path:
        return latest_version_dir(str(self.model_dir(name)))

    # -- helpers ------------------------------------------------------------ #
    def ensure(self, *dirs: Path) -> Path:
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
        return dirs[0] if dirs else self.root

    def rel(self, p: str | os.PathLike[str]) -> str:
        try:
            return str(Path(p).resolve().relative_to(self.root))
        except ValueError:
            return str(p)

    def __fspath__(self) -> str:  # pragma: no cover - convenience
        return str(self.root)


def _env_path(var: str, default: Path) -> Path:
    return Path(os.environ[var]).expanduser() if os.environ.get(var) else Path(default)


def _norm_version(v: str | int) -> str:
    s = str(v)
    return s if s.startswith("v") else f"v{s}"


def get_paths(root: str | os.PathLike[str] | None = None) -> Paths:
    return Paths.resolve(root)


# --------------------------------------------------------------------------- #
# JSON sanitising + atomic IO
# --------------------------------------------------------------------------- #
def sanitize(obj: Any) -> Any:
    """Recursively convert ``obj`` into something ``json.dumps(allow_nan=False)`` accepts.

    * ``NaN`` / ``±Inf`` (numpy or Python floats) become ``None`` -- the metrics file
      is consumed by Go and by the Flutter app, and neither can parse bare ``NaN``.
    * NumPy scalars / arrays / masked arrays and pandas containers are unwrapped.
    * ``datetime`` / ``date`` / ``Path`` / ``set`` / ``bytes`` get stable reprs.
    * Unknown objects fall back to ``str`` rather than raising, because a metrics
      writer must never be the reason a training run dies.
    """
    if obj is None:
        return None
    # -- fast paths for scalars ------------------------------------------- #
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, str):
        return obj
    # -- containers -------------------------------------------------------- #
    if isinstance(obj, Mapping):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return [sanitize(v) for v in obj.tolist()]
    if isinstance(obj, (list, tuple, set, frozenset)):
        seq = sorted(obj, key=str) if isinstance(obj, (set, frozenset)) else obj
        return [sanitize(v) for v in seq]
    # -- pandas ------------------------------------------------------------ #
    if _is_pandas(obj):
        if hasattr(obj, "to_dict"):
            try:
                return sanitize(obj.to_dict(orient="list") if hasattr(obj, "columns") else obj.to_dict())
            except Exception:  # noqa: BLE001 - never let reporting break a run
                pass
        return sanitize(getattr(obj, "tolist", lambda: obj)())
    # -- datetimes / paths -------------------------------------------------- #
    if isinstance(obj, (dt.datetime, dt.date, dt.time)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", "replace")
    if dataclasses.is_dataclass(obj):
        return sanitize(dataclasses.asdict(obj))
    if hasattr(obj, "item") and callable(obj.item):  # 0-d numpy
        with contextlib.suppress(Exception, ValueError):
            return sanitize(obj.item())
    if hasattr(obj, "__dataclass_fields__"):
        return sanitize(dataclasses.asdict(obj))
    with contextlib.suppress(Exception):
        return sanitize(vars(obj))
    return str(obj)


def _is_pandas(obj: Any) -> bool:
    mod = type(obj).__module__ or ""
    return mod.startswith("pandas")


def save_json(path: str | os.PathLike[str], obj: Any, *, indent: int = 2) -> Path:
    """Atomically write ``obj`` as UTF-8 JSON with no NaN/Infinity tokens."""
    p = Path(path)
    payload = json.dumps(sanitize(obj), indent=indent, sort_keys=False, allow_nan=False,
                         ensure_ascii=False)
    atomic_write_bytes(p, payload.encode("utf-8") + b"\n")
    return p


def load_json(path: str | os.PathLike[str]) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> Path:
    """Write ``data`` to ``path`` via a temp file in the same directory + ``os.replace``.

    ``os.replace`` is atomic on POSIX and on Windows when the target exists, so a
    killed process leaves either the old file or the new one -- never a half-written one.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return p


def atomic_joblib_dump(obj: Any, path: str | os.PathLike[str], *, compress: int = 3) -> Path:
    joblib = require("joblib")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
    os.close(fd)
    try:
        joblib.dump(obj, tmp, compress=compress)
        os.replace(tmp, p)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return p


def atomic_torch_save(obj: Any, path: str | os.PathLike[str]) -> Path:
    """``torch.save`` through a temp file.  Checkpoints are large and a crash during
    the write would otherwise destroy hours of training."""
    torch = require("torch")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, p)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return p


def torch_load(path: str | os.PathLike[str], map_location: str | Any = "cpu") -> Any:
    """``torch.load`` that is safe by default.

    ``weights_only=True`` is the default from torch 2.6 and blocks arbitrary pickle
    execution; we pass it explicitly so the behaviour does not change when someone
    runs2 this on torch 2.5.  A small allowlist is registered first because checkpoints
    written by this pipeline legitimately embed ``torch.torch_version.TorchVersion``
    (from ``torch.__version__`` in the run manifest), which the safe unpickler rejects
    otherwise.  Everything else stays blocked.  Set ``weights_only=False`` only for
    files from an untrusted source.
    """
    torch = require("torch")
    allow = []
    for dotted in ("torch.torch_version.TorchVersion",
                   "numpy.core.multiarray.scalar",
                   "numpy.dtype",
                   "numpy._core.multiarray._reconstruct"):
        try:
            mod, _, attr = dotted.rpartition(".")
            allow.append(getattr(__import__(mod, fromlist=[attr]), attr))
        except Exception:  # noqa: BLE001 - allowlist is best-effort
            continue
    if allow and hasattr(torch.serialization, "safe_globals"):
        try:
            with torch.serialization.safe_globals(allow):
                return torch.load(path, map_location=map_location, weights_only=True)
        except Exception:  # noqa: BLE001 - fall through to the unguarded path below
            pass
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # torch < 2.0 has no weights_only kwarg
        return torch.load(path, map_location=map_location)


# --------------------------------------------------------------------------- #
# Versioned model directories
# --------------------------------------------------------------------------- #
def list_version_dirs(base: str | os.PathLike[str]) -> list[Path]:
    b = Path(base)
    if not b.exists():
        return []
    vs = [p for p in b.iterdir() if p.is_dir() and _VERSION_RE.match(p.name)]
    return sorted(vs, key=lambda p: int(_VERSION_RE.match(p.name).group(1)))


def next_version_dir(base: str | os.PathLike[str]) -> Path:
    """Create and return the next ``vN`` directory under ``base``.

    Idempotent and gap-free: an abandoned ``v7`` from a crashed run is reused rather
    than skipped, so the registry never has a hole in its history.  The original
    implementation did ``mkdir()`` non-existently after computing ``1 + max(...)``,
    which crashed with ``FileExistsError`` whenever two stages started concurrently or
    a partially-created directory was left behind.
    """
    b = Path(base)
    b.mkdir(parents=True, exist_ok=True)
    existing = list_version_dirs(b)
    nxt = (int(_VERSION_RE.match(existing[-1].name).group(1)) + 1) if existing else 1
    while True:  # skip any holes / stale non-version leftovers deterministically
        target = b / f"v{nxt}"
        if not target.exists():
            target.mkdir(parents=True)
            return target
        nxt += 1


def latest_version_dir(base: str | os.PathLike[str]) -> Path:
    """Return the highest-numbered version directory, raising a useful error if empty."""
    vs = list_version_dirs(base)
    if not vs:
        raise FileNotFoundError(
            f"no registered model version under {base!s}. "
            f"Run the corresponding training stage first (see ml_pipeline/ml_README.md)."
        )
    return vs[-1]


# --------------------------------------------------------------------------- #
# Hashing + provenance
# --------------------------------------------------------------------------- #
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str], *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    """Stable hash of a JSON-able object (key-sorted, separators fixed)."""
    return sha256_bytes(json.dumps(sanitize(obj), sort_keys=True, separators=(",", ":"),
                                   allow_nan=False, ensure_ascii=True).encode("utf-8"))


def git_sha(root: str | os.PathLike[str] | None = None, *, short: bool = False) -> str | None:
    """Best-effort ``git rev-parse HEAD``.  Never raises: a missing git must not
    abort a training run."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD" if not short else "--short", "HEAD"],
            cwd=str(root) if root else None, capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return out or None
    except Exception:  # noqa: BLE001
        return None


def env_fingerprint(paths: Paths | None = None) -> dict:
    """Versions and platform for the run manifest.  Never raises."""
    paths = paths or get_paths()
    fp: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "git_sha": git_sha(paths.root),
        "packages": {},
    }
    for mod in ("numpy", "pandas", "sklearn", "scipy", "joblib", "pyarrow", "yaml",
                "lightgbm", "torch", "torchvision", "timm", "onnx", "onnxruntime",
                "ortools", "PIL", "sklearn"):
        try:
            m = importlib.import_module(mod)
            fp["packages"][mod] = getattr(m, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            continue
    fp["cuda"] = cuda_report()
    return fp


# --------------------------------------------------------------------------- #
# Optional-dependency helpers
# --------------------------------------------------------------------------- #
def try_import(name: str, *, extra: str | None = None) -> Any | None:
    """Import ``name`` or return ``None``.  Used for genuinely optional stages."""
    try:
        return importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        log = get_logger()
        log.debug("optional dependency %s unavailable: %s", name, exc)
        return None


def require(name: str, *, extra: str | None = None, hint: str | None = None) -> Any:
    """Import ``name`` or raise a message that says exactly what to install."""
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        req = extra or name
        raise ImportError(
            f"missing dependency '{name}'. Install it with:\n"
            f"    python -m pip install {req}\n"
            f"{('  ' + hint) if hint else ''}"
        ) from exc


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def set_determinism(seed: int = 42, *, deterministic: bool = False) -> dict:
    """Seed every RNG the pipeline touches and return the fingerprint that was applied.

    ``deterministic=True`` forces cuDNN into its reproducible algorithms.  It is off by
    default because it roughly halves throughput and the residual run-to-run
    variation on this workload is far smaller than the effect of the data itself.
    Turn it on for the final promotion run so ``metrics.json`` is bit-reproducible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))

    torch = try_import("torch")
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    return {
        "seed": seed,
        "deterministic": bool(deterministic),
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
    }


# --------------------------------------------------------------------------- #
# Device / dtype / performance
# --------------------------------------------------------------------------- #
def resolve_device(prefer: str = "cuda") -> tuple[Any, str]:
    """Return ``(torch.device, human_reason)``.

    Falls back to CPU with an explicit, loud reason string -- the single most common
    silent failure in this pipeline is a CPU-only torch wheel on a machine that *does*
    have a GPU, which turns a 20-minute run into a six-hour one without any error.
    """
    torch = require("torch")
    if prefer.startswith("cuda") and not torch.cuda.is_available():
        reason = ("CUDA requested but unavailable. If this machine has an NVIDIA GPU you "
                  "installed a CPU-only wheel; reinstall with:\n"
                  "    pip uninstall -y torch torchvision && "
                  "pip install torch torchvision --index-url "
                  "https://download.pytorch.org/whl/cu128")
        return torch.device("cpu"), reason
    if prefer.startswith("cuda"):
        return torch.device(prefer), torch.cuda.get_device_name(0)
    return torch.device("cpu"), "cpu requested"


def resolve_amp_dtype(device: Any) -> Any:
    """bf16 when the GPU supports it, else fp16, else fp32 (disabled AMP).

    On Ada/Blackwell (the target RTX 5050 is sm_120) bf16 needs no ``GradScaler`` and
    does not underflow at fp16's exponent range, which matters because several epochs of
    small-batch fine-tuning can produce gradients well below fp16's normal range.
    """
    torch = require("torch")
    if getattr(device, "type", "cpu") != "cuda":
        return None
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:  # noqa: BLE001
        pass
    return torch.float16


def cuda_report() -> dict:
    """GPU inventory + the memory budget the training loop will respect."""
    torch = try_import("torch")
    if torch is None or not torch.cuda.is_available():
        return {"available": False}
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {
        "available": True,
        "name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "total_vram_gb": round(props.total_memory / 1024**3, 2),
        "bf16_supported": bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()),
        "device_count": torch.cuda.device_count(),
    }


def configure_torch_performance(device: Any, *, vram_budget_gb: float | None = None) -> dict:
    """Turn on the matmul paths that are free on Ampere+ and report the memory budget.

    * TF32 for fp32 matmuls / cuBLAS: ~3x faster on Blackwell, and the accuracy delta is
      far below the run-to-run noise of this dataset.  bf16 autocast is used for the
      actual forward/backward anyway.
    * ``float32_matmul_precision("high")`` covers the paths TF32 does not.
    * ``inference_mode`` is used at call sites, not here.
    """
    torch = require("torch")
    applied: dict[str, Any] = {}
    if getattr(device, "type", "cpu") == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            applied["tf32"] = True
        except Exception as exc:  # noqa: BLE001
            applied["tf32_error"] = str(exc)
    total = None
    if getattr(device, "type", "cpu") == "cuda" and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    applied["vram_total_gb"] = None if total is None else round(total, 2)
    applied["vram_budget_gb"] = vram_budget_gb
    if total is not None and vram_budget_gb is not None and total < vram_budget_gb:
        applied["warning"] = (
            f"vram_budget_gb={vram_budget_gb} exceeds the {total:.1f} GB on this device; "
            f"effective budget clamped"
        )
        applied["vram_budget_gb"] = round(total * 0.9, 2)
    return applied


def count_parameters(module: Any, *, trainable_only: bool = False) -> int:
    ps = module.parameters()
    return sum(p.numel() for p in ps if (p.requires_grad or not trainable_only))


# --------------------------------------------------------------------------- #
# Schedules / optimiser groups
# --------------------------------------------------------------------------- #
def linear_warmup_cosine(step: int, total_steps: int, warmup_steps: int,
                         min_ratio: float = 0.01) -> float:
    """Multiplicative LR factor: linear warmup then cosine decay to ``min_ratio``.

    Pure-python so it is exactly reproducible and unit-testable without a GPU; the
    train loop turns it into a ``torch.optim.lr_scheduler.LambdaLR``.
    """
    if total_steps <= 0:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    if total_steps <= warmup_steps:
        return 1.0
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


def param_groups(model: Any, *, base_lr: float, backbone_prefixes: Sequence[str] = (),
                 head_lr_mult: float = 10.0, weight_decay: float = 0.05,
                 no_decay_keys: Sequence[str] = ("bias",)) -> list[dict]:
    """Discriminative LRs + AdamW decay groups.

    Two ideas that reliably matter for small-data fine-tuning:

    1.  **Head LR multiplier.**  A freshly initialised 4-way head on ~2k photos needs a
        much larger step than the ImageNet backbone; a single LR either destroys the
        pretrained features or never trains the head.
    2.  **No weight decay on 1-D parameters.**  Decaying biases and BatchNorm gains is
        the single most common silent mistake with AdamW; it costs 0.5-1.5 macro-F1 on
        small datasets.
    """
    head_keys, backbone_keys = [], []
    for name, _ in model.named_parameters():
        (head_keys if not any(name.startswith(p) or f".{p}" in name for p in backbone_prefixes)
         else backbone_keys).append(name)
    groups: list[dict] = []
    for keys, lr, tag in ((head_keys, base_lr * head_lr_mult, "head"),
                          (backbone_keys, base_lr, "backbone")):
        if not keys:
            continue
        decay = [n for n in keys if not any(k in n for k in no_decay_keys)]
        no_decay = [n for n in keys if any(k in n for k in no_decay_keys)]
        if decay:
            groups.append({"params": [dict(model.named_parameters())[n] for n in decay],
                           "lr": lr, "weight_decay": weight_decay, "group_name": f"{tag}_decay"})
        if no_decay:
            groups.append({"params": [dict(model.named_parameters())[n] for n in no_decay],
                           "lr": lr, "weight_decay": 0.0, "group_name": f"{tag}_no_decay"})
    return groups


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
class Timer:
    """``with Timer() as t: ...`` then read ``t.seconds`` / log ``t.log(msg)``."""

    def __init__(self, label: str = "", logger: logging.Logger | None = None):
        self.label = label
        self.logger = logger or get_logger()
        self._t0 = 0.0
        self.seconds = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self._t0
        if self.label:
            self.logger.info("%s took %.1fs", self.label, self.seconds)

    def log(self, msg: str, *args) -> None:
        self.logger.info("%s (%.1fs) %s", self.label or "step", self.seconds,
                         msg % args if args else msg)


class StageTimer:
    """Accumulates per-stage wall time into a dict that lands in ``metrics.json``.

    Reproducibility claims need the machine's speed in the record: "trained in 41 min on
    an RTX 5050" is a claim a judge can check.
    """

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self._stack: list[tuple[str, float]] = []

    @contextlib.contextmanager
    def stage(self, name: str):
        self._stack.append((name, time.perf_counter()))
        try:
            yield
        finally:
            label, t0 = self._stack.pop()
            dt = time.perf_counter() - t0
            self.totals[label] = round(self.totals.get(label, 0.0) + dt, 3)

    def as_dict(self) -> dict:
        return dict(sorted(self.totals.items(), key=lambda kv: -kv[1]))


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1024.0
    return f"{n:.1f}PiB"


# --------------------------------------------------------------------------- #
# Small generic helpers used across stages
# --------------------------------------------------------------------------- #
def sha256_of_frame(df: Any, *, columns: Iterable[str] | None = None) -> str:
    """Content hash of a dataframe, for recording which data produced a model."""
    pd = try_import("pandas")
    if pd is None:
        return sha256_json(list(df.columns))
    cols = list(columns) if columns is not None else list(df.columns)
    payload = df.sort_values(list(df.columns[: min(2, len(df.columns))]) or None)
    if cols:
        payload = payload[cols]
    hasher = hashlib.sha256()
    for chunk in np.array_split(payload.to_numpy(dtype=object), max(1, len(payload) // 50_000)):
        hasher.update(repr(chunk.tolist()).encode("utf-8"))
    return hasher.hexdigest()[:32]


def copy_tree(src: Path, dst: Path, *, overwrite: bool = False) -> Path:
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} already exists (pass overwrite=True to replace)")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def format_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> str:
    """Render a list of dicts as a fixed-width table for console reports."""
    if not rows:
        return "(no rows)"
    cols = list(columns) if columns else list(rows[0].keys())
    cells = [[_fmt_cell(r.get(c)) for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    out = ["  ".join(c.ljust(widths[i]) for i, c in enumerate(cols)),
           "  ".join("-" * w for w in widths)]
    out += ["  ".join(v.ljust(widths[i]) for i, v in enumerate(row)) for row in cells]
    return "\n".join(out)


def _fmt_cell(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return "n/a" if not math.isfinite(v) else f"{v:.4g}"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


#: ASCII-only status marks. The Windows console is cp1252 by default, so an emoji or a
#: block-drawing glyph here raises UnicodeEncodeError and kills a run that had already
#: finished its work. Deliberately plain -- see ``_force_utf8_streams``.
MARK_PASS = "[PASS]"
MARK_FAIL = "[FAIL]"
MARK_WARN = "[WARN]"


def gate_verdict(passed: bool | None) -> str:
    """``PASS`` / ``FAIL`` / ``WARN`` for a metric that may be ``None`` (not measured)."""
    if passed is None:
        return MARK_WARN
    return MARK_PASS if passed else MARK_FAIL


def format_scorecard(gates: Sequence[Mapping[str, Any]], title: str = "SCORECARD") -> str:
    """Console scorecard for gate checks: one row per metric, verdict on the right.

    ``gates`` is a sequence of ``{"metric", "value", "target", "pass"}``.  A gate with
    ``pass=None`` renders as WARN, not FAIL -- an unmeasured gate is a different
    statement from a failed one, and collapsing the two is how a run ends up reporting
    "all green" over a metric nobody actually computed.

    The verdict is printed from the same ``pass`` flag that gates the artefact, so the
    console and the JSON can never disagree.
    """
    rows: list[dict[str, Any]] = []
    for g in gates:
        rows.append({
            "metric": g.get("metric"),
            "value": g.get("value"),
            "target": g.get("target"),
            "verdict": gate_verdict(g.get("pass")),
        })
    n_pass = sum(1 for g in gates if g.get("pass") is True)
    n_fail = sum(1 for g in gates if g.get("pass") is False)
    n_warn = sum(1 for g in gates if g.get("pass") is None)
    line = f"{title}: {n_pass}/{len(gates)} pass"
    if n_fail:
        line += f", {n_fail} FAIL"
    if n_warn:
        line += f", {n_warn} not measured"
    return f"{line}\n" + format_table(rows, ["metric", "value", "target", "verdict"])


@functools.lru_cache(maxsize=1)
def _torch_cache_env() -> str:  # pragma: no cover - convenience for data caches
    os.environ.setdefault("HF_HOME", str(Paths.resolve().data / "hf_cache"))
    os.environ.setdefault("TORCH_HOME", str(Paths.resolve().data / "torch_cache"))
    return os.environ["HF_HOME"]