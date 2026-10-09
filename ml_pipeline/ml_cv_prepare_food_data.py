"""Download the food datasets and build ``data/cv/index.csv`` for the food classifier run.

    python ml_cv_prepare_food_data.py               # download + index (default datasets)
    python ml_cv_prepare_food_data.py --list-sources
    python ml_cv_prepare_food_data.py --hf-dataset Project-AgML/fresh_rotten_fruit_classification --config raw
    python ml_cv_prepare_food_data.py --snack-label GOOD      # snacks count as acceptable food
    python ml_cv_prepare_food_data.py --indian-foods-dataset LALIT324rt/indian-foods-dataset

What this does that a naive ``load_dataset + pd.DataFrame`` does not
---------------------------------------------------------------------
1. **Writes images to the pipeline's canonical layout** ``data/cv/raw/<source>/<class>/<file>``
   so the existing ``ml_cv_prepare_data.py`` contract is honoured and the same audit code
   can be reused.
2. **Maps each dataset's native labels into the pipeline's safety classes**
   (``GOOD / RISK / REJECTED / NOT_FOOD``) explicitly, with the mapping printed in the
   audit so a reader can see exactly what landed where.
3. **Deduplicates by perceptual hash and splits by whole source + whole dish group**,
   exactly like ``ml_cv_prepare_data.py`` -- a food model's test number is meaningless
   when a near-duplicate of a training photo leaks into test.
4. **Holds out one whole source for test** so the reported accuracy is a real
   cross-domain number, not a same-kitchen leak. By default the snacks source is the
   unseen test source (the model has never seen a snack during training), which is the
   honest way to claim "generalises to a different food category".

Mapping (configurable, printed in the audit)
-------------------------------------------
* fresh_rotten_fruit_classification: label 0 -> GOOD, label 1 -> REJECTED
* snacks (Matthijs/snacks): each snack class -> NOT_FOOD  (or GOOD with --snack-label GOOD)
* indian-foods (LALIT324rt/indian-foods-dataset): all 15 dish classes -> GOOD

The MM-Food-100K (Kaggle) path is **not** downloaded here -- it needs your Kaggle API key.
Drop ``~/.kaggle/kaggle.json`` in place and pass ``--kaggle-dataset jaisal228/MM-Food-100K``;
the script will ask the driver for the label mapping interactively before downloading.
"""

from __future__ import annotations

import argparse
import enum
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi
from PIL import Image, ImageFile

from ml_cv_common import CLASSES, LOG, PreprocessPolicy
from ml_cv_prepare_data import (
    DUP_DISTANCE,
    add_hashes,
    assign_splits,
    audit_splits,
    drop_near_duplicates,
)
from ml_utils import get_paths, save_json, sha256_of_frame

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 120_000_000  # decompression-bomb guard

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

DEFAULT_HF_FOOD = "Project-AgML/fresh_rotten_fruit_classification"
DEFAULT_HF_FOOD_CONFIG = "augmented"  # 12,335 imgs vs 3,200 for raw
DEFAULT_SNACKS = "Matthijs/snacks"
DEFAULT_INDIAN_FOODS = "LALIT324rt/indian-foods-dataset"
DEFAULT_TEST_SOURCE = "snacks"  # whole source held out for an honest cross-category number


class DownloadStats(enum.Enum):
    OK = "ok"
    SKIP_EXISTS = "skip_exists"
    BAD_IMAGE = "bad_image"
    ERROR = "error"


def _rel_root(paths) -> Path:
    return paths.cv_data / "raw"


def ensure_dir(root: Path, source: str, cls: str) -> Path:
    d = root / source / cls
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_filename(source: str, cls: str, idx: int, ext: str = ".jpg") -> str:
    return f"{source}_img{idx:05d}{ext}"


def _download_single(args: tuple[Path, str, str, Any, int]) -> tuple[str, DownloadStats, str]:
    """Download one image row to ``root/source/cls/file``. Returns (path, status, note)."""
    root, source, cls, row, idx = args
    out_dir = ensure_dir(root, source, cls)
    out_name = _safe_filename(source, cls, idx)
    out_path = out_dir / out_name
    if out_path.exists() and out_path.stat().st_size > 100:
        return str(out_path), DownloadStats.SKIP_EXISTS, ""
    try:
        if row is None:
            return "", DownloadStats.ERROR, "no row passed to worker"
        img = row.get("image", None)
        if img is None:
            return "", DownloadStats.BAD_IMAGE, "null image field"
        if not isinstance(img, Image.Image):
            if hasattr(img, "data"):
                img = Image.open(img)
            else:
                return "", DownloadStats.BAD_IMAGE, f"unsupported image type {type(img).__name__}"
        img = img.convert("RGB")
        # cap size so we do not write 20 MB originals that we downsample anyway at train time
        max_side = 1400
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, format="JPEG", quality=90)
        if out_path.stat().st_size < 50:
            out_path.unlink(missing_ok=True)
            return "", DownloadStats.BAD_IMAGE, "written file too small"
        return str(out_path), DownloadStats.OK, ""
    except Exception as exc:  # noqa: BLE001
        return "", DownloadStats.ERROR, repr(exc)[:160]


def _iter_stream_rows(dataset: str, config: str | None, split: str):
    """Yield (idx, row) from a streaming HF dataset without materialising decoded images.

    Streaming avoids the ``list(ds)`` path that decodes every image in host RAM at once
    (PIL ``Image.core.new`` MemoryError on the 12k-image augmented fruit dataset).
    """
    ds = load_dataset(dataset, name=config, split=split, streaming=True)
    for idx, row in enumerate(ds):
        yield idx, row


def _count_stream_rows(dataset: str, config: str | None, split: str, cap: int = 500000) -> int:
    """Best-effort row count for a streaming dataset (used only for progress %).

    We do **not** decode images here -- just count rows by advancing the iterator.
    For parquet-backed datasets this is cheap; for script-backed it still decodes, so
    we cap it. Returns the smaller of actual count and cap.
    """
    try:
        n = 0
        for _ in _iter_stream_rows(dataset, config, split):
            n += 1
            if n >= cap:
                break
        return n
    except Exception as exc:  # noqa: BLE001
        LOG.warning("row-count estimation failed for %s/%s: %s -- progress %% will be coarse",
                    dataset, config, exc)
        return -1


def _norm_status(status: Any) -> DownloadStats:
    """Normalize whatever a worker returned into a ``DownloadStats`` member."""
    if isinstance(status, DownloadStats):
        return status
    if isinstance(status, str):
        try:
            return DownloadStats(status)
        except ValueError:
            return DownloadStats.ERROR
    return DownloadStats.ERROR


def download_hf_dataset(
    dataset: str,
    config: str | None,
    source: str,
    label_map: dict[str, str],
    root: Path,
    *,
    max_workers: int = 8,
    limit: int | None = None,
    split: str = "train",
    progress_every: int = 200,
    row_count_estimate: int = -1,
) -> pd.DataFrame:
    """Download ``dataset`` (HF) into ``root/<source>/<cls>/`` and return an index frame.

    Images are decoded and written one at a time from a streaming iterator, so host RAM
    stays bounded regardless of dataset size (fixes the ``list(ds)`` MemoryError).

    The work queue carries the already-decoded row object, but we cap concurrency so at
    most ``max_workers`` decoded images are held in flight at once.
    """
    t0 = time.time()
    LOG.info("downloading HF dataset %s (config=%s, split=%s) -> %s",
             dataset, config, split, root)
    n_total = row_count_estimate
    if n_total < 0:
        n_total = _count_stream_rows(dataset, config, split)
        LOG.info("estimated %d rows (limit=%s)", n_total, limit)

    cls_counts: dict[str, int] = {c: 0 for c in CLASSES}
    label_seen: set[str] = set()
    plan: list[tuple[int, str, str, Any]] = []  # (idx, native, cls, row)
    for idx, row in _iter_stream_rows(dataset, config, split):
        if limit is not None and idx >= limit:
            break
        native = str(row.get("label", ""))
        if native not in label_map:
            label_seen.add(native)
            continue
        cls = label_map[native]
        if cls not in CLASSES:
            LOG.warning("label_map sends %r -> %r (not a pipeline class); skipping", native, cls)
            continue
        plan.append((idx, native, cls, row))
        cls_counts[cls] += 1

    if not plan:
        if label_seen:
            LOG.warning("unmapped native labels encountered: %s -- add to label_map",
                        sorted(label_seen))
        raise RuntimeError(f"no rows matched label_map for {dataset} ({config}) split {split}")

    if label_seen:
        LOG.warning("unmapped native labels encountered: %s -- add to label_map",
                    sorted(label_seen))

    LOG.info("%d image rows to download (limit=%s)", len(plan), limit)

    downloaded: dict[str, int] = {s.value: 0 for s in DownloadStats}
    path_by_idx: dict[int, str] = {}
    written_count = 0
    work_done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_to_idx: dict = {}
        plan_iter = iter(plan)

        def _submit_more() -> None:
            for _ in range(max_workers - len(fut_to_idx)):
                try:
                    idx, native, cls, row = next(plan_iter)
                except StopIteration:
                    break
                fut = pool.submit(_download_single, (root, source, cls, row, idx))
                fut_to_idx[fut] = idx

        _submit_more()
        while fut_to_idx:
            future = next(as_completed(fut_to_idx), None)
            if future is None:
                break
            idx = fut_to_idx.pop(future)
            work_done += 1
            try:
                path, status, note = future.result()
            except Exception as exc:  # noqa: BLE001
                status, note = DownloadStats.ERROR, repr(exc)[:160]
                path = ""
            status = _norm_status(status)
            downloaded[status.value] += 1
            if status in (DownloadStats.OK, DownloadStats.SKIP_EXISTS) and path:
                path_by_idx[idx] = path
                written_count += 1
            if work_done % progress_every == 0 or work_done == len(plan):
                elapsed = time.time() - t0
                rate = work_done / max(0.1, elapsed)
                LOG.info("download %.1f%% (%d/%d)  %.0f rows/s  "
                         "ok=%d skip=%d bad=%d err=%d",
                         100 * work_done / max(1, len(plan)), work_done, len(plan), rate,
                         downloaded[DownloadStats.OK.value],
                         downloaded[DownloadStats.SKIP_EXISTS.value],
                         downloaded[DownloadStats.BAD_IMAGE.value],
                         downloaded[DownloadStats.ERROR.value])
            _submit_more()

    if not path_by_idx:
        raise RuntimeError(f"no images written for {dataset} -- check label_map + connectivity")

    # Build the index from what we actually wrote.
    idx_to_native = {idx: native for idx, native, _cls, _row in plan}
    records: list[dict] = []
    for idx, path in sorted(path_by_idx.items()):
        st = Path(path).stat()
        stem = Path(path).stem
        native = idx_to_native.get(idx, "")
        cls = label_map.get(native, "?")
        records.append({
            "path": path,
            "rel_path": Path(path).relative_to(root).as_posix(),
            "source": source,
            "label": cls,
            "group": f"{source}/{stem}",
            "session": f"{source}/{stem}",
            "bytes": st.st_size,
            "mtime": int(st.st_mtime),
        })

    out = pd.DataFrame(records)
    out = out.sort_values(["source", "label", "path"]).reset_index(drop=True)
    elapsed = time.time() - t0
    LOG.info("download complete: %d images in %.1f s (%.0f img/s)",
             len(out), elapsed, len(out) / max(0.1, elapsed))
    LOG.info("class distribution: %s", {c: cls_counts.get(c, 0) for c in CLASSES})
    return out


# --------------------------------------------------------------------------- #
# Snacks via HF repo zip (bypasses the blocked dataset-script loader)
# --------------------------------------------------------------------------- #
def _placeholder_cleanup(root: Path, *sources: str) -> None:
    """Remove any stale ``__placeholder__`` dirs from prior interrupted runs2 so they cannot
    block creation of the real per-class dirs. Tolerant: if cleanup fails, just leave it --
    the class dirs are created by name below regardless.
    """
    import shutil

    for src in sources:
        ph = root / src / "__placeholder__"
        if ph.exists():
            try:
                if ph.is_dir():
                    if any(ph.iterdir()):
                        shutil.rmtree(ph, ignore_errors=True)
                    else:
                        ph.rmdir()
                else:
                    ph.unlink(missing_ok=True)
            except OSError:
                pass


def download_snacks_zip(
    dataset: str,
    source: str,
    label_map: dict[str, str],
    root: Path,
    *,
    max_workers: int = 8,
    limit: int | None = None,
    train_split: str = "train",
) -> pd.DataFrame:
    """Download the ``Matthijs/snacks`` images zip from HF and index it.

    ``datasets`` 5.x no longer runs2 dataset scripts (``snacks.py``), so we cannot use
    ``load_dataset`` for this repo. Instead we pull the repo's ``images.zip`` (110 MB,
    6,749 images across train/val/test) directly via ``hf_hub_download`` and extract the
    train split into ``root/<source>/<class>/``.

    The zip layout is ``data/{train,val,test}/{class}/image.jpg``.
    """
    import shutil
    import zipfile

    from huggingface_hub import hf_hub_download

    t0 = time.time()
    LOG.info("downloading snacks zip %s -> %s", dataset, root)
    zip_path = hf_hub_download(
        repo_id=dataset, filename="images.zip", repo_type="dataset",
        force_download=False, resume_download=True,
    )
    zip_size = Path(zip_path).stat().st_size
    LOG.info("snacks zip cached at %s (%.2f MB)", zip_path, zip_size / 1e6)

    _placeholder_cleanup(root, source)

    # Inspect the zip to build the class -> train-file map without extracting everything.
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist()
                 if not n.startswith("__MACOSX") and not n.endswith("/")]
        # layout: data/{split}/{class}/{file}
        train_files: dict[str, list[tuple[str, str]]] = {}
        class_counts: dict[str, int] = {}
        for n in names:
            parts = n.split("/")
            if len(parts) < 4 or parts[0] != "data" or parts[1] != train_split:
                continue
            cls = parts[2]
            fname = parts[3]
            train_files.setdefault(cls, []).append((n, fname))
            class_counts[cls] = class_counts.get(cls, 0) + 1

    if not train_files:
        raise RuntimeError(
            f"no train-split images found in {dataset} zip -- unexpected layout"
        )

    LOG.info("snacks train split: %d classes, %d images",
             len(train_files), sum(len(v) for v in train_files.values()))

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    records: list[dict] = []
    extracted = 0
    skipped = 0
    errors = 0

    with zipfile.ZipFile(zip_path) as z, ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_to_path: dict = {}

        def _extract(cls: str, zip_name: str, fname: str) -> tuple[str, str, int]:
            out_dir = ensure_dir(root, source, cls)
            out_path = out_dir / fname
            if out_path.exists() and out_path.stat().st_size > 100:
                return out_path.name, "skip_exists", 0
            data = z.read(zip_name)
            if len(data) < 50:
                return "", "bad_image", 0
            out_path.write_bytes(data)
            if out_path.suffix.lower() not in EXT:
                out_path.unlink(missing_ok=True)
                return "", "bad_image", 0
            return out_path.name, "ok", len(data)

        work: list[tuple[str, str, str]] = []
        for cls, files in train_files.items():
            for zip_name, fname in files:
                if limit is not None and len(work) >= limit:
                    break
                work.append((cls, zip_name, fname))

        idx = 0
        while idx < len(work):
            batch_end = min(idx + max_workers, len(work))
            for j in range(idx, batch_end):
                cls, zip_name, fname = work[j]
                fut = pool.submit(_extract, cls, zip_name, fname)
                fut_to_path[fut] = (cls, fname)
            for future in as_completed(fut_to_path):
                cls, fname = fut_to_path.pop(future)
                try:
                    name, status, size = future.result()
                except Exception as exc:  # noqa: BLE001
                    status, size = "error", 0
                    name = ""
                status = _norm_status(status)
                if status == DownloadStats.OK:
                    p = root / source / cls / name
                    records.append({
                        "path": str(p.resolve()),
                        "rel_path": p.relative_to(root).as_posix(),
                        "source": source,
                        "label": label_map.get(cls, "?"),
                        "group": f"{source}/{cls}/{fname}",
                        "session": f"{source}/{cls}/{fname}",
                        "bytes": size,
                        "mtime": int(p.stat().st_mtime),
                    })
                    extracted += 1
                elif status == DownloadStats.SKIP_EXISTS:
                    p = root / source / cls / fname
                    records.append({
                        "path": str(p.resolve()),
                        "rel_path": p.relative_to(root).as_posix(),
                        "source": source,
                        "label": label_map.get(cls, "?"),
                        "group": f"{source}/{cls}/{fname}",
                        "session": f"{source}/{cls}/{fname}",
                        "bytes": p.stat().st_size,
                        "mtime": int(p.stat().st_mtime),
                    })
                    skipped += 1
                else:
                    errors += 1
                if (extracted + skipped) % 500 == 0 or (extracted + skipped) == len(work):
                    elapsed = time.time() - t0
                    total_so_far = extracted + skipped
                    LOG.info("snacks extract %.1f%% (%d/%d)  %.0f img/s  ok=%d skip=%d err=%d",
                             100 * total_so_far / max(1, len(work)),
                             total_so_far, len(work), total_so_far / max(0.1, elapsed),
                             extracted, skipped, errors)
            idx = batch_end

    if not records:
        raise RuntimeError(f"no snacks images extracted -- check zip layout + label_map")

    out = pd.DataFrame(records)
    out = out.sort_values(["source", "label", "path"]).reset_index(drop=True)
    elapsed = time.time() - t0
    LOG.info("snacks download+extract complete: %d images in %.1f s (%.0f img/s)",
             len(out), elapsed, len(out) / max(0.1, elapsed))
    LOG.info("class distribution: %s", {c: class_counts.get(c, 0) for c in CLASSES})
    return out


# --------------------------------------------------------------------------- #
# Indian foods via streaming HF dataset (LALIT324rt/indian-foods-dataset)
# --------------------------------------------------------------------------- #
INDIAN_FOODS_SOURCE = "indian_foods"

# LALIT324rt/indian-foods-dataset is parquet-backed with an ``image`` column and a
# ``label`` column that is a ``ClassLabel``. In this environment the streaming loader emits
# the label as an **integer string** ("0".."14"), not the dish name, so the working label
# map must be keyed by those int strings.
#
# The class order is taken from the dataset README / ClassLabel.names:
#   0  biryani
#   1  cholebhature
#   2  dabeli
#   3  dal
#   4  dhokla
#   5  dosa
#   6  jalebi
#   7  kathiroll
#   8  kofta
#   9  naan
#   10 pakora
#   11 paneer
#   12 panipuri
#   13 pavbhaji
#   14 vadapav
#
# If the HF repo ever reorders these classes, the mapping below would silently assign the
# wrong dish-to-class mapping. The audit printout still reports the human-readable dish names
# via INDIAN_FOODS_NAME_BY_ID so a reader can verify the assignment.
INDIAN_FOODS_NAME_BY_ID = [
    "biryani", "cholebhature", "dabeli", "dal", "dhokla", "dosa", "jalebi",
    "kathiroll", "kofta", "naan", "pakora", "paneer", "panipuri", "pavbhaji", "vadapav",
]
INDIAN_FOODS_MAP = {str(i): "GOOD" for i in range(len(INDIAN_FOODS_NAME_BY_ID))}


def download_indian_foods_dataset(
    dataset: str,
    source: str,
    label_map: dict[str, str],
    root: Path,
    *,
    max_workers: int = 8,
    limit: int | None = None,
    split: str = "train",
) -> pd.DataFrame:
    """Download ``LALIT324rt/indian-foods-dataset`` (or any same-schema HF image dataset)
    into ``root/<source>/<cls>/`` and return an index frame.

    This dataset is parquet-backed with columns ``image`` and ``label`` where ``label`` is a
    ClassLabel whose names, in this environment, are streamed as integer strings "0".."14".
    All current classes are correctly-prepared Indian dishes, so they all map to GOOD by
    default -- override via ``label_map`` if you want a different framing.
    """
    return download_hf_dataset(
        dataset=dataset,
        config=None,
        source=source,
        label_map=label_map,
        root=root,
        max_workers=max_workers,
        limit=limit,
        split=split,
    )


# --------------------------------------------------------------------------- #
# Label mappings
# --------------------------------------------------------------------------- #
FRESH_ROTEN_MAP = {"0": "GOOD", "1": "REJECTED"}
SNACKS_MAP_default = {
    "apple": "NOT_FOOD", "banana": "NOT_FOOD", "cake": "NOT_FOOD",
    "candy": "NOT_FOOD", "carrot": "NOT_FOOD", "cookie": "NOT_FOOD",
    "doughnut": "NOT_FOOD", "grape": "NOT_FOOD", "hot dog": "NOT_FOOD",
    "ice cream": "NOT_FOOD", "juice": "NOT_FOOD", "muffin": "NOT_FOOD",
    "orange": "NOT_FOOD", "pineapple": "NOT_FOOD", "popcorn": "NOT_FOOD",
    "pretzel": "NOT_FOOD", "salad": "NOT_FOOD", "strawberry": "NOT_FOOD",
    "waffle": "NOT_FOOD", "watermelon": "NOT_FOOD",
}
SNACKS_CLASSES = sorted(SNACKS_MAP_default.keys())

INDIAN_FOODS_CLASSES = sorted(str(k) for k in INDIAN_FOODS_MAP.keys())


def snacks_label_map(snack_label: str) -> dict[str, str]:
    target = snack_label.upper()
    if target not in CLASSES:
        raise ValueError(f"--snack-label must be one of {CLASSES}, got {snack_label!r}")
    return {k: target for k in SNACKS_CLASSES}


def indian_foods_label_map(target: str = "GOOD") -> dict[str, str]:
    target = target.upper()
    if target not in CLASSES:
        raise ValueError(f"--indian-foods-label must be one of {CLASSES}, got {target!r}")
    return {k: target for k in INDIAN_FOODS_CLASSES}


# --------------------------------------------------------------------------- #
# Kaggle path (manual, because it needs your API key)
# --------------------------------------------------------------------------- #
def kaggle_download(dataset: str, dest: Path) -> Path:
    """Download a Kaggle dataset into ``dest``.

    Requires ``~/.kaggle/kaggle.json`` (or KAGGLE_USERNAME + KAGGLE_KEY env vars) and the
    ``kaggle`` CLI on PATH.  We do **not** ask for your key here -- that stays on your box.
    """
    import shutil
    from subprocess import PIPE, Popen

    if not shutil.which("kaggle"):
        raise RuntimeError(
            "kaggle CLI not on PATH. Install with ``pip install kaggle`` and put your key in "
            "~/.kaggle/kaggle.json (chmod 600). See https://www.kaggle.com/docs/api"
        )
    dest.mkdir(parents=True, exist_ok=True)
    LOG.info("running: kaggle datasets download -d %s -p %s", dataset, dest)
    proc = Popen(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", str(dest), "--unzip"],
        stdout=PIPE, stderr=PIPE, text=True,
    )
    stdout, stderr = proc.communicate(timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"kaggle download failed (rc={proc.returncode}):\n{stderr}")
    # kaggle unzips into ``dest`` directly; find the image tree
    return dest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    grp = ap.add_argument_group("sources")
    grp.add_argument("--hf-dataset", default=DEFAULT_HF_FOOD, help="HF dataset ID for fruit data")
    grp.add_argument("--hf-config", default=DEFAULT_HF_FOOD_CONFIG, help="config name (raw|augmented)")
    grp.add_argument("--hf-split", default="train", help="split to pull")
    grp.add_argument("--download-limit", type=int, default=None, help="cap downloaded images (debug)")
    grp.add_argument("--no-fruit", action="store_true", help="skip the fruit dataset")
    grp.add_argument("--snacks-dataset", default=DEFAULT_SNACKS, help="HF snacks dataset ID")
    grp.add_argument("--snacks-split", default="train", help="snacks split to pull for TRAIN (val/test go to eval)")
    grp.add_argument("--snack-label", default="NOT_FOOD", choices=sorted(CLASSES),
                     help="which pipeline class snack images map to")
    grp.add_argument("--no-snacks", action="store_true", help="skip the snacks dataset")
    grp.add_argument("--indian-foods-dataset", default=DEFAULT_INDIAN_FOODS,
                     help="HF Indian foods dataset ID (default: LALIT324rt/indian-foods-dataset)")
    grp.add_argument("--indian-foods-split", default="train", help="split to pull (train|test)")
    grp.add_argument("--indian-foods-label", default="GOOD", choices=sorted(CLASSES),
                     help="which pipeline class Indian food images map to (default: GOOD)")
    grp.add_argument("--no-indian-foods", action="store_true", help="skip the Indian foods dataset")
    grp.add_argument("--kaggle-dataset", default=None,
                     help="Kaggle dataset ID (e.g. jaisal228/MM-Food-100K). Needs ~/.kaggle/kaggle.json")
    grp.add_argument("--kaggle-root", default=None,
                     help="optional pre-downloaded Kaggle image root (skips the kaggle CLI download)")
    grp.add_argument("--kaggle-map", default=None,
                     help="JSON file mapping Kaggle folder/class names -> pipeline class, e.g. {'prepared':'GOOD','spoiled':'REJECTED'}")
    grp.add_argument("--kaggle-limit", type=int, default=None, help="cap Kaggle images (debug)")

    split_grp = ap.add_argument_group("splits")
    split_grp.add_argument("--test-source", default=DEFAULT_TEST_SOURCE,
                           help="whole source folder held out for test (default: snacks)")
    split_grp.add_argument("--val-frac", type=float, default=0.15)
    split_grp.add_argument("--dup-distance", type=int, default=DUP_DISTANCE)
    split_grp.add_argument("--no-dedupe", action="store_true")
    split_grp.add_argument("--require-unseen-test-source", action="store_true")
    split_grp.add_argument("--seed", type=int, default=42)

    io_grp = ap.add_argument_group("io")
    io_grp.add_argument("--out", default=None, help="index.csv path (default: data/cv/index.csv)")
    io_grp.add_argument("--root", default=None, help="raw image root (default: data/cv/raw)")
    io_grp.add_argument("--max-workers", type=int, default=8)
    io_grp.add_argument("--list-sources", action="store_true", help="print candidate sources and exit")
    return ap


def list_candidate_sources() -> None:
    api = HfApi()
    print("HF datasets considered by default:")
    for ds, cfg, desc in [
        (DEFAULT_HF_FOOD, DEFAULT_HF_FOOD_CONFIG, "fresh/rotten fruit (augmented 12,335 or raw 3,200)"),
        (DEFAULT_SNACKS, None, "20 snack classes, 6,745 images (Matthijs/snacks)"),
        (DEFAULT_INDIAN_FOODS, None, "15 prepared Indian dish classes, 4,770 images (LALIT324rt/indian-foods-dataset)"),
    ]:
        try:
            info = api.dataset_info(ds)
            ds_mb = getattr(info, "download_size", None)
            print(f"  {ds}  ~{(ds_mb or 0)/1e9:.2f} GB download")
            if cfg:
                print(f"      using config {cfg!r}")
            print(f"      -> {desc}")
        except Exception as exc:
            print(f"  {ds}: ERR {exc}")
    print()
    print("Kaggle (manual, needs ~/.kaggle/kaggle.json):")
    print("  jaisal228/MM-Food-100K  (prepared Indian foods -- label schema unknown until you map it)")
    print()
    print("Pass --kaggle-dataset + --kaggle-map to include it. Run with --list-sources again")
    print("after dropping kaggle.json in place to confirm the CLI is reachable.")


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)

    if a.list_sources:
        list_candidate_sources()
        return 0

    paths = get_paths()
    root = Path(a.root) if a.root else _rel_root(paths)
    root.mkdir(parents=True, exist_ok=True)
    out = Path(a.out) if a.out else paths.cv_data / "index.csv"

    # Clean up any stale placeholder dirs from prior interrupted runs2 so they cannot block
    # creation of the real per-class dirs.
    _placeholder_cleanup(root, "fresh_rotten_fruit", "snacks", "indian_foods", "mm_food_100k")

    frames: list[pd.DataFrame] = []
    mapping_report: dict[str, Any] = {}

    # ---- fruit dataset ---- #
    if not a.no_fruit:
        fruit_source = "fresh_rotten_fruit"
        mapping_report["fresh_rotten_fruit_classification"] = {
            "source": fruit_source,
            "dataset": a.hf_dataset,
            "config": a.hf_config,
            "split": a.hf_split,
            "label_map": FRESH_ROTEN_MAP,
        }
        df = download_hf_dataset(
            a.hf_dataset, a.hf_config, fruit_source, FRESH_ROTEN_MAP, root,
            max_workers=a.max_workers, limit=a.download_limit, split=a.hf_split,
        )
        frames.append(df)
        LOG.info("fruit dataset indexed: %d images", len(df))

    # ---- Indian foods dataset ---- #
    if not a.no_indian_foods:
        indian_map = indian_foods_label_map(a.indian_foods_label)
        mapping_report["indian_foods"] = {
            "source": INDIAN_FOODS_SOURCE,
            "dataset": a.indian_foods_dataset,
            "split": a.indian_foods_split,
            "label_map": indian_map,
            "note": "15 prepared Indian dish classes map to %s (ids 0..14, all are correctly prepared dishes by default)"
                    % indian_map.get("0", "?"),
        }
        df = download_indian_foods_dataset(
            a.indian_foods_dataset, INDIAN_FOODS_SOURCE, indian_map, root,
            max_workers=a.max_workers, limit=a.download_limit, split=a.indian_foods_split,
        )
        frames.append(df)
        LOG.info("Indian foods dataset indexed: %d images", len(df))

    # ---- snacks dataset ---- #
    if not a.no_snacks:
        snack_map = snacks_label_map(a.snack_label)
        snack_source = "snacks"
        mapping_report["snacks"] = {
            "source": snack_source,
            "dataset": a.snacks_dataset,
            "split": a.snacks_split,
            "label_map": snack_map,
            "note": "all snack classes map to %s (use --snack-label GOOD to treat snacks as acceptable food)"
                     % snack_label_display(snack_map),
        }
        df = download_snacks_zip(
            a.snacks_dataset, snack_source, snack_map, root,
            max_workers=a.max_workers, limit=a.download_limit,
            train_split=a.snacks_split,
        )
        frames.append(df)
        LOG.info("snacks dataset indexed: %d images", len(df))

    # ---- kaggle dataset (MM-Food-100K) ---- #
    if a.kaggle_dataset or a.kaggle_root:
        kaggle_source = "mm_food_100k"
        # load or ask for the mapping
        if a.kaggle_map:
            km = json.loads(Path(a.kaggle_map).read_text())
        else:
            # We do not know the schema without seeing the dataset. Ask the operator.
            LOG.warning("MM-Food-100K label schema is unknown to this script.")
            LOG.warning("Tell me the mapping interactively. Example: {'prepared': 'GOOD', 'spoiled': 'REJECTED'}")
            try:
                txt = input("paste JSON label map for MM-Food-100K (or empty line to skip): ")
            except EOFError:
                txt = ""
            txt = txt.strip()
            if not txt:
                LOG.warning("no mapping given -- skipping MM-Food-100K")
                a.kaggle_dataset = None
                a.kaggle_root = None
            else:
                try:
                    km = json.loads(txt)
                except json.JSONDecodeError as exc:
                    LOG.error("invalid JSON: %s", exc)
                    return 2
        if a.kaggle_dataset or a.kaggle_root:
            # resolve the image root
            if a.kaggle_root:
                kroot = Path(a.kaggle_root)
            else:
                kroot = kaggle_download(a.kaggle_dataset, root / kaggle_source)
            # Walk the tree: expect <kroot>/<source>/<CLASS>/<img> or <kroot>/<CLASS>/<img>
            rows = []
            for p in sorted(kroot.rglob("*")):
                if p.suffix.lower() not in EXT:
                    continue
                parts = p.relative_to(kroot).parts
                if len(parts) < 2:
                    LOG.warning("skipping %s: unexpected layout", p)
                    continue
                # Heuristic: last folder = class, everything before = source
                cls_folder = parts[-1]
                source_folder = parts[0] if len(parts) > 2 else "mm_food_100k"
                cls = km.get(cls_folder, km.get(cls_folder.lower(), cls_folder)).upper()
                if cls not in CLASSES:
                    LOG.warning("skipping %s: class %r not in pipeline classes", p, cls_folder)
                    continue
                rows.append({
                    "path": str(p.resolve()),
                    "rel_path": p.relative_to(root).as_posix() if p.is_relative_to(root) else p.as_posix(),
                    "source": source_folder,
                    "label": cls,
                    "group": f"{source_folder}/{p.stem}",
                    "session": f"{source_folder}/{p.stem}",
                    "bytes": p.stat().st_size,
                    "mtime": int(p.stat().st_mtime),
                })
            if not rows:
                LOG.warning("no images found under %s -- check layout", kroot)
            else:
                if a.kaggle_limit:
                    rows = rows[:a.kaggle_limit]
                df = pd.DataFrame(rows).sort_values(["source", "label", "path"]).reset_index(drop=True)
                frames.append(df)
                mapping_report["mm_food_100k"] = {
                    "source": kaggle_source,
                    "dataset": a.kaggle_dataset or "local",
                    "root": str(kroot),
                    "label_map": km,
                    "images": len(df),
                }
                LOG.info("MM-Food-100K indexed: %d images", len(df))

    if not frames:
        LOG.error("nothing to index -- enable at least one source")
        return 2

    df = pd.concat(frames, ignore_index=True)

    # ---- dedupe + split ---- #
    LOG.info("building index: %d images across %d sources",
             len(df), df["source"].nunique())
    test_sources = {s.strip() for s in a.test_source.split(",") if s.strip()}
    if not test_sources:
        LOG.warning("no --test-source; test split will share sources with train (optimistic)")

    df = add_hashes(df)
    if a.no_dedupe:
        dups = {}
    else:
        df, dups = drop_near_duplicates(df, a.dup_distance)

    df = assign_splits(df, test_sources=test_sources, val_frac=a.val_frac, seed=a.seed)
    report = audit_splits(df)
    report["near_duplicate_clusters"] = len(dups)
    report["label_mappings"] = mapping_report
    report["preprocess_policy"] = PreprocessPolicy(size=224).as_dict()
    report["index_fingerprint"] = sha256_of_frame(df, columns=["rel_path", "label", "split"])

    if report["groups_in_multiple_splits"]:
        LOG.error("%d dish groups appear in more than one split", report["groups_in_multiple_splits"])

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    save_json(out.with_suffix(".audit.json"), report)

    print()
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
    print()
    print(f"sources            : {report['sources']}")
    print(f"images / dishes    : {report['images']} / {report['groups']}")
    print(f"test unseen source : {report['test_has_unseen_source']}")
    print(f"duplicate clusters : {report['near_duplicate_clusters']}")
    print(f"label mappings     :")
    for k, v in mapping_report.items():
        lm = v.get("label_map")
        print(f"   {k}: {lm}  (source={v.get('source')})")
    print(f"wrote {out}  (+ .audit.json)")
    return 0


def snack_label_display(snack_map: dict[str, str]) -> str:
    vals = set(snack_map.values())
    if len(vals) == 1:
        return next(iter(vals))
    return " / ".join(sorted(vals))


if __name__ == "__main__":
    raise SystemExit(main())
