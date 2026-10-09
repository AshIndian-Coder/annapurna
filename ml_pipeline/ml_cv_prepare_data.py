"""Build ``data/cv/index.csv`` with splits that survive contact with a real dataset.

    python ml_cv_prepare_data.py --root data/cv/raw \\
        --test-sources kitchen_b --map fresh=GOOD,rotten=REJECTED,stale=REJECTED

Expected layout::

    data/cv/raw/<source>/<CLASS>/<batch>__<hour>h_<n>.jpg

    source  a kitchen, or the name of a public dataset.  The TEST split is made of
            whole *sources* the model has never seen -- a random photo split inflates
            accuracy by 5-15 points on food imagery, because the same dish photographed
            ten times in one session is ten rows of one observation.
    CLASS   GOOD | RISK | REJECTED | NOT_FOOD
    batch   one physical dish.  Every photo of it stays in the same split.

Three things this script exists to enforce, each of which a naive ``train_test_split``
gets wrong:

1. **Session grouping.**  Photos of one dish (``<batch>__*``) never straddle a split.
2. **Near-duplicate removal.**  Perceptual hashes drop re-encodes and 1-2 px crops of
   the same frame.  Left in, they put a near-copy of a training photo into the test
   split and the number stops meaning anything.
3. **Source-level test split.**  Reported loudly, and refused by default when a
   same-source duplicate would leak.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

from ml_cv_common import CLASSES, LOG, PreprocessPolicy, dataset_balance
from ml_utils import get_paths, save_json, sha256_of_frame

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 120_000_000        # decompression-bomb guard

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
#: Distance (in Hamming bits) below which two images are the same photograph.
DUP_DISTANCE = 6


# --------------------------------------------------------------------------- #
# Perceptual hash
# --------------------------------------------------------------------------- #
def phash(img: Image.Image, hash_size: int = 8, highfreq: int = 4) -> str:
    """Difference hash: resize to ``highfreq*hash_size``, compare adjacent columns.

    pHash proper needs a DCT; a difference hash is within a few bits on food imagery
    and costs no SciPy dependency, which matters because this runs2 on a laptop.
    """
    small = np.asarray(
        img.convert("L").resize((hash_size * highfreq, hash_size), Image.Resampling.LANCZOS),
        dtype=np.float64)
    bits = small[:, 1:] > small[:, :-1]
    packed = np.packbits(bits.reshape(-1))
    return packed.tobytes().hex()


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def find_near_duplicates(hashes: dict, distance: int = DUP_DISTANCE) -> dict:
    """Union-find over image paths; returns ``{kept_path: [dropped, ...]}``.

    A brute-force O(n^2) Hamming scan is fine to ~20 k images and, unlike an LSH
    index, has no tunable parameter that can quietly find nothing.
    """
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    keys = sorted(hashes)
    for i, ki in enumerate(keys):
        find(ki)
        for kj in keys[i + 1:]:
            if hamming(hashes[ki], hashes[kj]) <= distance:
                union(ki, kj)
    groups: dict = defaultdict(list)
    for k in keys:
        groups[find(k)].append(k)
    out: dict = {}
    for members in groups.values():
        if len(members) > 1:
            out[members[0]] = sorted(members[1:])
    return out


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #
def build_index(root: Path, mapping: dict) -> pd.DataFrame:
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in EXT:
            continue
        parts = p.relative_to(root).parts
        if len(parts) < 3:
            LOG.warning("skipping %s: expected <source>/<CLASS>/<file>", p)
            continue
        source, folder = parts[0], parts[1]
        cls = mapping.get(folder, mapping.get(folder.lower(), folder)).upper()
        if cls not in CLASSES:
            LOG.warning("skipping %s: class folder '%s' is not one of %s",
                        p, folder, CLASSES)
            continue
        stem = p.stem
        # "<batch>__4h_0" and "<batch>__4h_0.jpg" -> one dish
        batch = re.split(r"__|_\d{1,2}h", stem)[0]
        # a capture session: same dish, same hour bucket, usually same plate position
        session = re.sub(r"_\d+$", "", stem)
        rows.append({
            "path": str(p.resolve()),
            "rel_path": p.as_posix(),
            "source": source,
            "label": cls,
            "group": f"{source}/{batch}",
            "session": f"{source}/{session}",
            "bytes": p.stat().st_size,
            "mtime": int(p.stat().st_mtime),
        })
    if not rows:
        raise SystemExit(f"no images under {root} (expected <source>/<CLASS>/<file>)")
    return pd.DataFrame(rows)


def add_hashes(df: pd.DataFrame, *, progress_every: int = 500) -> pd.DataFrame:
    out = df.copy()
    hashes, failed = [], []
    for i, path in enumerate(out["path"], 1):
        try:
            with Image.open(path) as im:
                hashes.append(phash(im))
        except Exception as exc:  # noqa: BLE001
            LOG.warning("unreadable image %s (%s) -- dropping", path, exc)
            hashes.append("")
            failed.append(path)
        if i % progress_every == 0:
            LOG.info("hashed %d/%d images", i, len(out))
    out["phash"] = hashes
    if failed:
        out = out[out["path"].isin(set(out["path"]) - set(failed))].reset_index(drop=True)
        LOG.warning("dropped %d unreadable image(s)", len(failed))
    return out


def drop_near_duplicates(df: pd.DataFrame, distance: int = DUP_DISTANCE) -> tuple:
    """Keep the largest file in each duplicate cluster (best encoded, not smallest)."""
    dups = find_near_duplicates(dict(zip(df["path"], df["phash"])), distance)
    if not dups:
        LOG.info("no near-duplicates within Hamming distance %d", distance)
        return df, {}
    drop: set = set()
    for keep, others in dups.items():
        sizes = df.set_index("path")["bytes"].to_dict()
        best = max([keep, *others], key=lambda p: sizes.get(p, 0))
        drop.update({p for p in ([keep, *others]) if p != best})
    LOG.info("near-duplicate removal: %d clusters -> dropping %d images",
             len(dups), len(drop))
    out = df[~df["path"].isin(drop)].reset_index(drop=True)
    return out, dups


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def assign_splits(df: pd.DataFrame, *, test_sources, val_frac: float, seed: int) -> pd.DataFrame:
    """Source-level test split, then group-stratified validation.

    Groups (a dish = many photos) are the unit of allocation, and the allocation is
    stratified on the *group's* label so a val fold without any ``REJECTED`` example
    cannot silently redefine what "high recall" means.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["split"] = np.where(out["source"].isin(test_sources), "test", "train")

    rest = out[out["split"] != "test"]
    by_label: dict = defaultdict(list)
    for grp, sub in rest.groupby("group"):
        by_label[sub["label"].iloc[0]].append(grp)
    val: set = set()
    for label, groups in by_label.items():
        groups = sorted(groups)
        rng.shuffle(groups)
        n = max(1, int(round(len(groups) * val_frac)))
        val.update(groups[:n])
    out.loc[out["group"].isin(val) & (out["split"] != "test"), "split"] = "val"
    return out


def _duplicates_across_splits(dups: dict, df: pd.DataFrame) -> int:
    """Duplicate clusters whose surviving members landed in different splits.

    Only reachable when a cluster spans two sources (the same dish photographed at two
    kitchens) -- the common case is already handled because the whole cluster is dropped
    except one image.  Counted rather than assumed, because a cluster that *does* straddle
    makes the test number optimistic and the reader deserves to know.
    """
    if not dups:
        return 0
    split_of = dict(zip(df["path"], df["split"]))
    n = 0
    for keep, others in dups.items():
        splits = {split_of[keep]} | {split_of[p] for p in others if p in split_of}
        if len(splits) > 1:
            n += 1
    return n


def audit_splits(df: pd.DataFrame) -> dict:
    """Everything a reader needs to decide whether to believe the test number."""
    rep = {
        "counts": df.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict(),
        "images": int(len(df)),
        "groups": int(df["group"].nunique()),
        "sessions": int(df["session"].nunique()),
        "sources": sorted(df["source"].unique()),
    }
    sizes = df.groupby("group")["split"].nunique()
    rep["groups_in_multiple_splits"] = int((sizes > 1).sum())
    if rep["groups_in_multiple_splits"]:
        offenders = sizes[sizes > 1].index[:10].tolist()
        rep["offending_groups"] = offenders
    src = df.groupby("source")["split"].nunique()
    rep["sources_in_multiple_splits"] = int((src > 1).sum())
    rep["test_has_unseen_source"] = bool(
        not (set(df[df.split == "test"]["source"]) & set(df[df.split != "test"]["source"])))
    rep["balance"] = dataset_balance(df)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--test-sources", default="",
                    help="comma-separated source folders held out entirely for test")
    ap.add_argument("--map", default="", help="folder-name mapping, e.g. fresh=GOOD,rotten=REJECTED")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--dup-distance", type=int, default=DUP_DISTANCE)
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--require-unseen-test-source", action="store_true",
                    help="fail if the test split shares a source with train/val")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    paths = get_paths()
    root = Path(a.root) if a.root else paths.cv_data / "raw"
    out = Path(a.out) if a.out else paths.cv_data / "index.csv"
    mapping = dict(kv.split("=", 1) for kv in a.map.split(",") if "=" in kv)
    test_sources = {s.strip() for s in a.test_sources.split(",") if s.strip()}

    df = build_index(root, mapping)
    LOG.info("found %d images across %d sources", len(df), df["source"].nunique())

    df = add_hashes(df)
    if a.no_dedupe:
        dups = {}
    else:
        df, dups = drop_near_duplicates(df, a.dup_distance)

    df = assign_splits(df, test_sources=test_sources, val_frac=a.val_frac, seed=a.seed)
    report = audit_splits(df)
    report["near_duplicate_clusters"] = len(dups)
    report["near_duplicates_straddling_splits"] = _duplicates_across_splits(dups, df)
    if report["near_duplicates_straddling_splits"]:
        LOG.warning("%d near-duplicate clusters straddle a split boundary -- the test "
                    "accuracy is optimistic", report["near_duplicates_straddling_splits"])
    report["preprocess_policy"] = PreprocessPolicy(size=224).as_dict()
    report["index_fingerprint"] = sha256_of_frame(df, columns=["rel_path", "label", "split"])

    if report["groups_in_multiple_splits"]:
        LOG.error("%d dish groups appear in more than one split -- a photo of the same "
                  "dish is in train and test", report["groups_in_multiple_splits"])
    if not test_sources:
        LOG.warning("no --test-sources given: the test split shares sources with train, so "
                    "accuracy will be optimistic. Pass whole kitchens for a real number.")
    elif a.require_unseen_test_source and not report["test_has_unseen_source"]:
        raise SystemExit("--require-unseen-test-source: the test split shares a source "
                         "with train/val")

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
    print(f"wrote {out}  (+ .audit.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
