from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ml_cv_common import CLASSES, LOG, eval_tf
from ml_utils import get_paths, save_json, torch_load

OOD_FILE = "ood.npz"
SHRINKAGE = 0.01


class FeatureExtractor(nn.Module):
    """Backbone features with the classification head removed.

    Written against ``timm``'s ``forward_features`` so it works for every supported
    backbone without a per-architecture special case; ``forward_head(..., pre_logits=True)``
    is the documented way to get the penultimate vector.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        f = self.net.forward_features(x)
        return self.net.forward_head(f, pre_logits=True)


def _load_model(run_dir: Path):
    import joblib
    import timm

    cal = joblib.load(run_dir / "calib.joblib")
    net = timm.create_model(cal["arch"], pretrained=False, num_classes=len(CLASSES))
    net.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu",
                                   weights_only=True))
    net.eval()
    return net, cal


def _batch_features(extractor, paths, size, mean, std, device, bs=64) -> np.ndarray:
    tf = eval_tf(size)
    m = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            chunk = paths[i:i + bs]
            imgs = []
            for p in chunk:
                with Image.open(p) as im:
                    imgs.append(tf(im.convert("RGB")))
            x = torch.from_numpy(np.stack(imgs)).to(device)
            f = extractor((x - m) / s)
            out.append(f.float().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 1), np.float32)


def fit(run_dir: Path, index_csv: Path, *, device: str = "cpu") -> dict:
    """Fit the class-conditional Gaussians on the *training* split and save them."""
    import pandas as pd

    df = pd.read_csv(index_csv)
    train = df[df.split == "train"]
    if train.empty:
        raise SystemExit("no train rows in the index")
    net, cal = _load_model(run_dir)
    dev = torch.device(device)
    extractor = FeatureExtractor(net).to(dev).eval()

    feats = _batch_features(extractor, train["path"].tolist(), int(cal["size"]),
                            cal["mean"], cal["std"], dev)
    labels = np.array([CLASSES.index(c) for c in train["label"]])
    if len(feats) != len(labels):
        raise RuntimeError(f"feature/label length mismatch: {len(feats)} vs {len(labels)}")

    mus, precs, counts = {}, {}, {}
    global_mean = feats.mean(axis=0)
    for c, name in enumerate(CLASSES):
        sel = feats[labels == c]
        counts[name] = int(len(sel))
        if len(sel) < 2:
            mus[name] = global_mean
            cov = np.cov(feats.T) if len(feats) > 1 else np.eye(feats.shape[1], np.float32)
            cov = np.atleast_2d(cov)
        else:
            mus[name] = sel.mean(axis=0)
            cov = np.atleast_2d(np.cov(sel.T))
        cov = cov + np.eye(cov.shape[0], dtype=np.float32) * (SHRINKAGE * float(np.trace(cov))
                                                               / max(cov.shape[0], 1))
        precs[name] = np.linalg.pinv(cov).astype(np.float32)

    dim = feats.shape[1]
    np.savez_compressed(
        run_dir / OOD_FILE,
        classes=np.array(CLASSES),
        mus=np.stack([mus[c] for c in CLASSES]),
        precs=np.stack([precs[c] for c in CLASSES]),
        counts=np.array([counts[c] for c in CLASSES]),
        threshold=np.array([-1.0], dtype=np.float32),
        dim=np.array([dim], dtype=np.int32),
    )
    LOG.info("fitted OOD on %d train images, dim=%d", len(train), dim)
    return {"dim": dim, "counts": counts, "n_train": int(len(train))}


def mahalanobis(feats: np.ndarray, model: dict) -> np.ndarray:
    """Minimum squared Mahalanobis distance over the class Gaussians."""
    mus, precs = model["mus"], model["precs"]
    d = np.empty((feats.shape[0], len(CLASSES)), dtype=np.float64)
    for c in range(len(CLASSES)):
        diff = feats - mus[c][None, :]
        d[:, c] = np.einsum("ij,jk,ik->i", diff, precs[c], diff)
    return d.min(axis=1)


def calibrate_threshold(run_dir: Path, index_csv: Path, *, device: str = "cpu") -> dict:
    """Choose the distance threshold on the *validation* split at max F2.

    F2 rather than accuracy or F1: flagging an in-domain photo is an inconvenience,
    missing an out-of-domain one is a safety-relevant error, so recall is weighted
    double.  The score curve is saved so the choice can be argued with.
    """
    import pandas as pd

    df = pd.read_csv(index_csv)
    val = df[df.split == "val"]
    if val.empty:
        raise SystemExit("no val rows in the index; cannot calibrate a threshold")
    net, cal = _load_model(run_dir)
    dev = torch.device(device)
    feats = _batch_features(FeatureExtractor(net).to(dev).eval(), val["path"].tolist(),
                            int(cal["size"]), cal["mean"], cal["std"], dev)
    npz = dict(np.load(run_dir / OOD_FILE, allow_pickle=False))
    d = mahalanobis(feats, npz)
    labels = np.array([CLASSES.index(c) for c in val["label"]])
    per_class = {}
    for c, name in enumerate(CLASSES):
        sel = labels == c
        if sel.any():
            per_class[name] = {"n": int(sel.sum()), "p50": float(np.median(d[sel])),
                               "p95": float(np.quantile(d[sel], 0.95))}
    if len(per_class) < 2:
        return {"applicable": False, "reason": "val split has fewer than two classes",
                "per_class": per_class}
    worst = max(per_class, key=lambda k: per_class[k]["p50"])
    pos = d[labels == CLASSES.index(worst)]
    neg = d[labels != CLASSES.index(worst)]

    grid = np.quantile(np.concatenate([pos, neg]), np.linspace(0.01, 0.99, 200))
    best = None
    curve = []
    for t in grid:
        tp = float((pos > t).sum())
        fn = float((pos <= t).sum())
        fp = float((neg > t).sum())
        prec = tp / max(tp + fp, 1e-9)
        rec = tp / max(tp + fn, 1e-9)
        f2 = 5 * prec * rec / max(4 * prec + rec, 1e-9)
        curve.append({"t": float(t), "precision": prec, "recall": rec, "f2": f2})
        if best is None or f2 > best["f2"]:
            best = curve[-1]
    thr = float(np.quantile(neg, 0.995))
    if best and best["f2"] > 0:
        thr = best["t"]
    npz["threshold"] = np.array([thr], dtype=np.float32)
    np.savez_compressed(run_dir / OOD_FILE, **npz)
    save_json(run_dir / "ood_threshold.json", {
        "threshold": thr, "f2_point": best, "positive_class": worst,
        "per_class_distance": per_class,
        "caveat": "there is no real out-of-domain class in the index; positives are the "
                  "validation images of the class with the largest median distance, so "
                  "this threshold is a starting point, not a calibrated OOD detector. "
                  "Collect genuinely foreign images before trusting it.",
    })
    LOG.info("OOD threshold %.3f (positives = '%s', F2 %.3f)", thr, worst, (best or {}).get("f2", 0))
    return {"threshold": thr, "f2_point": best, "positive_class": worst}


def score(paths, run_dir: Path, *, device: str = "cpu") -> list:
    npz = dict(np.load(run_dir / OOD_FILE, allow_pickle=False))
    thr = float(npz["threshold"][0])
    net, cal = _load_model(run_dir)
    dev = torch.device(device)
    feats = _batch_features(FeatureExtractor(net).to(dev).eval(), list(paths),
                            int(cal["size"]), cal["mean"], cal["std"], dev)
    d = mahalanobis(feats, npz)
    return [{"path": str(p), "distance": round(float(x), 3), "ood": bool(x > thr),
             "threshold": round(thr, 3)} for p, x in zip(paths, d)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fit", "calibrate", "score"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        p.add_argument("--index", default=None)
        p.add_argument("--device", default="cpu")
        p.add_argument("--images", nargs="*", default=[])
    a = ap.parse_args(argv)
    paths = get_paths()
    index = Path(a.index) if a.index else paths.cv_data / "index.csv"
    run = Path(a.run)

    if a.cmd == "fit":
        print(json.dumps(fit(run, index, device=a.device), indent=2))
    elif a.cmd == "calibrate":
        print(json.dumps(calibrate_threshold(run, index, device=a.device), indent=2))
    else:
        if not a.images:
            raise SystemExit("score needs --images")
        rows = score(a.images, run, device=a.device)
        print(json.dumps(rows, indent=2))
        print(f"\nflagged {sum(r['ood'] for r in rows)}/{len(rows)} as out-of-domain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
