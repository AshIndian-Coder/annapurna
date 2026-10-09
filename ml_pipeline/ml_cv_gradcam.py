from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ml_cv_common import CLASSES, LOG, eval_tf, softmax_np
from ml_utils import get_paths, save_json, sha256_json

class GradCam:
    """Grad-CAM for a ``timm`` CNN, captured at the backbone's spatial feature map.

    The tensor to explain is the **output of ``forward_features``** -- for
    EfficientNet-Lite0 at 224 px that is ``(B, 1280, 7, 7)``, the last convolutional
    map before global pooling.

    Two tempting alternatives, both wrong on timm:

    * ``getattr(net, "forward_features")`` returns a bound *method*, and
      ``method.register_forward_hook`` raises ``AttributeError`` at construction.
    * hooking the **classifier** fires, but timm's ``forward_features`` has *already*
      global-pooled, so the captured input is ``(B, 1280)`` -- no spatial axis and
      nothing to localise.

    So ``forward_features`` is wrapped on the instance: the wrapper records the output
    and registers the tensor hook that will carry the gradient.  The original bound
    method is restored in :meth:`close`; a wrapper that is not undone leaves the model
    capturing activations forever, and a server calling this per request then leaks
    GPU memory until it falls over -- quietly and spectacularly.
    """

    def __init__(self, net: nn.Module):
        self.net = net.eval()
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._tensor_hooks: list = []
        if not hasattr(self.net, "forward_features"):
            raise RuntimeError(
                f"{type(net).__name__} exposes no forward_features(); Grad-CAM needs a CNN "
                f"backbone. For a ViT-style backbone use an attention-based explainer -- a "
                f"Grad-CAM heatmap over a transformer is a picture of nothing.")
        self._original = self.net.forward_features
        self.net.forward_features = self._wrapped
        self._hook_fired = False

    def _wrapped(self, *args, **kwargs):
        out = self._original(*args, **kwargs)
        if isinstance(out, (tuple, list)):
            out = out[0]
        self.activations = out
        if out.requires_grad:
            h = out.register_hook(self._grad_hook)
            self._tensor_hooks.append(h)
            self._hook_fired = True
        return out

    def _grad_hook(self, grad):
        self.gradients = grad

    def close(self) -> None:
        for h in self._tensor_hooks:
            h.remove()
        self._tensor_hooks.clear()
        self.net.forward_features = self._original
        self.activations = self.gradients = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __call__(self, x: torch.Tensor, class_index: int | None = None):
        """Return ``(probs, cam, class_index)`` with ``cam`` in ``[0, 1]``."""
        self.net.zero_grad(set_to_none=True)
        self.activations = self.gradients = None
        logits = self.net(x)
        probs = softmax_np(logits.detach().cpu().numpy(), axis=-1)
        idx = int(class_index) if class_index is not None else int(probs.argmax(axis=1)[0])
        if not self._hook_fired or self.activations is None:
            raise RuntimeError("forward_features was never called -- the model was run "
                               "under torch.no_grad(), or it was wrapped twice")
        logits[:, idx].sum().backward(retain_graph=False)
        if self.gradients is None:
            raise RuntimeError("no gradient reached the captured activations")
        a = self.activations.detach()
        if a.dim() != 4:
            raise RuntimeError(f"captured activations have shape {tuple(a.shape)}; Grad-CAM "
                               f"needs (B, C, H, W)")
        g = self.gradients.detach()
        w = g.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * a).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1).cpu().numpy()
        flat = cam.reshape(len(cam), -1)
        lo = flat.min(axis=1)[:, None, None]
        hi = flat.max(axis=1)[:, None, None]
        cam = (cam - lo) / np.maximum(hi - lo, 1e-8)
        return probs, cam, idx

def overlay(original: Image.Image, cam: np.ndarray, *, alpha: float = 0.45) -> Image.Image:
    """Blend a jet-coloured heatmap onto the original photo at original resolution."""
    cam_img = Image.fromarray((np.clip(cam, 0, 1) * 255).astype(np.uint8))
    heat = np.asarray(cam_img.resize(original.size, Image.Resampling.BILINEAR),
                      dtype=np.float32) / 255.0
    t = heat[..., None]
    heat_rgb = 255.0 * np.concatenate(
        [np.clip(1.5 - np.abs(4 * t - 3), 0, 1),
         np.clip(1.5 - np.abs(4 * t - 2), 0, 1),
         np.clip(1.5 - np.abs(4 * t - 1), 0, 1)], axis=-1)
    base = np.asarray(original.convert("RGB"), dtype=np.float32)
    out = base * (1 - alpha) + heat_rgb * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def bounding_box(cam: np.ndarray, *, threshold: float = 0.5) -> list | None:
    """The region the explanation actually points at, in normalised ``[x, y, w, h]``.

    Reported because the API contract's ``detections[].bbox`` needs *some* box, and a
    whole-image box is not a detection.  ``None`` when the activation is diffuse, which
    is the honest answer for a plate where nothing localises.
    """
    mask = cam >= threshold
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return [round(x0 / w, 4), round(y0 / h, 4),
            round((x1 - x0) / w, 4), round((y1 - y0) / h, 4)]


# --------------------------------------------------------------------------- #
# Evidence record
# --------------------------------------------------------------------------- #
def _load(run_dir: Path):
    import joblib
    import timm

    cal = joblib.load(run_dir / "calib.joblib")
    net = timm.create_model(cal["arch"], pretrained=False, num_classes=len(CLASSES))
    net.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu",
                                   weights_only=True))
    return net.eval(), cal


def explain(paths, run_dir: Path, *, out_dir: Path | None = None, device: str = "cpu",
            temperature: float = 1.0) -> list:
    net, cal = _load(run_dir)
    dev = torch.device(device)
    net = net.to(dev)
    size = int(cal["size"])
    tf = eval_tf(size)
    m = torch.tensor(cal["mean"], dtype=torch.float32, device=dev).view(1, 3, 1, 1)
    s = torch.tensor(cal["std"], dtype=torch.float32, device=dev).view(1, 3, 1, 1)
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with GradCam(net) as cam_fn:
        for p in paths:
            with Image.open(p) as im:
                original = im.convert("RGB")
                x = torch.from_numpy(tf(original).numpy()).unsqueeze(0).to(dev)
            probs, cam, idx = cam_fn(x)
            calibrated = softmax_np(
                np.log(np.clip(probs, 1e-12, 1.0)) / max(temperature, 1e-6),
                axis=-1)[0]
            record = {
                "image": str(p),
                "class_index": idx,
                "label": CLASSES[idx],
                "probs": {c: round(float(calibrated[j]), 5) for j, c in enumerate(CLASSES)},
                "bbox": bounding_box(cam[0]),
                "cam_resolution": list(cam[0].shape),
                "model_version": run_dir.name,
            }
            record["evidence_hash"] = sha256_json(
                {"probs": record["probs"], "label": record["label"],
                 "cam": np.round(cam[0], 4).tolist()})
            if out_dir:
                overlay(original, cam[0]).save(out_dir / f"{Path(p).stem}_gradcam.png")
                record["heatmap"] = f"{out_dir.name}/{Path(p).stem}_gradcam.png"
            records.append(record)
    return records


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", default=None, help="write the evidence records here")
    a = ap.parse_args(argv)
    paths = get_paths()
    run = Path(a.run) if Path(a.run).exists() else paths.latest_model_dir("cv")
    records = explain(a.images, run, out_dir=Path(a.out) if a.out else None,
                      device=a.device)
    if a.json:
        save_json(a.json, records)
    for r in records:
        print(f"{r['label']:9s} p={r['probs'][r['label']]:.3f}  bbox={r['bbox']}  "
              f"evidence={r['evidence_hash'][:16]}")
    LOG.info("evidence hash binds the CV result into the custody chain (D30); the phone "
             "cannot produce one, so a hash here is proof the decision was server-side")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
