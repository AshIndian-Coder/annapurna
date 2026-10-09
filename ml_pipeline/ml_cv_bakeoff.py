from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from ml_cv_common import LOG
from ml_utils import format_table, get_paths, save_json

TARGET_UNSAFE_RECALL = 0.98

PHONE_BUDGET_MB = 40.0


def collect(runs_root: Path) -> list:
    rows = []
    for f in sorted(runs_root.glob("*/metrics.json")):
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            LOG.error("%s is not valid JSON (%s) -- skipped", f, exc)
            continue
        test = m.get("test_unseen_sources") or {}
        val = m.get("val") or {}
        run = f.parent
        rows.append({
            "run": run.name,
            "arch": m.get("arch") or m.get("model"),
            "size_px": m.get("size"),
            "params_m": m.get("params_m"),
            "unseen_test": bool(test),
            "test_sources": ",".join(m.get("test_sources") or []) or None,
            "val_macro_f1": val.get("macro_f1"),
            "test_macro_f1": test.get("macro_f1"),
            "test_unsafe_recall": test.get("unsafe_recall"),
            "false_accepts": test.get("false_accepts"),
            "good_auto_accept": test.get("good_auto_accept_rate"),
            "ece": test.get("ece"),
            "meets_recall": test.get("meets_recall_target"),
            "onnx_mb": _size_mb(run, "model_int8.onnx"),
            "tflite_mb": _size_mb(run, "model_int8.tflite"),
        })
    return rows


def _size_mb(run: Path, name: str):
    p = run / name
    return round(p.stat().st_size / 1e6, 2) if p.exists() else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs2", default=None)
    ap.add_argument("--require-eval-report", action="store_true",
                    help="warn when a run has no eval_report.json")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args(argv)

    paths = get_paths()
    root = Path(a.runs) if a.runs else paths.runs
    rows = collect(root)
    if not rows:
        raise SystemExit(f"no runs2 with metrics.json under {root}; train something first")

    no_test = [r for r in rows if not r["unseen_test"]]
    eligible = [r for r in rows if r["unseen_test"]]
    qualifiers = [r for r in eligible
                  if (r["test_unsafe_recall"] or 0) >= TARGET_UNSAFE_RECALL]

    cols = ["run", "arch", "params_m", "test_macro_f1", "test_unsafe_recall",
            "false_accepts", "good_auto_accept", "ece", "onnx_mb", "tflite_mb"]
    print(format_table(rows, cols))

    if no_test:
        print(f"\n{len(no_test)} run(s) have NO unseen-source test split and do not compete:")
        for r in no_test:
            print(f"   {r['run']}  ({r['arch']}) -- add --test-sources to "
                  f"ml_cv_prepare_data.py, otherwise the number is optimistic")
    if not qualifiers:
        print(f"\nNo run reaches unsafe recall >= {TARGET_UNSAFE_RECALL} on unseen sources. "
              f"Collect more photos (more kitchens, lighting, hours-since-cooking) before "
              f"changing models -- a threshold lowered to make a slide fit is how unsafe "
              f"food gets redistributed.")
        return 1

    qualifiers.sort(key=lambda r: (-(r["good_auto_accept"] or 0),
                                   -(r["test_macro_f1"] or 0),
                                   r["onnx_mb"] or 1e9))
    winner = qualifiers[0]
    phone = [r for r in qualifiers if (r["tflite_mb"] or 1e9) <= PHONE_BUDGET_MB]
    phone.sort(key=lambda r: (r["tflite_mb"] or 1e9, -(r["good_auto_accept"] or 0)))

    print(f"\nserver pick : {winner['run']}  "
          f"(unsafe recall {winner['test_unsafe_recall']}, "
          f"good auto-accept {winner['good_auto_accept']}, macro-F1 {winner['test_macro_f1']})")
    if phone:
        p = phone[0]
        print(f"phone pick  : {p['run']}  ({p['tflite_mb']} MB TFLite, "
              f"unsafe recall {p['test_unsafe_recall']})")
    else:
        print("phone pick  : none -- no qualifying run has a TFLite artifact. The phone "
              "cannot serve a model that was never exported; see ml_cv_export.py "
              "--require-device.")
    if a.require_eval_report:
        missing = [r["run"] for r in qualifiers
                   if not (root / r["run"] / "eval_report.json").exists()]
        if missing:
            LOG.warning("no eval_report.json for: %s -- run ml_cv_eval_report.py --all",
                        ", ".join(missing))

    if a.json_out:
        save_json(a.json_out, {"target_unsafe_recall": TARGET_UNSAFE_RECALL,
                               "rows": rows, "qualifiers": [r["run"] for r in qualifiers],
                               "server_pick": winner["run"],
                               "phone_pick": phone[0]["run"] if phone else None})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
