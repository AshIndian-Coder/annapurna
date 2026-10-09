"""Run every stage in dependency order, one command, resumable.

    python ml_run_all.py                 # every CPU stage, then CV if data exists
    python ml_run_all.py demand quality  # only the named stages
    python ml_run_all.py --list          # what would run, and what each stage needs
    python ml_run_all.py --with-cv       # include the GPU image stages
    python ml_run_all.py --from quality  # resume after a failure

Each stage is a separate process on purpose: a stage that leaks memory, wedges a thread
pool, or segfaults in NumPy takes down only itself, and the orchestrator reports which
stage died instead of leaving a half-written artefact and a stack trace with no context.

Stages are ordered so that a failure is always cheap: simulators before trainers,
trainers before the report that reads their output, tests first of all.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from ml_utils import get_logger, get_paths, save_json

LOG = get_logger("run_all")


@dataclass
class Stage:
    name: str
    argv: list
    needs: list = field(default_factory=list)   # names of stages that must run first
    gpu: bool = False
    minutes: float = 30.0                        # a budget, not a guarantee
    note: str = ""
    #: Files that must exist for this stage to have anything to do. A stage whose inputs
    #: are absent is SKIPPED with a reason, not run and failed: an image stage on an empty
    #: ``data/cv/raw`` is a missing photo library, not a broken pipeline, and reporting it
    #: as a failure trains people to ignore red.
    requires: list = field(default_factory=list)


def _py(*args) -> list:
    return [sys.executable, "-W", "ignore", *args]


def stages() -> list:
    return [
        # -- tests first: a broken invariant should stop everything ---------------
        Stage("tests",
              _py("ml_tests_test_demand.py", "ml_tests_test_fusion_monotonic.py"),
              minutes=10,
              note="leakage guard, CQR coverage, newsvendor arithmetic, API contract"),

        # -- demand (PS requirement 1 and 8) -------------------------------------
        Stage("demand-data", _py("ml_data_gen_kitchen_simulator.py", "--days", "540"),
              minutes=10,
              note="SYNTHETIC institutional-kitchen consumption, 4 kitchens"),
        Stage("demand", _py("ml_demand_train.py", "--promote"),
              needs=["demand-data"], minutes=25,
              note="leakage-safe features -> baselines -> rolling CV -> CQR -> newsvendor "
                   "-> policy backtest -> promotion gate"),

        # -- safety fusion (PS requirement 2) -------------------------------------
        Stage("quality-data", _py("ml_data_gen_sensor_simulator.py"),
              minutes=10, note="SYNTHETIC time-temperature sensor streams"),
        Stage("shelf-life", _py("ml_quality_train_shelf_life.py"),
              needs=["quality-data"], minutes=10,
              note="remaining shelf life from time-temperature history"),
        Stage("quality", _py("ml_quality_train_fusion.py"),
              needs=["quality-data", "shelf-life"], minutes=15,
              note="monotone fusion of CV + danger-zone minutes; the monotonicity test "
                   "runs2 in the 'tests' stage"),

        # -- processing efficiency (PS requirements 5 and 6) ----------------------
        Stage("processing-data", _py("ml_data_gen_processing_simulator.py"),
              minutes=5, note="SYNTHETIC processing metrics with injected anomalies"),
        Stage("processing", _py("ml_analytics_processing_anomaly.py"),
              needs=["processing-data"], minutes=10,
              note="D2/D3 KPI formulas, EWMA/CUSUM + IsolationForest"),

        # -- impact and logistics (PS requirements 3, 4, 7) ----------------------
        Stage("sustainability", _py("ml_analytics_sustainability.py", "--demo"),
              minutes=10, note="impact + ESG report (labelled estimate)"),
        # One stage per script: ``python a.py b.py`` runs2 a.py and puts b.py in sys.argv,
        # so a combined "stage" silently skips its second half and still reports success.
        Stage("logistics-bench", _py("ml_logistics_benchmark.py", "--save-instances"),
              minutes=10, note="solver benchmark scored on ON-TIME portions (OR-Tools vs "
                               "nearest-neighbour); writes data/raw/sim_logistics/"),
        Stage("logistics-match", _py("ml_logistics_match.py"),
              needs=["logistics-bench"], minutes=10, note="fair matching as a linear programme"),
        Stage("logistics-route", _py("ml_logistics_route.py"),
              needs=["logistics-bench"], minutes=10, note="VRPTW routing with expiry deadlines"),

        # -- image model (PS requirement 2) -- GPU, and needs a photo library ----
        Stage("cv-prepare", _py("ml_cv_prepare_data.py", "--test-sources", "kitchen_b",
                                "--require-unseen-test-source"),
              gpu=False, minutes=30,
              requires=["data/cv/raw"],
              note="needs data/cv/raw/<source>/<CLASS>/; pHash dedupe + group splits"),
        Stage("cv-probe", _py("ml_cv_dino_probe.py"), needs=["cv-prepare"],
              gpu=True, minutes=45, note="frozen DINOv2 + linear head (the fast teacher)"),
        Stage("cv-train", _py("ml_cv_train_cls.py", "--out", "lite0_staged",
                              "--epochs", "40", "--head-epochs", "3", "--amp", "auto"),
              needs=["cv-prepare"], gpu=True, minutes=180,
              note="staged unfreeze + EMA + mixup/cutmix; --resume to continue"),
        Stage("cv-ood", _py("ml_cv_ood.py", "fit", "--run", "lite0_staged"),
              needs=["cv-train"], gpu=True, minutes=15,
              note="class-conditional Mahalanobis on the penultimate features"),
        Stage("cv-export", _py("ml_cv_export.py", "--run", "lite0_staged", "--tflite",
                               "--bundle", "--require-device"),
              needs=["cv-train", "cv-ood"], gpu=True, minutes=45,
              note="ONNX fp32+int8, D21 parity gate, manifest.json"),
        Stage("cv-report", _py("ml_cv_eval_report.py", "--run", "lite0_staged"),
              needs=["cv-train"], minutes=10, gpu=True,
              note="the artefact a judge reads; refuses to quote a number it cannot "
                   "regenerate. gpu=True: it reads runs2/<name>/ from cv-train, so with no "
                   "trained run it has nothing to regenerate"),
        Stage("bakeoff", _py("ml_cv_bakeoff.py"), needs=["cv-report"], minutes=5,
              gpu=True,
              note="picks a winner, or refuses to. gpu=True because it consumes the cv "
                   "chain's output: with no trained run there is nothing to compare and it "
                   "would exit non-zero on an empty runs2/ directory"),

        # -- food image model (multi-dataset: fresh/rotten fruit + snacks + optional MM-Food-100K) --
        # This is the only stage you need to run now that the CPU-side models are done.
        # It is the foreground, resumable, interruptible training run: closing this chat kills
        # the process cleanly; re-running with --from food resumes from the last checkpoint.
        Stage("food-prepare", _py("ml_cv_prepare_food_data.py", "--test-source", "snacks"),
              gpu=False, minutes=120,
              requires=["data/cv/raw"],
              note="downloads HF fresh_rotten_fruit_classification (augmented) + Matthijs/snacks "
                   "into data/cv/raw/<source>/<CLASS>/, maps labels into GOOD/RISK/REJECTED/NOT_FOOD, "
                   "pHashes + dedups + group-stratified split + whole-source test holdout. "
                   "Needs internet + HF token in ~/.huggingface/token. "
                   "Add MM-Food-100K with --kaggle-dataset + ~/.kaggle/kaggle.json."),
        Stage("food-train", _py("ml_cv_train_food.py", "--out", "food_efficientnet",
                                "--epochs", "30", "--head-epochs", "3", "--amp", "auto",
                                "--aug", "strong"),
              needs=["food-prepare"], gpu=True, minutes=540,
              note="EfficientNet-Lite0 on the multi-dataset food index. Staged unfreeze + EMA + "
                   "mixup/cutmix + label smoothing + AMP + gradient clip. "
                   "Resume with: python ml_cv_train_food.py --out runs2/food_efficientnet --resume"),
        Stage("food-probe", _py("ml_cv_dino_probe.py"), needs=["food-prepare"],
              gpu=True, minutes=120,
              note="frozen DINOv2-S/14 + linear head on the same index (stronger teacher, slower). "
                   "Optional --arch facebook/dinov2-small in food-train gives the same backbone as a "
                   "full fine-tune."),
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="*", help="run only these stages")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--with-cv", action="store_true", help="include the GPU image stages")
    ap.add_argument("--from", dest="from_", default=None, help="start at this stage")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", default=None, help="write a JSON run summary here")
    a = ap.parse_args(argv)

    paths = get_paths()
    all_stages = stages()
    by_name = {s.name: s for s in all_stages}
    order = [s.name for s in all_stages]

    selected = order
    if a.stages:
        unknown = [s for s in a.stages if s not in by_name]
        if unknown:
            ap.error(f"unknown stage(s): {unknown}. Known: {order}")
        selected = [s for s in order if s in set(a.stages)]
    if a.from_:
        if a.from_ not in by_name:
            ap.error(f"unknown stage: {a.from_}. Known: {order}")
        selected = [s for s in order if order.index(s) >= order.index(a.from_)]
    if a.list:
        # --list shows the whole plan, GPU stages included: the point of a plan is to
        # show what a later run will do.  Skipped stages are marked in the run itself.
        for name in selected:
            s = by_name[name]
            print(f"{s.name:16s} {'gpu' if s.gpu else '   ':4s} "
                  f"{','.join(s.needs) or '-':16s} {s.note}")
        return 0
    if not a.with_cv and not a.stages:
        selected = [s for s in selected if not by_name[s].gpu]

    summary_path = Path(a.report) if a.report else None

    # dependency closure over the selection
    todo = list(selected)
    closed: list = []
    seen: set = set()

    def visit(name: str) -> None:
        if name in seen or name not in by_name:
            return
        seen.add(name)
        for dep in by_name[name].needs:
            visit(dep)
        closed.append(name)

    for name in todo:
        visit(name)

    if a.dry_run:
        # the closure, not the raw selection: --dry-run must show what would *run*,
        # including the dependencies it drags in
        for name in closed:
            cmd = " ".join(by_name[name].argv)
            tag = "  [gpu]" if by_name[name].gpu else ""
            print(f"{name:16s} {cmd}{tag}")
        return 0

    summary = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "stages": {}}
    t_all = time.perf_counter()
    for i, name in enumerate(closed, 1):
        s = by_name[name]
        if s.gpu and not a.with_cv:
            LOG.info("[%d/%d] SKIP %s (GPU stage; pass --with-cv)", i, len(closed), name)
            summary["stages"][name] = {"status": "skipped", "reason": "gpu"}
            continue
        missing = [r for r in s.requires if not (paths.pipeline_dir / r).exists()]
        if missing:
            LOG.info("[%d/%d] SKIP %s (missing input: %s)", i, len(closed), name,
                     ", ".join(missing))
            summary["stages"][name] = {"status": "skipped", "reason": f"missing input: {missing}"}
            continue
        LOG.info("[%d/%d] %s -- %s", i, len(closed), name, s.note)
        t0 = time.perf_counter()
        try:
            rc = subprocess.call(s.argv, cwd=str(paths.pipeline_dir))
        except KeyboardInterrupt:
            LOG.error("interrupted during '%s' -- re-run with --from %s to continue", name, name)
            summary["stages"][name] = {"status": "interrupted"}
            return 130
        took = round(time.perf_counter() - t0, 1)
        summary["stages"][name] = {"status": "ok" if rc == 0 else "failed",
                                   "returncode": rc, "seconds": took}
        if rc != 0:
            LOG.error("stage '%s' failed (rc=%d) after %.1fs", name, rc, took)
            LOG.error("re-run just this stage: python ml_run_all.py --from %s %s",
                      name, name)
            summary["failed_at"] = name
            break
        if took > s.minutes * 60:
            LOG.warning("stage '%s' took %.1f min (budget was %.0f min) -- the budget in "
                        "ml_run_all.py is a guess, update it", name, took / 60, s.minutes)

    summary["elapsed_seconds"] = round(time.perf_counter() - t_all, 1)
    save_json(summary_path or (paths.runs / "run_all_summary.json"), summary)

    ok = [n for n, v in summary["stages"].items() if v["status"] == "ok"]
    bad = [n for n, v in summary["stages"].items() if v["status"] == "failed"]
    print(f"\n{len(ok)}/{len(closed)} stages ok in {summary['elapsed_seconds']:.0f}s"
          + (f"; FAILED: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
