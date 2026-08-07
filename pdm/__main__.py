"""CLI entry point: `python -m pdm` (analysis summary) / `--deliverables` (PDF+Excel)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="pdm", description=__doc__)
    ap.add_argument("--deliverables", action="store_true", help="write PDF + Excel deliverables")
    ap.add_argument("--outdir", default="deliverables", help="output directory")
    ap.add_argument(
        "--alert-econ-out",
        default=None,
        metavar="DIR",
        help="(re)write DIR/alert_economics.{csv,svg} reference artifacts (e.g. docs)",
    )
    ap.add_argument(
        "--rul-out",
        default=None,
        metavar="DIR",
        help="(re)write DIR/rul_eval.{csv,svg} reference artifacts (e.g. docs)",
    )
    args = ap.parse_args(argv)

    from pdm.alert_economics import build_reference_artifacts, economics_from_result
    from pdm.exports import build_deliverables
    from pdm.pipeline import run_pipeline
    from pdm.rul import build_reference_artifacts as build_rul_artifacts
    from pdm.rul import evaluate_rul

    print("[pdm] running pipeline on the synthetic seeded fleet (no real telemetry)")
    result = run_pipeline()
    cfg = result.data.config
    print(
        f"[data] {cfg.n_machines} machines x {cfg.n_days} days, seed {cfg.seed}, "
        f"{len(result.data.faulty_machines())} machines with injected faults"
    )
    for name, rep in result.reports.items():
        print(
            f"[detect] {name}: ROC-AUC {rep.roc_auc:.3f}  PR-AUC {rep.pr_auc:.3f}  "
            f"precision@{rep.k} {rep.precision_at_k:.3f}  "
            f"detected {rep.n_detected}/{rep.n_faulty}  "
            f"mean delay {rep.mean_delay_days:.1f} d  val FPR {rep.val_fpr:.1%}"
        )
    margin = "n/a" if result.margin != result.margin else f"{result.margin:+.4f}"
    print(f"[detect] recommended: {result.recommended} (AE - PCA PR-AUC margin: {margin})")
    top = result.health.ranking[:5]
    tops = ", ".join(f"M{m:02d}({result.health.final[m]:.0f})" for m in top)
    print(f"[health] most urgent (heuristic degradation score, not RUL): {tops}")
    sched = result.schedule
    print(
        f"[schedule] CP-SAT {sched.optimized.status}: weighted delay "
        f"{sched.optimized.weighted_delay} vs FIFO {sched.fifo.weighted_delay} "
        f"(-{sched.reduction_pct:.1f}%)"
    )

    econ = economics_from_result(result)
    b = econ.best
    print(
        f"[alert-econ] ILLUSTRATIVE costs (missed:false = {econ.cost.ratio:g}:1) -> "
        f"cost-optimal threshold {b.threshold:.4f}: {b.alerts_per_period:.2f} alerts/day, "
        f"{b.missed_failures}/{econ.n_positives} failure-days missed, "
        f"P/R {b.precision:.2f}/{b.recall:.2f}"
    )
    if econ.current is not None:
        cur = econ.current
        print(
            f"[alert-econ] current 5%-FPR threshold {cur.threshold:.4f}: "
            f"{cur.alerts_per_period:.2f} alerts/day, {cur.missed_failures} missed "
            f"(synthetic data; costs illustrative, not a business guarantee)"
        )

    rul = evaluate_rul(result)
    print(
        f"[rul] leave-one-machine-out over {rul.n_machines} progressive-fault machines "
        f"({rul.n_samples} degradation-days): MAE {rul.mae:.2f} d, RMSE {rul.rmse:.2f} d "
        f"(naive-mean baseline {rul.baseline_mae:.2f} d)"
    )
    print(
        f"[rul] near-failure MAE (RUL<={rul.config.horizon_days}d) {rul.mae_within_horizon:.2f} d, "
        f"alpha-accuracy {rul.alpha_accuracy:.2f} "
        f"(ILLUSTRATIVE failure level; not calibrated field RUL)"
    )

    if args.rul_out:
        for kind, p in build_rul_artifacts(result, args.rul_out).items():
            print(f"[rul] wrote {kind}: {p}")

    if args.alert_econ_out:
        paths = build_reference_artifacts(result, args.alert_econ_out)
        for kind, p in paths.items():
            print(f"[alert-econ] wrote {kind}: {p}")

    if args.deliverables:
        sizes = build_deliverables(result, args.outdir)
        ok = True
        for path, size in sizes.items():
            status = "OK" if size > 10 * 1024 else "TOO SMALL"
            if size <= 10 * 1024:
                ok = False
            print(f"[deliverable] {Path(path).name}: {size / 1024:.1f} KiB [{status}]")
        if not ok:
            print("[fail] a deliverable is under 10 KiB")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
