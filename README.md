# predictive-maintenance

I built this project to work through a full predictive-maintenance loop the way an
operations team would actually use it: watch machine sensors, flag the machines that
are drifting toward failure, rank them by urgency, and then schedule the maintenance
crews so the most urgent work happens first. Everything runs locally on a synthetic,
seeded dataset — I say that up front because honest framing matters more to me than
impressive-sounding numbers.

## The operations situation

Unplanned downtime is the expensive kind. When a pump or compressor fails without
warning, you pay for the emergency repair, the idle line behind it, and the scramble
to reshuffle crews. Planned maintenance is far cheaper — but only if you know *which*
machines to service *when*, and your crews are a scarce resource. That gives two
coupled problems:

1. **Detection** — notice a machine degrading before it fails, from its sensor data.
2. **Scheduling** — turn the resulting urgency list into a crew schedule that services
   the riskiest machines first, within real capacity limits.

This repo does both, end to end.

## What is in here

- `pdm/data.py` — seeded generator: 20 machines x 60 days of correlated multivariate
  sensor readings (temperature, vibration, current, pressure) with daily cycles.
  10 machines get faults with known onset days: slow drift, sudden spikes,
  correlation breaks, accelerating drift. Ground truth is exposed for evaluation.
- `pdm/features.py` — rolling-window features per machine/day (mean, std, slope,
  FFT band powers, pairwise sensor correlations). Features at day *t* use only data
  up to *t* — a test rebuilds features on truncated data and asserts they are identical.
- `pdm/detect.py` — two detectors trained on healthy-only windows and compared honestly:
  a numpy PCA-SVD reconstruction-error baseline and a small PyTorch autoencoder.
  Thresholds come from a clean validation split at a stated 5% false-positive-rate
  budget. Per-feature error explains each alert ("flagged on vibration + temperature").
- `pdm/health.py` — a per-machine health index (0-100) from the smoothed anomaly-score
  trend. **This is a heuristic degradation score, not a certified remaining-useful-life
  prediction**, and it is labelled that way everywhere it appears.
- `pdm/schedule.py` — CP-SAT maintenance scheduling: urgency-weighted jobs, 2 crews,
  8-hour day windows, no overlap per crew, minimizing total weighted delay. Compared
  against a named FIFO baseline (jobs in request order, earliest-free crew).
- `pdm/alert_economics.py` — a decision layer on top of the detector: sweep the
  alert threshold over the model's risk scores on the held-out machine-days and,
  at each operating point, report precision/recall, **alerts per day** (the analyst
  workload / alert-fatigue axis), missed failures, and an expected cost
  (`cost_missed_failure·FN + cost_false_alarm·FP`). It returns the cost-minimising
  threshold and a plain-language recommendation. The cost rates are illustrative
  synthetic assumptions, not a business figure.
- `pdm/exports.py` — an executive PDF (cover with disclaimer, PR curves, health
  ranking, before/after Gantt, alert-economics cost curve) and an Excel workbook
  (Machines, Alerts, HealthIndex, Schedule, Comparison, AlertEconomics).

## Measured results (seed 42, default config)

Both detectors are evaluated on machine-days the models never trained on, against the
generator's ground-truth onset labels.

| Metric | PCA (baseline) | Autoencoder |
|---|---|---|
| ROC-AUC | **0.937** | 0.921 |
| PR-AUC | **0.926** | 0.909 |
| precision@50 | 1.000 | 1.000 |
| Machines detected | 10/10 | 10/10 |
| Mean detection delay | **3.4 days** after true onset | 4.8 days |
| Validation FPR (5% budget) | 5.4% | 5.4% |

**I recommend the PCA baseline.** The autoencoder did not beat it — the PR-AUC margin
is -0.017 (AE minus PCA) — and PCA is simpler, faster, and easier to explain. My
policy was fixed before measuring: recommend the simpler detector unless the other
wins by more than 0.03 PR-AUC. It didn't, so the fancier model loses. If the numbers
had gone the other way, this section would say so.

**Scheduling:** on the top-8 urgent machines with 2 crews over 4 days, the CP-SAT
schedule reaches total weighted delay **778 vs 1013 for FIFO — a 23.2% reduction** —
and the solver proves the schedule **OPTIMAL** (not just better). A separate small
fixed instance is asserted proven-optimal in the tests.

## Alert-threshold economics (alert fatigue)

A good detector still needs a threshold, and picking it is a business decision, not
a modelling one: too low and analysts drown in false alarms (alert fatigue); too
high and real failures slip through unalerted. `pdm/alert_economics.py` sweeps the
threshold over the recommended detector's risk scores on the held-out machine-days
and books an expected cost at each operating point.

The pipeline's default threshold is set to a **5% false-positive budget** — a
*statistical* choice, not an economic one. Under an **illustrative** cost assumption
that a missed failure is 20× the cost of a false alarm an analyst can dismiss
(`cost_missed_failure=1000`, `cost_false_alarm=50` — labelled synthetic assumptions,
**not a business guarantee**), the cost-minimising operating point differs:

| Operating point | Threshold | Alerts/day | Missed failure-days | Precision / Recall | Expected cost |
|---|---|---|---|---|---|
| Current (5% FPR) | 0.262 | 4.5 | 41 / 247 | 0.82 / 0.83 | 43,200 |
| **Cost-minimising** | **0.124** | **8.4** | **8 / 247** | **0.51 / 0.97** | **19,550** |

Read this as the *shape* of the coverage/fatigue trade-off, not a currency amount:
under these illustrative costs the analysis recommends alerting **more aggressively**
than the statistical default — roughly doubling analyst load (4.5 → 8.4 alerts/day
across the 13 monitored machines) to cut missed failure-days from 41 to 8. The
cost-minimising threshold moves with the cost ratio, so plug in your own verified
costs before acting.

The cost curve is drawn in [docs/alert_economics.svg](docs/alert_economics.svg) and
the full sweep is in [docs/alert_economics.csv](docs/alert_economics.csv); both are
also folded into the PDF report (a cost-curve page) and the Excel workbook (an
`AlertEconomics` sheet). Regenerate them with `python -m pdm --alert-econ-out docs`.

## How to run

```
pip install -r requirements.txt
python -m pytest -q               # 29 tests
python -m ruff check .            # lint gate
python -m pdm                     # run the pipeline, print the summary above
python -m pdm --deliverables      # also write deliverables/pdm_report.pdf + pdm_workbook.xlsx
python -m pdm --alert-econ-out docs  # regenerate docs/alert_economics.{svg,csv}
```

Torch note: CI installs the CPU wheel best-effort; if it is unavailable, the
autoencoder tests skip via `importorskip` and the numpy-based suite still gates.

## Honest limitations

- **The data is synthetic.** The fault signatures are ones I designed, so the
  detectors are being tested against my own assumptions about what failure looks
  like. Real telemetry has messier faults, sensor dropouts, and regime changes this
  generator does not model. The numbers above measure the pipeline's mechanics, not
  field performance.
- **The health index is a heuristic.** It maps the smoothed anomaly score onto 0-100.
  It is useful for ranking urgency; it is not an RUL model and carries no calibrated
  time-to-failure meaning.
- **Detection delay favours my generator.** Onsets are step changes at known days;
  real degradation onset is ambiguous, so "3.4 days after onset" would not transfer
  as-is.
- **The scheduling model is deliberately small** — single-day jobs, uniform crews,
  no travel time or parts inventory. CP-SAT handles richer models, but I kept the
  instance small enough to prove optimality.
- **The alert-economics cost rates are illustrative.** The 20:1 missed-failure /
  false-alarm ratio is a stand-in I chose to show the shape of the trade-off, not a
  measured business figure, and the "cost" units are not currency. The recommended
  threshold is only as trustworthy as those rates and the synthetic base rate of
  failures — swap in your own verified costs before drawing any conclusion. Because
  every post-onset day counts as a positive here, "missed failure-days" is a
  machine-day count, not a count of distinct failure events.

## License

© 2026 Dimitres Kisimov — all rights reserved; published for portfolio review. See LICENSE. Credits for the libraries I built on are in
[CREDITS.md](CREDITS.md). The business framing is in
[docs/BUSINESS_CASE.md](docs/BUSINESS_CASE.md).
