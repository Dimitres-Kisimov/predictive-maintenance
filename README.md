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
  prediction**, and it is labelled that way everywhere it appears. The actual
  time-to-failure model lives in `pdm/rul.py`.
- `pdm/rul.py` — remaining-useful-life (RUL) estimation, evaluated against a
  ground-truth failure day rather than asserted. For the two progressive-degradation
  fault kinds it reuses the recommended detector's *own* risk score (EWMA-smoothed,
  expressed as a log-scale exceedance over the alert threshold) and maps it to
  days-to-failure with a one-feature log-linear model. Scored **leave-one-machine-out**
  (each machine predicted by a fit that never saw it) with MAE/RMSE in days, a naive
  mean-RUL baseline, a near-failure error, and the alpha-lambda accuracy cone. The
  failure *level* is an illustrative synthetic assumption, labelled as such.
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
- `pdm/policy.py` — maintenance-policy economics: a right-censored Weibull
  lifetime fit (shape/scale, MTBF, B10/B50) on the fleet's ground-truth failure
  days, the classic **age-replacement optimization** (the cheapest preventive
  replacement age T*), and a **condition-based** policy priced from the
  detector's *measured* alerts — all three policies compared on cost per
  machine-day under labelled illustrative cost rates.
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

## Remaining-useful-life (RUL) estimation

The health index above is deliberately only a heuristic urgency ranking. `pdm/rul.py`
is the actual prognostics layer: it predicts *days to failure* and is **evaluated
against a ground-truth failure day** instead of asserted.

Two of the injected fault kinds — `slow_drift` and `accelerating_drift` — degrade
monotonically, so a failure day is well defined: `PlantData.failure_day` solves the
exact injected growth law for the first day the fault-induced vibration rise reaches
an **illustrative** functional-failure level (`FAILURE_VIB_RISE = 0.45` mm/s, ≈ 5.6×
the vibration sensor noise — a synthetic stand-in, *not* a measured engineering
limit). RUL at day *t* is then `failure_day − t`. Spike and correlation-break faults
have no monotone magnitude, so RUL is undefined for them and they are excluded — the
honest scope for degradation-based RUL.

The predictor **reuses the recommended detector's own risk score** — no second model
on the raw sensors. The daily anomaly score is EWMA-smoothed (the same smoothing the
health index uses) and expressed as a log-scale exceedance over the alert threshold
("severity"); a one-feature log-linear model maps `log1p(RUL) = a + b·severity`.
Critically it uses *only observable* signals and never the true onset day, so it
reflects what would be knowable in production.

Evaluation is **leave-one-machine-out**: each machine's RUL is predicted by a model
fit only on the *other* machines, and RUL is scored only on the degradation phase
(`onset ≤ day ≤ failure_day`) of machines whose failure is observed inside the 60-day
window. Measured on the seed-42 fleet (6 progressive-fault machines, 90 degradation
machine-days):

| Metric | Value |
|---|---|
| MAE | **2.43 days** |
| RMSE | 3.15 days |
| Naive mean-RUL baseline MAE | 4.06 days |
| MAE within the actionable window (true RUL ≤ 10 d) | **1.58 days** |
| alpha-accuracy (within ±(0.5·RUL + 1 d) cone) | 0.87 |
| Mean prognostic horizon (in-cone lead time) | 3.7 days |

The severity-driven model beats the naive mean-RUL baseline by **1.63 days of MAE**
(2.43 vs 4.06), and it is tightest where it matters — inside the last ~10 days before
failure the mean error is 1.58 days. A single global model spans both a slower (drift)
and a faster (accelerating) degradation law, so some of the residual is the price of
not knowing the fault kind at inference — which is the realistic condition.

The predicted-vs-true scatter (with the accuracy cone) is in
[docs/rul_eval.svg](docs/rul_eval.svg) and the full per-machine-day held-out
predictions are in [docs/rul_eval.csv](docs/rul_eval.csv); both are also folded into
the PDF report (a RUL page) and the Excel workbook (a `RUL` sheet). Regenerate them
with `python -m pdm --rul-out docs`.

## Maintenance-policy comparison (Weibull + age replacement + CBM)

Detection, RUL and alert thresholds all feed one final question a maintenance
engineer actually has to answer: **which policy do we run this fleet on?**
`pdm/policy.py` prices three named policies on the same synthetic fleet.

**Lifetime model.** Machine lives come from the generator's ground-truth
functional-failure day (the same illustrative vibration-rise definition the RUL
layer uses). On the seed-42 fleet that gives **6 observed failures** (days 43,
45, 47, 49, 50, 51) and **14 suspensions** right-censored at the 60-day window
end — the standard field-data "failures + suspensions" situation. A
two-parameter Weibull fit by censored maximum likelihood gives:

| Weibull fit (MLE) | Value |
|---|---|
| shape β | **4.81** (β > 1: wear-out — preventive action can pay) |
| scale η | 73.6 days |
| MTBF | 67.4 days |
| B10 life | 46.1 days |
| B50 (median) life | 68.2 days |

These are **modelled, not measured**: one heavily censored synthetic window,
six failures. The fit's own MLE identities and its likelihood optimality are
test-verified; the numbers' field meaning is not claimed.

**Policy pricing** (illustrative cost rates, consistent with the
alert-economics section: unplanned failure 1000, planned replacement 250,
false-alarm inspection 50 — labelled assumptions, not currency):

| Policy | Cost / machine-day | Detail |
|---|---|---|
| Run-to-failure | 14.84 | Cf / MTBF baseline |
| **Age-replacement** | **7.16** | replace at **T\* = 44.4 d** (−51.7% vs run-to-failure) |
| Condition-based | 8.28 | 6/6 failures alerted ≥ 3 d early; +4.57/machine-day false-alarm inspections |

Under these assumptions the recommendation is **age-replacement**, and the
result is honest about why the fancier option loses: the detector converts
*every* failure into a planned repair (6/6 alerted with ≥ 3 days of lead time),
but at the default 5%-FPR threshold its 0.091 false alarms per machine-day book
4.57 of the 8.28 — more than half the policy's cost — in inspections. Meanwhile
β ≈ 4.8 means failures cluster tightly around the characteristic life, which is
exactly the regime where a calendar rule is hard to beat (note T* ≈ B10: replace
just before the wear-out knee). With real fleets' messier lifetime scatter
(lower β) or a better-tuned alert threshold the ranking can flip — the point of
the module is that the trade-off is *computed*, not asserted.

The textbook base case is also test-verified: for memoryless (β = 1) lifetimes
the optimizer reports that **no** finite replacement age beats run-to-failure,
exactly as renewal theory says it must.

The cost-rate curve g(T) with all three policies is drawn in
[docs/policy_comparison.svg](docs/policy_comparison.svg) and the full
comparison + curve is in
[docs/policy_comparison.csv](docs/policy_comparison.csv). Regenerate with
`python -m pdm --policy-out docs`.

## How to run

```
pip install -r requirements.txt
python -m pytest -q               # 63 tests
python -m ruff check .            # lint gate
python -m pdm                     # run the pipeline, print the summary above
python -m pdm --deliverables      # also write deliverables/pdm_report.pdf + pdm_workbook.xlsx
python -m pdm --alert-econ-out docs  # regenerate docs/alert_economics.{svg,csv}
python -m pdm --rul-out docs         # regenerate docs/rul_eval.{svg,csv}
python -m pdm --policy-out docs      # regenerate docs/policy_comparison.{svg,csv}
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
  It is useful for ranking urgency; it is not the RUL model and carries no calibrated
  time-to-failure meaning. The calibrated time-to-failure estimate is the separate
  `pdm/rul.py` layer.
- **The RUL failure level is illustrative, and RUL only covers progressive faults.**
  The 0.45 mm/s functional-failure level is a synthetic stand-in I chose so failures
  land inside the observation window, not a measured engineering limit — every day
  count scales with it, so read the RUL numbers as the *shape* of the
  degradation-to-failure relationship, not calibrated field RUL. RUL is defined only
  for the monotone-degradation faults (`slow_drift`, `accelerating_drift`); spike and
  correlation-break faults have no time-to-failure target and are excluded. With only
  six progressive machines the leave-one-machine-out estimate is honest but
  small-sample, and the reported error would not transfer to messier real telemetry.
- **Detection delay favours my generator.** Onsets are step changes at known days;
  real degradation onset is ambiguous, so "3.4 days after onset" would not transfer
  as-is.
- **The scheduling model is deliberately small** — single-day jobs, uniform crews,
  no travel time or parts inventory. CP-SAT handles richer models, but I kept the
  instance small enough to prove optimality.
- **The policy comparison is a modelled planning exercise, not a certified
  policy.** The Weibull fit rests on six synthetic failures plus fourteen
  suspensions from one observation window, so MTBF and the B-lives are model
  extrapolations with real small-sample uncertainty (no confidence bounds are
  reported). The three cost rates are illustrative assumptions and the policy
  ranking moves with them. The condition-based price divides the per-cycle cost
  by the mean life (ignoring the slightly shortened cycle when a repair happens
  a few days early) and assumes every alert with ≥ 3 days of lead time converts
  the failure into a planned repair — both stated simplifications. Swap in
  measured lifetimes and verified costs before treating any of it as a decision.
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
