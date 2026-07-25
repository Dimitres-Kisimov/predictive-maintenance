# Business case: predictive maintenance for a small plant fleet

*This document frames the project the way it would be pitched internally at a plant.
All fleet data in the project is synthetic and seeded; every cost figure below is a
labelled estimate for illustration, not a measurement.*

## Situation

A plant runs ~20 rotating machines (pumps, compressors, fans) monitored by basic
sensors: temperature, vibration, current draw, pressure. Maintenance is reactive —
machines are serviced when they fail or on a fixed calendar — and two crews handle
all work in 8-hour day shifts. Failures arrive unannounced; the maintenance queue is
handled first-in-first-out.

## Quantified problem (labelled estimates)

- *Estimate:* an unplanned failure on a machine of this class costs **EUR 10k-50k**
  per event (emergency labour, expedited parts, upstream/downstream idle time).
  Industry surveys commonly put unplanned downtime at 3-10x the cost of the same
  work done planned. These are assumptions, not measurements from this project.
- *Estimate:* a fleet of 20 machines with a handful of degrading units sees
  **5-15 unplanned events per year** under reactive maintenance.
- *Measured in this project:* on the synthetic fleet, 10 of 10 degrading machines
  were flagged, on average **3.4 days after true degradation onset** (PCA detector,
  5% false-positive budget) — days of warning during which the work can be planned
  instead of suffered.
- *Measured in this project:* ordering the maintenance queue by urgency with CP-SAT
  instead of FIFO cut total urgency-weighted delay by **23.2%** (778 vs 1013,
  proven optimal by the solver on this instance).

## Solution

A pipeline that runs on data the plant already collects:

1. **Detect** — rolling-window features per machine/day; a PCA reconstruction-error
   detector trained on healthy operation flags abnormal days and names the sensors
   responsible. (A neural autoencoder was evaluated head-to-head and did not beat
   the simpler model; the project recommends PCA and reports the margin.)
2. **Rank** — a 0-100 health index (an explicitly heuristic degradation score, not a
   certified RUL prediction) turns alert streams into an urgency ranking.
3. **Schedule** — a CP-SAT model assigns the most urgent jobs to crews inside real
   capacity windows, minimizing urgency-weighted completion time, with a FIFO
   baseline reported for comparison.

## ROI sketch (illustrative arithmetic on the estimates above)

If early warning converts even 3 unplanned events per year into planned work, at an
*assumed* 3x cost multiple and EUR 20k per unplanned event, that is roughly
**EUR 40k/year avoided** (3 x (20k - 20k/3)) — against a system built from
open-source components on existing sensors. The 23.2% scheduling improvement
compounds this by getting crews to the riskiest machines sooner without adding
headcount. The arithmetic is only as good as the assumptions, which is why they are
labelled.

## Stakeholders

- **Maintenance manager** — owns the urgency ranking and crew schedule; gets the
  Gantt view and the Excel workbook.
- **Plant/operations manager** — owns the downtime budget; gets the executive PDF
  with headline numbers and the disclaimer.
- **Reliability engineer** — owns thresholds and false-positive budget; gets the
  detector comparison, per-sensor explanations, and the alert log.
- **Crews** — get a day plan that fits inside shift windows instead of a queue.

## Deliverable

`python -m pdm --deliverables` produces:

- **pdm_report.pdf** — cover with synthetic-data disclaimer and headline numbers;
  detector PR-curve comparison; fleet health ranking; before/after schedule Gantt.
- **pdm_workbook.xlsx** — Machines, Alerts, HealthIndex, Schedule, and Comparison
  sheets for anyone who wants the numbers behind the charts.

Everything is reproducible from a fixed seed; the repo's tests gate determinism,
leak-free features, detector validity, schedule feasibility, and optimality of the
small reference instance.
