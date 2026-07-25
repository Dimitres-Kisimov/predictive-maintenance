from __future__ import annotations

from pdm.schedule import (
    Job,
    compare_schedules,
    fifo_schedule,
    small_proven_instance,
    solve_schedule,
)

JOBS = [
    Job(machine=0, duration_h=4, weight=3),
    Job(machine=1, duration_h=2, weight=8),
    Job(machine=2, duration_h=6, weight=5),
    Job(machine=3, duration_h=3, weight=2),
    Job(machine=4, duration_h=5, weight=9),
    Job(machine=5, duration_h=2, weight=4),
]


def _check_no_overlap_and_windows(result, n_crews, n_days, day_hours):
    by_crew = {c: [] for c in range(n_crews)}
    for a in result.assignments:
        assert 0 <= a.crew < n_crews
        assert 0 <= a.day < n_days
        assert a.start_h >= 0
        assert a.start_h + a.duration_h <= day_hours  # fits inside one day window
        start = a.day * day_hours + a.start_h
        by_crew[a.crew].append((start, start + a.duration_h))
    for intervals in by_crew.values():
        intervals.sort()
        for (_, end_prev), (start_next, _) in zip(intervals, intervals[1:], strict=False):
            assert start_next >= end_prev  # one job at a time per crew


def test_cpsat_schedule_respects_crews_and_windows():
    res = solve_schedule(JOBS, n_crews=2, n_days=4, day_hours=8)
    assert len(res.assignments) == len(JOBS)
    _check_no_overlap_and_windows(res, n_crews=2, n_days=4, day_hours=8)


def test_fifo_schedule_respects_crews_and_windows():
    res = fifo_schedule(JOBS, n_crews=2, n_days=4, day_hours=8)
    _check_no_overlap_and_windows(res, n_crews=2, n_days=4, day_hours=8)


def test_cpsat_never_worse_than_fifo():
    comp = compare_schedules(JOBS, n_crews=2, n_days=4, day_hours=8)
    assert comp.optimized.weighted_delay <= comp.fifo.weighted_delay
    assert comp.reduction_pct >= 0.0


def test_small_instance_proven_optimal():
    jobs, res = small_proven_instance()
    assert res.status == "OPTIMAL"
    _check_no_overlap_and_windows(res, n_crews=2, n_days=3, day_hours=8)
    assert res.weighted_delay <= fifo_schedule(jobs, n_crews=2, n_days=3).weighted_delay


def test_pipeline_schedule_beats_fifo_and_is_valid(pipeline_result):
    sched = pipeline_result.schedule
    assert sched.optimized.status in ("OPTIMAL", "FEASIBLE")
    assert sched.optimized.weighted_delay <= sched.fifo.weighted_delay
    _check_no_overlap_and_windows(
        sched.optimized, n_crews=sched.n_crews, n_days=sched.n_days, day_hours=sched.day_hours
    )
