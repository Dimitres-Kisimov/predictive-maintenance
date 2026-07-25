"""Maintenance scheduling with CP-SAT, compared against a FIFO baseline.

Given the most urgent machines (jobs with an integer urgency weight and a
service duration in hours), M crews, and a working window of `day_hours` per
day, we minimize total weighted delay = sum(weight * completion_hour), where
completion is measured in compressed working hours from the start of the
horizon. Each job runs within a single day's window and each crew does one
job at a time (no-overlap).

The named baseline is FIFO: jobs in request order (machine id order), each
assigned to the earliest-free crew, no reordering by urgency. The report is
the measured weighted-delay reduction of CP-SAT vs that baseline.

CP-SAT proves optimality on these instance sizes (status OPTIMAL); a small
fixed instance is asserted proven-optimal in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass

from ortools.sat.python import cp_model


@dataclass(frozen=True)
class Job:
    machine: int
    duration_h: int
    weight: int  # integer urgency weight, >= 1


@dataclass(frozen=True)
class Assignment:
    machine: int
    crew: int
    day: int
    start_h: int  # hour offset within the day window
    duration_h: int
    completion_h: int  # compressed working-hour timeline


@dataclass
class ScheduleResult:
    assignments: list[Assignment]
    weighted_delay: int
    status: str


def _weighted_delay(assignments: list[Assignment], jobs: list[Job]) -> int:
    w = {j.machine: j.weight for j in jobs}
    return sum(w[a.machine] * a.completion_h for a in assignments)


def solve_schedule(
    jobs: list[Job],
    n_crews: int = 2,
    n_days: int = 4,
    day_hours: int = 8,
    time_limit_s: float = 10.0,
) -> ScheduleResult:
    """CP-SAT: minimize sum(weight * completion) with per-crew no-overlap."""
    for j in jobs:
        if j.duration_h > day_hours:
            raise ValueError(f"job on machine {j.machine} longer than a day window")
    model = cp_model.CpModel()
    horizon = n_days * day_hours
    starts, ends, crew_lits = [], [], []
    intervals_by_crew: list[list[cp_model.IntervalVar]] = [[] for _ in range(n_crews)]
    for i, job in enumerate(jobs):
        day = model.new_int_var(0, n_days - 1, f"day_{i}")
        off = model.new_int_var(0, day_hours - job.duration_h, f"off_{i}")
        start = model.new_int_var(0, horizon - 1, f"start_{i}")
        end = model.new_int_var(1, horizon, f"end_{i}")
        model.add(start == day * day_hours + off)
        model.add(end == start + job.duration_h)
        lits = [model.new_bool_var(f"crew_{i}_{c}") for c in range(n_crews)]
        model.add_exactly_one(lits)
        for c, lit in enumerate(lits):
            intervals_by_crew[c].append(
                model.new_optional_interval_var(start, job.duration_h, end, lit, f"iv_{i}_{c}")
            )
        starts.append((day, off, start))
        ends.append(end)
        crew_lits.append(lits)
    for c in range(n_crews):
        model.add_no_overlap(intervals_by_crew[c])
    model.minimize(sum(job.weight * end for job, end in zip(jobs, ends, strict=True)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = 0
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"no schedule found (status {solver.status_name(status)})")

    assignments = []
    for i, job in enumerate(jobs):
        day, off, start = starts[i]
        crew = next(c for c, lit in enumerate(crew_lits[i]) if solver.value(lit))
        assignments.append(
            Assignment(
                machine=job.machine,
                crew=crew,
                day=solver.value(day),
                start_h=solver.value(off),
                duration_h=job.duration_h,
                completion_h=solver.value(ends[i]),
            )
        )
    assignments.sort(key=lambda a: (a.crew, a.completion_h))
    return ScheduleResult(
        assignments=assignments,
        weighted_delay=_weighted_delay(assignments, jobs),
        status=solver.status_name(status),
    )


def fifo_schedule(
    jobs: list[Job], n_crews: int = 2, n_days: int = 4, day_hours: int = 8
) -> ScheduleResult:
    """Named baseline: jobs in request (list) order to the earliest-free crew."""
    crew_free = [0] * n_crews  # compressed working-hour timeline
    assignments = []
    for job in jobs:
        if job.duration_h > day_hours:
            raise ValueError(f"job on machine {job.machine} longer than a day window")
        crew = min(range(n_crews), key=lambda c: crew_free[c])
        start = crew_free[crew]
        day, off = divmod(start, day_hours)
        if off + job.duration_h > day_hours:  # does not fit today -> next day
            day += 1
            off = 0
            start = day * day_hours
        if day >= n_days:
            raise RuntimeError("FIFO schedule exceeds the horizon")
        end = start + job.duration_h
        crew_free[crew] = end
        assignments.append(
            Assignment(
                machine=job.machine,
                crew=crew,
                day=day,
                start_h=off,
                duration_h=job.duration_h,
                completion_h=end,
            )
        )
    return ScheduleResult(
        assignments=assignments,
        weighted_delay=_weighted_delay(assignments, jobs),
        status="FIFO",
    )


@dataclass
class ScheduleComparison:
    jobs: list[Job]
    optimized: ScheduleResult
    fifo: ScheduleResult
    n_crews: int
    n_days: int
    day_hours: int

    @property
    def reduction_pct(self) -> float:
        if self.fifo.weighted_delay == 0:
            return 0.0
        return 100.0 * (1.0 - self.optimized.weighted_delay / self.fifo.weighted_delay)


def compare_schedules(
    jobs: list[Job], n_crews: int = 2, n_days: int = 4, day_hours: int = 8
) -> ScheduleComparison:
    return ScheduleComparison(
        jobs=jobs,
        optimized=solve_schedule(jobs, n_crews=n_crews, n_days=n_days, day_hours=day_hours),
        fifo=fifo_schedule(jobs, n_crews=n_crews, n_days=n_days, day_hours=day_hours),
        n_crews=n_crews,
        n_days=n_days,
        day_hours=day_hours,
    )


def small_proven_instance() -> tuple[list[Job], ScheduleResult]:
    """Fixed 5-job / 2-crew instance small enough for CP-SAT to prove OPTIMAL."""
    jobs = [
        Job(machine=0, duration_h=4, weight=2),
        Job(machine=1, duration_h=2, weight=9),
        Job(machine=2, duration_h=6, weight=4),
        Job(machine=3, duration_h=3, weight=7),
        Job(machine=4, duration_h=2, weight=1),
    ]
    return jobs, solve_schedule(jobs, n_crews=2, n_days=3, day_hours=8)
