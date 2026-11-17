"""The P10/P90 band: what it learns from, and what it refuses to."""

import math

import numpy as np
import pytest

from custom_components.soular.core.band import (
    BAND_LEADS,
    MAX_TICK_SECONDS,
    MIN_SAMPLES,
    PRIOR,
    WINDOW_DAYS,
    BandTracker,
    Residual,
    Snapshot,
    bounds_for,
    decode,
    encode,
    hourly_snapshot,
    scale_by_day,
)

DAY = 86400
HOUR = 3600
FIRST = 740000  # an arbitrary date ordinal


def flat_snapshot(day: int, start: int, watts: float) -> Snapshot:
    """Build a forecast of constant power for the whole week, to keep the arithmetic obvious."""
    return Snapshot(issued_day=day, start=start, hourly_w=[watts] * (BAND_LEADS * 24))


def run_day(tracker: BandTracker, day: int, start: int, measured: float | None, *, every: int = 300) -> None:
    """Begin a day and feed it a reading every ``every`` seconds."""
    tracker.start_day(day, start, start + DAY, flat_snapshot(day, start, 1000.0))
    for now in range(start, start + DAY, every):
        tracker.tick(now, measured)


def test_too_few_days_reports_the_prior() -> None:
    """A lead with little history keeps the starting band, and says so."""
    residuals = [Residual(day=FIRST + i, lead=1, log_ratio=0.0) for i in range(MIN_SAMPLES - 1)]
    bounds = bounds_for(residuals, lead=1, today=FIRST + MIN_SAMPLES)
    assert (bounds.low, bounds.high) == PRIOR[1]
    assert not bounds.calibrated


def test_enough_days_reports_their_own_percentiles() -> None:
    """With enough history the band is the tenth and ninetieth percentile of the ratios."""
    ratios = np.linspace(0.5, 1.5, 41)
    residuals = [Residual(day=FIRST + i, lead=2, log_ratio=math.log(r)) for i, r in enumerate(ratios)]
    bounds = bounds_for(residuals, lead=2, today=FIRST + len(ratios))
    assert bounds.calibrated
    assert bounds.low == pytest.approx(math.exp(np.quantile(np.log(ratios), 0.1)))
    assert bounds.high == pytest.approx(math.exp(np.quantile(np.log(ratios), 0.9)))


def test_only_the_past_and_only_the_window_count() -> None:
    """Today's outcome and anything older than the window are both excluded."""
    old = [Residual(day=FIRST + i, lead=1, log_ratio=5.0) for i in range(MIN_SAMPLES)]
    recent = [Residual(day=FIRST + WINDOW_DAYS + i, lead=1, log_ratio=0.0) for i in range(MIN_SAMPLES)]
    today = FIRST + WINDOW_DAYS + MIN_SAMPLES
    future = [Residual(day=today, lead=1, log_ratio=5.0)]
    bounds = bounds_for(old + recent + future, lead=1, today=today)
    assert bounds.samples == MIN_SAMPLES
    assert bounds.high == pytest.approx(1.0)


def test_a_day_becomes_one_ratio_per_lead() -> None:
    """Each lead whose snapshot covered the day contributes actual over forecast."""
    tracker = BandTracker()
    start = 1_800_000_000 - 1_800_000_000 % DAY
    # Snapshots issued on the two previous days, then today's.
    tracker.snapshots[FIRST - 2] = flat_snapshot(FIRST - 2, start - 2 * DAY, 2000.0)
    tracker.snapshots[FIRST - 1] = flat_snapshot(FIRST - 1, start - DAY, 500.0)
    run_day(tracker, FIRST, start, measured=1000.0)
    tracker.start_day(FIRST + 1, start + DAY, start + 2 * DAY, None)

    ratios = {r.lead: math.exp(r.log_ratio) for r in tracker.residuals}
    assert ratios == {
        0: pytest.approx(1.0, rel=1e-3),
        1: pytest.approx(2.0, rel=1e-3),
        2: pytest.approx(0.5, rel=1e-3),
    }


def test_curtailed_instants_are_left_out_of_both_sides() -> None:
    """Half a day curtailed still gives the true ratio, from the other half."""
    tracker = BandTracker()
    start = 1_800_000_000 - 1_800_000_000 % DAY
    tracker.start_day(FIRST, start, start + DAY, flat_snapshot(FIRST, start, 1000.0))
    for now in range(start, start + DAY, 300):
        # The first fifth of the day is curtailed and reads low; it must not count.
        tracker.tick(now, None if now < start + DAY // 5 else 800.0)
    tracker.start_day(FIRST + 1, start + DAY, start + 2 * DAY, None)
    (residual,) = tracker.residuals
    assert math.exp(residual.log_ratio) == pytest.approx(0.8, rel=1e-3)


def test_a_mostly_unusable_day_says_nothing() -> None:
    """Below the usable share, the day is dropped rather than half-measured."""
    tracker = BandTracker()
    start = 1_800_000_000 - 1_800_000_000 % DAY
    tracker.start_day(FIRST, start, start + DAY, flat_snapshot(FIRST, start, 1000.0))
    for now in range(start, start + DAY, 300):
        tracker.tick(now, 1000.0 if now < start + DAY // 2 else None)
    tracker.start_day(FIRST + 1, start + DAY, start + 2 * DAY, None)
    assert tracker.residuals == []


def test_a_gap_in_readings_adds_nothing() -> None:
    """An outage is not credited as one long reading."""
    tracker = BandTracker()
    start = 1_800_000_000 - 1_800_000_000 % DAY
    tracker.start_day(FIRST, start, start + DAY, flat_snapshot(FIRST, start, 1000.0))
    tracker.tick(start, 1000.0)
    tracker.tick(start + MAX_TICK_SECONDS + 1, 1000.0)
    assert tracker.ledger is not None
    assert tracker.ledger.actual_wh == 0.0


def test_a_restart_keeps_the_midnight_snapshot() -> None:
    """Starting the same day again, with a later forecast, changes nothing."""
    tracker = BandTracker()
    start = 1_800_000_000 - 1_800_000_000 % DAY
    tracker.start_day(FIRST, start, start + DAY, flat_snapshot(FIRST, start, 1000.0))
    tracker.tick(start + HOUR, 900.0)
    tracker.tick(start + HOUR + 300, 900.0)
    restored = decode(encode(tracker))
    restored.start_day(FIRST, start, start + DAY, flat_snapshot(FIRST, start, 5000.0))
    assert restored.snapshots[FIRST].hourly_w[0] == 1000.0
    assert restored.ledger is not None
    assert restored.ledger.actual_wh == pytest.approx(75.0)


def test_old_snapshots_and_residuals_are_pruned() -> None:
    """Storage stays bounded: a week of snapshots and the window of residuals."""
    tracker = BandTracker()
    start = 1_800_000_000 - 1_800_000_000 % DAY
    tracker.residuals = [Residual(day=FIRST - WINDOW_DAYS - BAND_LEADS - 1, lead=1, log_ratio=0.0)]
    for offset in range(BAND_LEADS + 3):
        tracker.start_day(FIRST + offset, start + offset * DAY, start + (offset + 1) * DAY, flat_snapshot(0, 0, 1.0))
    assert len(tracker.snapshots) == BAND_LEADS
    assert tracker.residuals == []


def test_round_trip_and_garbage() -> None:
    """Encoding round-trips, and an unreadable payload starts afresh."""
    tracker = BandTracker()
    start = 1_800_000_000 - 1_800_000_000 % DAY
    run_day(tracker, FIRST, start, 1000.0)
    tracker.start_day(FIRST + 1, start + DAY, start + 2 * DAY, None)
    restored = decode(encode(tracker))
    assert restored.residuals == [
        Residual(day=r.day, lead=r.lead, log_ratio=round(r.log_ratio, 5)) for r in tracker.residuals
    ]
    assert decode({"snapshots": [{"nonsense": 1}]}).snapshots == {}
    assert decode(None).residuals == []


def test_hourly_snapshot_keeps_the_energy() -> None:
    """Hourly means of a series integrate to the same energy."""
    start = 1_800_000_000 - 1_800_000_000 % DAY
    times = np.arange(start, start + 2 * DAY + 1, 300).astype("datetime64[s]")
    power = np.full(times.size, 1500.0)
    snapshot = hourly_snapshot(FIRST, start, times, power, hours=48)
    assert snapshot.hourly_w == pytest.approx([1500.0] * 48)
    assert snapshot.energy_wh(start, start + DAY) == pytest.approx(36000.0)


def test_scale_by_day_uses_each_instants_own_day() -> None:
    """The P10 series is each day's forecast scaled by that day's ratio."""
    midnights = tuple(np.datetime64(1_800_000_000 + i * DAY, "s") for i in range(3))
    times = np.array([midnights[0] - 1, midnights[0], midnights[1] + 10, midnights[2] + 10], dtype="datetime64[s]")
    assert scale_by_day(times, midnights, (0.5, 0.7)).tolist() == [0.5, 0.5, 0.7, 0.7]


def test_a_lead_pools_with_its_neighbours_only() -> None:
    """Lead 3 draws on leads 2 to 4, and nothing further out."""
    near = [Residual(day=FIRST + i, lead=lead, log_ratio=0.0) for i in range(5) for lead in (2, 3, 4)]
    far = [Residual(day=FIRST + i, lead=lead, log_ratio=3.0) for i in range(30) for lead in (1, 5)]
    bounds = bounds_for(near + far, lead=3, today=FIRST + 30)
    assert bounds.samples == 15
    assert (bounds.low, bounds.high) == (pytest.approx(1.0), pytest.approx(1.0))
