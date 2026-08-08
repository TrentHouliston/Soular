"""Tests for the offline backtest harness.

The harness is the instrument that decides whether the model is any good, so a
bug in it would invalidate every number the project reports without failing
anything. It is exercised here against a synthetic site built in-test, which
means CI covers it without needing the 221 MB production database.
"""

import datetime as dt
from itertools import pairwise
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from tools.backtest import (
    Accumulator,
    Site,
    is_fit_day,
    lead_days,
    load_actuals,
    load_site,
    to_local_dates,
    uniform_grid,
)
from tools.fetch_archive import SCHEMA, date_chunks

SYDNEY = ZoneInfo("Australia/Sydney")


def test_lead_never_uses_a_run_that_had_not_been_issued() -> None:
    """Lead is rounded up, so the harness cannot see the future.

    A run labelled N days before valid time T was issued around ``T - N days``.
    Requiring that to be at or before the issue time forces ``N >= (T - t0)/24h``,
    and rounding down instead would let a forecast up to a day newer than the
    issue time leak in -- which would silently flatter every short-lead number.
    """
    issue = np.datetime64("2026-03-10T06:00:00", "s")
    valid = np.array(
        [
            "2026-03-10T06:00:00",  # the issue instant itself
            "2026-03-10T07:00:00",  # 1 hour out
            "2026-03-11T05:00:00",  # 23 hours out
            "2026-03-11T07:00:00",  # 25 hours out
            "2026-03-12T07:00:00",  # 49 hours out
        ],
        dtype="datetime64[s]",
    )
    np.testing.assert_array_equal(lead_days(issue, valid), [0.0, 1.0, 1.0, 2.0, 3.0])


def test_lead_is_never_negative() -> None:
    """Valid times at or before the issue are lead zero, not a negative lead."""
    issue = np.datetime64("2026-03-10T06:00:00", "s")
    past = np.array(["2026-03-09T06:00:00", "2026-03-10T05:00:00"], dtype="datetime64[s]")
    assert np.all(lead_days(issue, past) == 0.0)


def test_fit_and_holdout_days_are_disjoint_and_both_populated() -> None:
    """The split covers every day exactly once, roughly one in three fitting."""
    days = [int(f"2026{month:02d}{day:02d}") for month in range(1, 13) for day in range(1, 29)]
    fitting = [day for day in days if is_fit_day(day)]
    holdout = [day for day in days if not is_fit_day(day)]

    assert len(fitting) + len(holdout) == len(days)
    assert not set(fitting) & set(holdout)
    assert 0.2 < len(fitting) / len(days) < 0.45


class TestAccumulator:
    """The scoring algebra, which every reported number passes through."""

    def test_gain_recovers_a_known_scale(self) -> None:
        """A prediction that is uniformly 80% of actual fits a gain of 1.25."""
        predicted = np.linspace(100.0, 5000.0, 500)
        accumulator = Accumulator()
        accumulator.add(predicted, predicted * 1.25)
        assert accumulator.gain == pytest.approx(1.25, rel=1e-12)

    def test_rmse_under_the_fitted_gain_is_zero_for_a_pure_scale_error(self) -> None:
        """Re-scaling removes an error that is purely a scale error."""
        predicted = np.linspace(100.0, 5000.0, 500)
        accumulator = Accumulator()
        accumulator.add(predicted, predicted * 0.9)
        assert accumulator.rmse(1.0) > 100.0
        # Not exactly zero: the closed form subtracts terms of order 1e10 to get
        # a residual of order zero, so a perfect fit lands at the noise floor of
        # that cancellation. Irrelevant on real data, where RMSE is thousands of
        # watts and the cancellation is nowhere near total.
        assert accumulator.rmse(accumulator.gain) == pytest.approx(0.0, abs=1e-3)

    def test_cross_moments_match_direct_computation(self) -> None:
        """The closed form agrees with computing the error series directly."""
        rng = np.random.default_rng(7)
        predicted = rng.uniform(0.0, 8000.0, 1000)
        actual = predicted * 0.95 + rng.normal(0.0, 300.0, 1000)

        accumulator = Accumulator()
        accumulator.add(predicted, actual)
        for gain in (0.8, 1.0, 1.2):
            direct = float(np.sqrt(np.mean((gain * predicted - actual) ** 2)))
            assert accumulator.rmse(gain) == pytest.approx(direct, rel=1e-9)
            assert accumulator.mbe(gain) == pytest.approx(float(np.mean(gain * predicted - actual)), rel=1e-9)

    def test_the_fitted_gain_minimises_rmse(self) -> None:
        """No other scalar scores better, which is what makes the ablation fair."""
        rng = np.random.default_rng(11)
        predicted = rng.uniform(0.0, 8000.0, 2000)
        accumulator = Accumulator()
        accumulator.add(predicted, predicted * 1.1 + rng.normal(0.0, 200.0, 2000))

        best = accumulator.gain
        assert accumulator.rmse(best) <= min(accumulator.rmse(best + delta) for delta in (-0.05, -0.01, 0.01, 0.05))

    def test_non_finite_samples_are_skipped(self) -> None:
        """Gaps in the record must not poison the statistics."""
        predicted = np.array([100.0, np.nan, 300.0])
        actual = np.array([100.0, 200.0, np.nan])
        accumulator = Accumulator()
        accumulator.add(predicted, actual)
        assert accumulator.n == 1


def test_local_dates_follow_daylight_saving() -> None:
    """Local day boundaries move with the clock, not with a fixed offset.

    This site shifts by an hour twice a year. Binning daily energy on a fixed
    offset would misfile a day's production either side of each transition, and
    the inverter's own counters are local-day totals.
    """
    # 2026-04-05 02:00 local is when Sydney leaves daylight saving.
    times = np.array(
        ["2026-01-14T13:30:00", "2026-01-14T12:30:00", "2026-07-14T13:30:00", "2026-07-14T14:30:00"],
        dtype="datetime64[s]",
    )
    days = to_local_dates(times, SYDNEY)
    # UTC+11 in January: 13:30Z is the 15th local, 12:30Z is still the 14th.
    assert days[0] == 20260115
    assert days[1] == 20260114
    # UTC+10 in July: the boundary sits an hour later.
    assert days[2] == 20260714
    assert days[3] == 20260715


def build_synthetic_db(path: Path, times: np.ndarray) -> None:
    """Write a small Sigen-shaped database with a known channel swap."""
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE measurements (
            station_id TEXT, device_sn TEXT, metric TEXT, ts_utc TEXT,
            value REAL, unit TEXT, source_endpoint TEXT
        );
        CREATE TABLE daily_totals (
            station_id TEXT, date_local TEXT, metric TEXT, value REAL, unit TEXT, source_endpoint TEXT
        );
        """
    )
    rows = []
    for position, stamp in enumerate(times):
        text = f"{np.datetime_as_string(stamp, unit='s')}+00:00"
        # A distinguishable constant per channel, so a mis-mapped swap is visible.
        for channel, level in (("PV1_POWER", 1.0), ("PV2_POWER", 2.0), ("PV3_POWER", 3.0), ("PV4_POWER", 4.0)):
            rows.append(("s", "d", channel, text, level, "kW", "t"))
        rows.append(("s", "d", "FROM_SOLAR", text, 10.0, "kW", "t"))
        rows.append(("s", "d", "SOC", text, 50.0 + (position % 2), "%", "t"))
    connection.executemany("INSERT INTO measurements VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    connection.execute("INSERT INTO daily_totals VALUES ('s', '20260618', 'FROM_SOLAR', 40.0, 'kWh', 't')")
    connection.commit()
    connection.close()


def test_channel_swap_is_honoured(tmp_path: Path) -> None:
    """PV1 and PV2 change arrays when the MPPT inputs were re-plugged.

    Getting this wrong makes east and west look mysteriously broken for the last
    seven weeks of the record, in a way that reads as a shading error.
    """
    times = uniform_grid(dt.date(2026, 6, 16), dt.date(2026, 6, 22))
    db = tmp_path / "solar.db"
    build_synthetic_db(db, times)

    actuals = load_actuals(db, times, ["east", "west", "north", "south"])

    swap = np.datetime64("2026-06-19", "s")
    before = times < swap - np.timedelta64(86400, "s")
    after = times > swap + np.timedelta64(86400, "s")

    # PV1 feeds east before the swap and west after it; PV2 the other way round.
    assert np.nanmedian(actuals.dc_by_array["east"][before]) == pytest.approx(1000.0)
    assert np.nanmedian(actuals.dc_by_array["west"][before]) == pytest.approx(2000.0)
    assert np.nanmedian(actuals.dc_by_array["east"][after]) == pytest.approx(2000.0)
    assert np.nanmedian(actuals.dc_by_array["west"][after]) == pytest.approx(1000.0)
    # Unaffected channels stay put.
    assert np.nanmedian(actuals.dc_by_array["north"][before]) == pytest.approx(3000.0)
    assert np.nanmedian(actuals.dc_by_array["south"][after]) == pytest.approx(4000.0)


def test_days_around_the_swap_are_excluded(tmp_path: Path) -> None:
    """The exact hour of the re-plug is unrecorded, so those days are dropped."""
    times = uniform_grid(dt.date(2026, 6, 16), dt.date(2026, 6, 22))
    db = tmp_path / "solar.db"
    build_synthetic_db(db, times)

    actuals = load_actuals(db, times, ["east", "west", "north", "south"])
    swap = np.datetime64("2026-06-19", "s")
    buffer = np.timedelta64(86400, "s")
    near = (times >= swap - buffer) & (times < swap + buffer)

    assert not actuals.valid[near].any()
    assert actuals.valid[~near].all()


def test_kilowatts_are_converted_to_watts(tmp_path: Path) -> None:
    """The database is in kW and the model is in W."""
    times = uniform_grid(dt.date(2026, 1, 10), dt.date(2026, 1, 11))
    db = tmp_path / "solar.db"
    build_synthetic_db(db, times)

    actuals = load_actuals(db, times, ["east", "west", "north", "south"])
    assert np.nanmedian(actuals.ac_w) == pytest.approx(10_000.0)
    # State of charge is a percentage and must survive the conversion unscaled.
    assert 50.0 <= float(np.nanmedian(actuals.soc_pct)) <= 51.0


def test_curtailed_samples_are_identified(tmp_path: Path) -> None:
    """A full battery curtails production, which no forecast can predict."""
    times = uniform_grid(dt.date(2026, 1, 10), dt.date(2026, 1, 10))
    db = tmp_path / "solar.db"
    connection = sqlite3.connect(db)
    connection.executescript(
        """
        CREATE TABLE measurements (
            station_id TEXT, device_sn TEXT, metric TEXT, ts_utc TEXT,
            value REAL, unit TEXT, source_endpoint TEXT
        );
        CREATE TABLE daily_totals (
            station_id TEXT, date_local TEXT, metric TEXT, value REAL, unit TEXT, source_endpoint TEXT
        );
        """
    )
    rows = []
    for position, stamp in enumerate(times):
        text = f"{np.datetime_as_string(stamp, unit='s')}+00:00"
        rows.append(("s", "d", "SOC", text, 99.0 if position % 2 else 50.0, "%", "t"))
    connection.executemany("INSERT INTO measurements VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    connection.commit()
    connection.close()

    actuals = load_actuals(db, times, ["east"])
    curtailed = actuals.curtailed()
    assert curtailed.any()
    assert not curtailed.all()


def test_date_chunks_cover_the_range_without_gaps_or_overlap() -> None:
    """The fetcher must request every day exactly once."""
    start, end = dt.date(2025, 10, 16), dt.date(2026, 8, 8)
    chunks = list(date_chunks(start, end, 45))

    assert chunks[0][0] == start
    assert chunks[-1][1] == end
    for (_, previous_end), (next_start, _) in pairwise(chunks):
        assert next_start == previous_end + dt.timedelta(days=1)
    assert sum((stop - begin).days + 1 for begin, stop in chunks) == (end - start).days + 1


def test_cache_schema_applies_cleanly(tmp_path: Path) -> None:
    """The cache schema is idempotent, so re-running the fetcher is safe."""
    path = tmp_path / "cache.db"
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    connection.executescript(SCHEMA)
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"raw_responses", "nwp", "satellite", "ensemble", "meta"} <= tables
    connection.close()


def test_site_config_parses_the_real_layout(tmp_path: Path) -> None:
    """``arrays.toml`` maps onto the core's specs, timezone included."""
    path = tmp_path / "arrays.toml"
    path.write_text(
        """
        [site]
        latitude = -33.11915471966274
        longitude = 151.53401076793673
        timezone = "Australia/Sydney"
        altitude = 10.0

        [arrays.east]
        azimuth = 84.0
        tilt = 25.0
        dc_watts = 7920

        [arrays.north]
        azimuth = 354.0
        tilt = 25.0
        dc_watts = 5720
        """
    )
    site: Site = load_site(path, ac_limit_w=20000.0)

    assert site.spec.latitude == pytest.approx(-33.11915471966274)
    assert site.timezone == SYDNEY
    assert [array.name for array in site.arrays] == ["east", "north"]
    assert site.arrays[0].dc_capacity_w == 7920.0
    assert site.inverter.ac_limit_w == 20000.0
