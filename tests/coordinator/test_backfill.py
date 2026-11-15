"""Catching a fresh array's correction up from recorded history."""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
import numpy as np
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.soular.const import CONF_POWER_SENSOR
from custom_components.soular.coordinator.actuals import Learner, samples_from
from custom_components.soular.coordinator.backfill import (
    BACKFILL_DAYS,
    PERIOD_SECONDS,
    arrays_to_backfill,
    async_read_history,
    backfill_window,
    replay,
)
from custom_components.soular.core.correction import build_features
from custom_components.soular.core.nwp import weather_from_hourly
from custom_components.soular.core.pipeline import SystemSpec, forecast
from custom_components.soular.core.types import ArraySpec, TimeGrid
from custom_components.soular.sources.open_meteo import HourlyWeather
from tests.core.conftest import INVERTER, SITE
from tests.integration.conftest import ARRAY_DATA, build_entry, synthetic_weather

EAST = ArraySpec(name="east", azimuth_deg=84.0, tilt_deg=25.0, dc_capacity_w=7920.0)
SYSTEM = SystemSpec(site=SITE, arrays=(EAST,), inverters={"default": INVERTER})
EFFICIENCY = 0.9


def recorded_week(weather: HourlyWeather, efficiency: float = EFFICIENCY) -> dict[int, float]:
    """Five-minute means for a week, as an array running at ``efficiency`` of the model would produce."""
    start, end = backfill_window(dt_util.utcnow())
    starts = np.arange(int(start.timestamp()), int(end.timestamp()), PERIOD_SECONDS, dtype=np.int64)
    times = (starts + PERIOD_SECONDS // 2).astype("datetime64[s]")
    grid = TimeGrid(times=times, step_seconds=np.full(times.size, float(PERIOD_SECONDS)))
    series = weather_from_hourly(
        hourly_times=weather.times,
        grid=grid,
        site=SITE,
        ghi=weather.ghi,
        temperature=weather.temperature,
        wind_speed_10m=weather.wind_speed_10m,
        direct_horizontal=weather.direct_horizontal,
        diffuse=weather.diffuse,
    )
    predicted = forecast(SYSTEM, grid, series).array("east").dc_power_w
    return {int(s): float(p) * efficiency for s, p in zip(starts, predicted, strict=True)}


def test_a_week_of_history_calibrates_the_array() -> None:
    """Replaying a week lands the correction on the array's real efficiency, with no live samples."""
    weather = synthetic_weather(hours=24 * (BACKFILL_DAYS + 3), history_hours=24 * (BACKFILL_DAYS + 1))
    learner = Learner()

    offered = replay(learner, SYSTEM, weather, {"east": "sensor.east"}, {"sensor.east": recorded_week(weather)}, None)

    state = learner.states["east"]
    assert offered == BACKFILL_DAYS * 24 * 12
    assert state.samples > 500, "a week of daylight is past the ramp-in"

    # Evaluate the learned correction at tomorrow's daylight instants.
    times = np.arange(
        np.datetime64(int(dt_util.utcnow().timestamp()) // 86400 * 86400 + 86400 - 3 * 3600, "s"),
        np.datetime64(int(dt_util.utcnow().timestamp()) // 86400 * 86400 + 86400 + 6 * 3600, "s"),
        np.timedelta64(900, "s"),
    )
    grid = TimeGrid(times=times, step_seconds=np.full(times.size, 900.0))
    series = weather_from_hourly(
        hourly_times=weather.times,
        grid=grid,
        site=SITE,
        ghi=weather.ghi,
        temperature=weather.temperature,
        wind_speed_10m=weather.wind_speed_10m,
        direct_horizontal=weather.direct_horizontal,
        diffuse=weather.diffuse,
    )
    result = forecast(SYSTEM, grid, series)
    samples = samples_from(result, SYSTEM, series.ghi, np.arange(times.size))["east"]
    features = build_features(
        elevation_deg=np.array([s.elevation_deg for s in samples]),
        clearness=np.array([s.clearness for s in samples]),
        day_of_year=np.array([s.day_of_year for s in samples]),
        transmittance=np.array([s.transmittance for s in samples]),
        temperature_c=np.array([s.temperature_c for s in samples]),
    )
    assert float(np.median(state.correction(features))) == pytest.approx(EFFICIENCY, abs=0.03)


def test_history_outside_the_weather_is_skipped() -> None:
    """Periods the weather does not cover are not compared against a made-up sky."""
    weather = synthetic_weather(hours=48, history_hours=24)
    learner = Learner()
    old = int(dt_util.utcnow().timestamp()) - 5 * 86400
    old -= old % PERIOD_SECONDS

    assert replay(learner, SYSTEM, weather, {"east": "sensor.east"}, {"sensor.east": {old: 5000.0}}, None) == 0
    assert learner.states == {}


def test_only_arrays_with_nothing_learned_are_backfilled() -> None:
    """An array that has already learned keeps its state; a new one is caught up."""
    learner = Learner()
    learner.state_for("east").samples = 10
    learner.state_for("west")

    assert arrays_to_backfill(learner, {"east": "sensor.e", "west": "sensor.w", "north": "sensor.n"}) == {
        "west": "sensor.w",
        "north": "sensor.n",
    }


def test_the_window_ends_on_a_period_boundary() -> None:
    """The last replayed period is the one that has finished, not the one in progress."""
    start, end = backfill_window(datetime(2026, 11, 15, 4, 7, 31, tzinfo=UTC))
    assert end == datetime(2026, 11, 15, 4, 5, tzinfo=UTC)
    assert (end - start).days == BACKFILL_DAYS


async def test_no_recorder_means_no_history(hass: HomeAssistant) -> None:
    """Without the recorder the backfill quietly does nothing."""
    now = dt_util.utcnow()
    assert await async_read_history(hass, {"sensor.east"}, now, now) == {}


async def test_recorder_rows_are_read_as_watts_by_period(hass: HomeAssistant) -> None:
    """Statistics rows become a period-start -> mean map, asking the recorder for watts."""
    hass.config.components.add("recorder")
    start = datetime(2026, 11, 15, 1, 0, tzinfo=UTC)
    rows = {
        "sensor.east": [{"start": start.timestamp(), "mean": 1234.5}, {"start": start.timestamp() + 300, "mean": None}]
    }
    recorder = AsyncMock()
    recorder.async_add_executor_job = AsyncMock(return_value=rows)

    with patch("homeassistant.helpers.recorder.get_instance", return_value=recorder):
        history = await async_read_history(hass, {"sensor.east"}, start, start)

    assert history == {"sensor.east": {int(start.timestamp()): 1234.5}}
    args = recorder.async_add_executor_job.call_args.args
    assert args[5] == "5minute"
    assert args[6] == {"power": "W"}


async def test_a_fresh_install_learns_from_history_at_setup(hass: HomeAssistant) -> None:
    """Setup replays recorded output once, so the first forecast is already calibrated."""
    await hass.config.async_set_time_zone("Australia/Sydney")
    weather = synthetic_weather(hours=24 * (BACKFILL_DAYS + 3), history_hours=24 * (BACKFILL_DAYS + 1))
    recorded = recorded_week(weather)
    arrays = tuple(
        {**a, CONF_POWER_SENSOR: f"sensor.{a['name']}_power"} if a["name"] == "east" else a for a in ARRAY_DATA
    )
    entry: MockConfigEntry = build_entry(arrays)
    entry.add_to_hass(hass)

    fetch = AsyncMock(return_value=weather)
    history = AsyncMock(return_value={"sensor.east_power": recorded})
    with (
        patch("custom_components.soular.coordinator.coordinator.fetch", fetch),
        patch("custom_components.soular.coordinator.coordinator.async_read_history", history),
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        coordinator: Any = entry.runtime_data.coordinator
        assert coordinator.learner.states["east"].samples > 500
        assert "west" not in coordinator.learner.states, "arrays without a power sensor have nothing to learn from"

        await coordinator.async_refresh()
        assert history.await_count == 1, "the history is read once, not on every refresh"
        assert fetch.await_args_list[1].kwargs["past_days"] == BACKFILL_DAYS + 1
