"""Catching a fresh weather correction up from recorded history."""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
import numpy as np
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.soular.const import CONF_POWER_SENSOR
from custom_components.soular.coordinator.backfill import (
    BACKFILL_DAYS,
    PERIOD_SECONDS,
    async_read_history,
    backfill_window,
    replay,
)
from custom_components.soular.core.nwp import weather_from_hourly
from custom_components.soular.core.pipeline import SystemSpec, forecast
from custom_components.soular.core.types import ArraySpec, TimeGrid
from custom_components.soular.core.weather_bias import WeatherBias
from custom_components.soular.sources.open_meteo import HourlyWeather
from tests.core.conftest import INVERTER, SITE
from tests.integration.conftest import ARRAY_DATA, build_entry, synthetic_weather

EAST = ArraySpec(name="east", azimuth_deg=84.0, tilt_deg=25.0, dc_capacity_w=7920.0)
SYSTEM = SystemSpec(site=SITE, arrays=(EAST,), inverters={"default": INVERTER})
DELIVERED = 0.8


def recorded_week(weather: HourlyWeather, delivered: float = DELIVERED) -> dict[int, float]:
    """Five-minute means for a week, as a sky delivering ``delivered`` of the forecast would give."""
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
    return {int(s): float(p) * delivered for s, p in zip(starts, predicted, strict=True)}


def utc_day(epoch: int) -> int:
    """Day ordinal in UTC, which is all these tests need of a local day."""
    return datetime.fromtimestamp(epoch, UTC).date().toordinal()


def test_a_week_of_history_moves_the_correction() -> None:
    """A week delivering less than forecast pulls the correction down, held back by the prior."""
    weather = synthetic_weather(hours=24 * (BACKFILL_DAYS + 3), history_hours=24 * (BACKFILL_DAYS + 1))
    bias = WeatherBias()
    recorded = recorded_week(weather)
    counted = replay(bias, SYSTEM, weather, {"east": "sensor.east"}, {"sensor.east": recorded}, None, utc_day)

    assert counted == BACKFILL_DAYS * 24 * 12
    assert bias.day_count >= BACKFILL_DAYS
    tomorrow = utc_day(int(dt_util.utcnow().timestamp())) + 1
    ratios = bias.bin_ratios(tomorrow)
    moved = ratios[ratios != 1.0]
    assert moved.size, "the week said nothing"
    assert np.all((moved < 1.0) & (moved > DELIVERED)), "pulled towards the week, but not all the way"


def test_a_full_battery_teaches_nothing() -> None:
    """Instants with the battery full are curtailed, not cloudy."""
    weather = synthetic_weather(hours=24 * (BACKFILL_DAYS + 3), history_hours=24 * (BACKFILL_DAYS + 1))
    recorded = recorded_week(weather)
    full = dict.fromkeys(recorded, 100.0)
    bias = WeatherBias()
    assert replay(bias, SYSTEM, weather, {"east": "sensor.east"}, {"sensor.east": recorded}, full, utc_day) == 0
    assert bias.day_count == 0


def test_history_outside_the_weather_is_skipped() -> None:
    """Periods the weather does not cover are not compared against a made-up sky."""
    weather = synthetic_weather(hours=48, history_hours=24)
    bias = WeatherBias()
    old = int(dt_util.utcnow().timestamp()) - 5 * 86400
    old -= old % PERIOD_SECONDS

    assert replay(bias, SYSTEM, weather, {"east": "sensor.east"}, {"sensor.east": {old: 5000.0}}, None, utc_day) == 0
    assert bias.day_count == 0


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
    """Setup replays recorded output once, so the first forecast is already corrected."""
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
        assert coordinator.weather_bias.day_count >= BACKFILL_DAYS - 1

        await coordinator.async_refresh()
        assert history.await_count == 1, "the history is read once, not on every refresh"
        assert fetch.await_args_list[1].kwargs["past_days"] == BACKFILL_DAYS + 1


def test_an_array_that_is_off_teaches_nothing() -> None:
    """A second array reading zero in full sun is an outage, not cloud."""
    weather = synthetic_weather(hours=24 * (BACKFILL_DAYS + 3), history_hours=24 * (BACKFILL_DAYS + 1))
    west = ArraySpec(name="west", azimuth_deg=264.0, tilt_deg=25.0, dc_capacity_w=7920.0)
    system = SystemSpec(site=SITE, arrays=(EAST, west), inverters={"default": INVERTER})
    east = recorded_week(weather, delivered=1.0)
    dead = dict.fromkeys(east, 0.0)
    bias = WeatherBias()
    arrays = {"east": "sensor.east", "west": "sensor.west"}
    counted = replay(bias, system, weather, arrays, {"sensor.east": east, "sensor.west": dead}, None, utc_day)
    working = replay(WeatherBias(), system, weather, arrays, {"sensor.east": east, "sensor.west": east}, None, utc_day)
    assert counted < working, "daylight with west dead should not count"
    tomorrow = utc_day(int(dt_util.utcnow().timestamp())) + 1
    assert bias.bin_ratios(tomorrow).min() > 0.95, "nothing was learned as a dark sky"
