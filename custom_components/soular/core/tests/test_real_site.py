"""Checks against the real measured shading maps, when they are available.

These are marked ``backtest`` and skipped when the analysis outputs are not on
this machine, because the grids are not vendored into the repo. They are worth
running because they exercise the actual file layout and the actual measured
values, which no synthetic fixture can stand in for.
"""

from pathlib import Path
import time

import numpy as np
import pytest

from custom_components.soular.core.clearsky import clear_sky
from custom_components.soular.core.geometry import solar_geometry
from custom_components.soular.core.pipeline import SystemSpec, WeatherSeries, build_time_grid, forecast
from custom_components.soular.core.shading import from_horizon, from_npz
from custom_components.soular.core.tests.conftest import ARRAYS, INVERTER, SITE, day_times
from custom_components.soular.core.types import TimeArray, TimeGrid

SOLAR_DATA = Path("/Users/trenthouliston/Code/solar-data/outputs")
GRID_FILE = SOLAR_DATA / "transmittance_grid.npz"

pytestmark = [
    pytest.mark.backtest,
    pytest.mark.skipif(not GRID_FILE.exists(), reason="solar-data analysis outputs not present"),
]

# Sun angles the maps were measured over. In midsummer the sun clears the canopy
# for most of the day; in midwinter it spends the whole day behind it.
SUMMER = "2026-01-15"
WINTER = "2026-06-21"


def load_system(*, shaded: bool) -> SystemSpec:
    """Build the real four-array site, with or without its measured shading."""
    shading = {}
    if shaded:
        raw = GRID_FILE.read_bytes()
        shading = {array.name: from_npz(raw, array.name) for array in ARRAYS}
    return SystemSpec(site=SITE, arrays=ARRAYS, inverters={"default": INVERTER}, shading=shading)


def clear_day(times: TimeArray) -> tuple[TimeGrid, WeatherSeries]:
    """Build a clear-sky weather series on a uniform grid."""
    geometry = solar_geometry(times, SITE)
    clearsky = clear_sky(times, SITE, geometry)
    grid = TimeGrid(times=times, step_seconds=np.full(times.size, 300.0))
    weather = WeatherSeries(
        ghi=clearsky.ghi,
        dni=clearsky.dni,
        dhi=clearsky.dhi,
        temp_air=np.full(times.size, 25.0),
        wind_speed_10m=np.full(times.size, 2.0),
    )
    return grid, weather


def daily_kwh(power_w: np.ndarray, step_seconds: float = 300.0) -> float:
    """Integrate a power series to kWh."""
    return float(np.sum(power_w) * step_seconds / 3.6e6)


def test_every_array_grid_loads() -> None:
    """All four named grids are present and quantise to the documented size."""
    raw = GRID_FILE.read_bytes()
    for array in ARRAYS:
        grid = from_npz(raw, array.name)
        assert grid.azimuth_deg.size == 360, "the duplicated 360-degree column should be dropped"
        assert grid.elevation_deg.size == 91
        # 8-bit storage is the whole point: 33 kB rather than 262 kB per array.
        assert grid.values.nbytes < 40_000


def test_horizon_files_parse() -> None:
    """The hard-horizon files the incumbent integration reads still load."""
    for array in ARRAYS:
        path = SOLAR_DATA / f"horizon-{array.name}.txt"
        if not path.exists():
            continue
        grid = from_horizon(path.read_text())
        assert grid.values.max() > 0, f"{array.name} horizon blocks the entire sky"


def test_shading_costs_more_in_winter_than_summer() -> None:
    """Low winter sun spends far longer behind the canopy.

    This is the signature that the map is applied in the right orientation. A
    mirrored or rotated map would still reduce output, but it would not track
    the season this way.
    """
    shaded, unshaded = load_system(shaded=True), load_system(shaded=False)

    losses: dict[str, float] = {}
    for date in (SUMMER, WINTER):
        grid, weather = clear_day(day_times(date))
        with_maps = daily_kwh(forecast(shaded, grid, weather).ac_power_w)
        without = daily_kwh(forecast(unshaded, grid, weather).ac_power_w)
        losses[date] = 1.0 - with_maps / without

    assert 0.0 < losses[SUMMER] < 0.25, f"summer loss {losses[SUMMER]:.1%} is outside the expected range"
    assert losses[WINTER] > 2 * losses[SUMMER], (
        f"winter loss {losses[WINTER]:.1%} should far exceed summer {losses[SUMMER]:.1%}"
    )


def test_north_is_more_shaded_than_its_geometric_twin() -> None:
    """North and south are identical arrays; only shading separates them.

    Both face 354 degrees at 25 degrees tilt with the same 13 modules, so under
    an unshaded model they produce exactly the same power. The analysis found
    north to be the more obstructed of the two, and that difference is the only
    thing the model can use to tell them apart.
    """
    grid, weather = clear_day(day_times(WINTER))

    unshaded = forecast(load_system(shaded=False), grid, weather)
    assert daily_kwh(unshaded.array("north").ac_power_w) == pytest.approx(
        daily_kwh(unshaded.array("south").ac_power_w), rel=1e-12
    ), "the twins must be indistinguishable without shading"

    shaded = forecast(load_system(shaded=True), grid, weather)
    north = daily_kwh(shaded.array("north").ac_power_w)
    south = daily_kwh(shaded.array("south").ac_power_w)
    assert north < south, f"north {north:.2f} kWh should be more shaded than south {south:.2f} kWh"


@pytest.mark.benchmark
def test_forecast_stays_within_its_time_budget() -> None:
    """A full 48-hour forecast is fast enough to stay off the critical path.

    This runs in an executor in the integration, but a slow model would still
    delay every refresh. If this starts failing, find out why before raising it.
    """
    system = load_system(shaded=True)
    grid = build_time_grid(np.datetime64("2026-01-15T00:00:00", "s"))
    geometry = solar_geometry(grid.times, SITE)
    clearsky = clear_sky(grid.times, SITE, geometry)
    weather = WeatherSeries(
        ghi=clearsky.ghi,
        dni=clearsky.dni,
        dhi=clearsky.dhi,
        temp_air=np.full(grid.times.size, 25.0),
        wind_speed_10m=np.full(grid.times.size, 2.0),
    )

    forecast(system, grid, weather)  # warm caches
    start = time.perf_counter()
    runs = 10
    for _ in range(runs):
        forecast(system, grid, weather)
    elapsed_ms = (time.perf_counter() - start) / runs * 1000.0

    assert elapsed_ms < 50.0, f"48h forecast took {elapsed_ms:.1f} ms"
