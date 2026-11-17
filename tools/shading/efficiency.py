"""Each array's clear-sky efficiency, measured against the integration's own model.

The map is normalised to exactly 1 in open sky, which divides out how well the
array actually performs. That number is not noise to be thrown away: it is a
property of the panels -- soiling, module tolerance, wiring -- and of the model
-- the clear-sky irradiance it assumes. Measured here, once, from the same
verified-clear record the map is built from, it no longer has to be learned
live, where a week of cloud forecast as sun is enough to drag it down.

It is measured against :func:`custom_components.soular.core.pipeline.forecast`
itself, driven by a clear sky and shaded by the map just built, so it means
exactly "what the integration under-predicts on a clear day" and nothing that
only the map builder's simplified model gets wrong.
"""

import numpy as np
import pandas as pd

from custom_components.soular.core.clearsky import clear_sky
from custom_components.soular.core.geometry import solar_geometry
from custom_components.soular.core.pipeline import SystemSpec, WeatherSeries, forecast
from custom_components.soular.core.shading import TransmittanceGrid
from custom_components.soular.core.types import TimeGrid
from tools.site import Site

# The integration's own fallback ambient conditions. The map builder has no
# temperature record, and the seasonal swing this leaves is a few per cent.
AMBIENT_C = 20.0
WIND_MS = 2.0

STEP_SECONDS = 300.0


def measure(site: Site, frames: dict[str, pd.DataFrame], grids: dict[str, TransmittanceGrid]) -> dict[str, float]:
    """Return measured over modelled energy, per array, over its clear samples.

    Energy-weighted rather than a median of ratios, so the sun-high samples that
    carry the energy decide it, and a low-sun mismatch cannot.
    """
    system = SystemSpec(site=site.spec, arrays=site.arrays, inverters={"default": site.inverter}, shading=grids)
    out: dict[str, float] = {}
    for array in site.arrays:
        frame = frames.get(array.name)
        if frame is None or frame.empty:
            continue
        times = pd.DatetimeIndex(frame.index).tz_convert("UTC").tz_localize(None).to_numpy().astype("datetime64[s]")
        geometry = solar_geometry(times, site.spec)
        sky = clear_sky(times, site.spec, geometry)
        weather = WeatherSeries(
            ghi=sky.ghi,
            dni=sky.dni,
            dhi=sky.dhi,
            temp_air=np.full(times.size, AMBIENT_C),
            wind_speed_10m=np.full(times.size, WIND_MS),
        )
        grid = TimeGrid(times=times, step_seconds=np.full(times.size, STEP_SECONDS))
        modelled_w = forecast(system, grid, weather).array(array.name).dc_power_w
        measured_w = frame["p"].to_numpy() * 1000.0
        usable = np.isfinite(measured_w) & (modelled_w > 0.0)
        if not usable.any():
            continue
        out[array.name] = float(np.sum(measured_w[usable]) / np.sum(modelled_w[usable]))
    return out
