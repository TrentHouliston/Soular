"""Replay the forecasting core against measured production and score it.

This is the mechanism by which "as good a forecast as I can get" becomes a
number rather than an assertion. It calls exactly the same
:func:`custom_components.soular.core.pipeline.forecast` the integration calls,
which is the entire reason ``core`` is kept free of Home Assistant.

Three things it is careful about, because each would otherwise flatter the
result:

**Lead time.** At issue time ``t0``, forecasting valid time ``T``, the freshest
model run that existed was issued at or before ``t0``. The previous-runs archive
resolves runs to whole days, so the harness rounds *up*: it uses the run from
``ceil((T - t0) / 24h)`` days before ``T``. That is conservative at short lead --
in production you would usually hold a run a few hours old, not a day -- so
short-lead numbers here are a floor, not an estimate.

**Drivers.** Shading and weather are separable questions, so they are measured
separately. Driving the model with *observed* satellite irradiance isolates the
optics, which is the comparison the shading maps were validated under. Driving it
with forecast irradiance measures what a user actually experiences, where weather
error dominates. Reporting only the second would understate the shading work;
reporting only the first would overstate the forecast.

**Truth.** DC strings are compared against modelled DC and the AC meter against
modelled AC -- the strings integrate to 11-13% more than the inverter's AC output,
so crossing them would manufacture a bias. Daily energy is scored against the
inverter's own daily counters rather than an integral of the five-minute series,
which undershoots them by 1-3%.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import datetime as dt
import json
import math
from pathlib import Path
import sqlite3
import sys
import tomllib
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from custom_components.soular.core.blend import (
    Observation,
    apply_to_weather,
    observation_from_irradiance,
    satellite_observation,
)
from custom_components.soular.core.clearsky import clear_sky, resample_to_grid
from custom_components.soular.core.geometry import solar_geometry
from custom_components.soular.core.irradiance import decompose
from custom_components.soular.core.pipeline import SystemSpec, WeatherSeries, forecast
from custom_components.soular.core.shading import TransmittanceGrid, from_horizon, from_npz
from custom_components.soular.core.types import ArraySpec, FloatArray, InverterSpec, SiteSpec, TimeArray, TimeGrid

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterator

STEP_SECONDS = 300
HORIZON_HOURS = 48

# The satellite archive publishes in arrears. At issue time the freshest
# observation available is roughly this old, and pretending otherwise would let
# the nowcast see cloud that had not been reported yet.
SATELLITE_LATENCY_MINUTES = 30
# How many recent observations to persist from. More than a couple adds little:
# they are ten minutes apart and highly correlated.
SATELLITE_OBSERVATION_COUNT = 3

# The MPPT channels were re-plugged when an EV charger went in. Days either side
# of the changeover are dropped rather than guessed at, since the exact hour is
# not recorded and a wrong attribution would look like a shading error.
CHANNEL_SWAP_DATE = dt.date(2026, 6, 19)
CHANNEL_SWAP_BUFFER_DAYS = 1

CHANNELS_BEFORE = {"PV1_POWER": "east", "PV2_POWER": "west", "PV3_POWER": "north", "PV4_POWER": "south"}
CHANNELS_AFTER = {"PV1_POWER": "west", "PV2_POWER": "east", "PV3_POWER": "north", "PV4_POWER": "south"}

# Battery-full curtailment and export limiting are real lost production that no
# forecast can predict, so those samples are excluded from scoring. The threshold
# is an option because a conclusion that moves when you change it is a conclusion
# about the mask, not about the model.
DEFAULT_CURTAILMENT_SOC_PCT = 97.0

# A hard horizon is lit or not; the incumbent has no notion of partial.
HORIZON_LIT_THRESHOLD = 0.5
# Daily energy is only attributable from the first day of an issue.
FIRST_DAY_HOURS = 24.0

# The incumbent integration's constants, for the emulation baseline.
INCUMBENT_ALBEDO = 0.2
INCUMBENT_ROSS_COEFFICIENT = 0.0342  # "not so well cooled", hardcoded upstream
INCUMBENT_ALPHA_TEMP = -0.004
INCUMBENT_EFFICIENCY = 0.9


@dataclass(frozen=True, slots=True)
class Site:
    """A parsed ``arrays.toml``."""

    spec: SiteSpec
    arrays: tuple[ArraySpec, ...]
    inverter: InverterSpec
    timezone: ZoneInfo


def load_site(path: Path, *, ac_limit_w: float) -> Site:
    """Read site and array geometry from the analysis repo's config."""
    config = tomllib.loads(path.read_text())
    site = config["site"]
    arrays = tuple(
        ArraySpec(
            name=name,
            azimuth_deg=float(entry["azimuth"]),
            tilt_deg=float(entry["tilt"]),
            dc_capacity_w=float(entry["dc_watts"]),
        )
        for name, entry in sorted(config["arrays"].items())
    )
    return Site(
        spec=SiteSpec(
            latitude=float(site["latitude"]),
            longitude=float(site["longitude"]),
            elevation_m=float(site.get("altitude", 0.0)),
        ),
        arrays=arrays,
        inverter=InverterSpec(name="default", ac_limit_w=ac_limit_w),
        timezone=ZoneInfo(str(site.get("timezone", "UTC"))),
    )


@dataclass(frozen=True, slots=True)
class Actuals:
    """Measured production on a uniform five-minute grid, plus masking inputs."""

    times: TimeArray
    dc_by_array: dict[str, FloatArray]
    ac_w: FloatArray
    soc_pct: FloatArray
    valid: np.ndarray
    daily_ac_kwh: dict[dt.date, float]
    curtailment_soc_pct: float = DEFAULT_CURTAILMENT_SOC_PCT

    def curtailed(self) -> np.ndarray:
        """Flag samples where the battery was full enough to curtail production."""
        return self.soc_pct >= self.curtailment_soc_pct


def uniform_grid(start: dt.date, end: dt.date) -> TimeArray:
    """Five-minute UTC instants spanning an inclusive date range."""
    first = np.datetime64(f"{start.isoformat()}T00:00:00", "s")
    last = np.datetime64(f"{(end + dt.timedelta(days=1)).isoformat()}T00:00:00", "s")
    return np.arange(first, last, np.timedelta64(STEP_SECONDS, "s")).astype("datetime64[s]")


def load_actuals(
    db_path: Path,
    times: TimeArray,
    array_names: Sequence[str],
    curtailment_soc_pct: float = DEFAULT_CURTAILMENT_SOC_PCT,
) -> Actuals:
    """Load measured DC strings and AC output onto ``times``.

    The database stores kilowatts; everything downstream is watts.
    """
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    index = {np.datetime64(stamp, "s"): position for position, stamp in enumerate(times)}

    def series(metric: str) -> FloatArray:
        values = np.full(times.size, np.nan)
        rows = connection.execute(
            "SELECT ts_utc, value FROM measurements WHERE metric = ? AND value IS NOT NULL",
            (metric,),
        )
        for stamp, value in rows:
            # Stored as ISO with a +00:00 offset; numpy wants a naive UTC string.
            position = index.get(np.datetime64(stamp[:19], "s"))
            if position is not None:
                values[position] = value * 1000.0
        return values

    channels = {name: series(name) for name in CHANNELS_BEFORE}
    ac_w = series("FROM_SOLAR")
    soc = series("SOC") / 1000.0  # SOC is a percentage, not a power

    swap = np.datetime64(CHANNEL_SWAP_DATE.isoformat(), "s")
    after = times >= swap
    dc_by_array: dict[str, FloatArray] = {name: np.full(times.size, np.nan) for name in array_names}
    for channel, values in channels.items():
        before_name, after_name = CHANNELS_BEFORE[channel], CHANNELS_AFTER[channel]
        if before_name in dc_by_array:
            dc_by_array[before_name] = np.where(after, dc_by_array[before_name], values)
        if after_name in dc_by_array:
            dc_by_array[after_name] = np.where(after, values, dc_by_array[after_name])

    daily: dict[dt.date, float] = {}
    for date_local, value in connection.execute(
        "SELECT date_local, value FROM daily_totals WHERE metric = 'FROM_SOLAR' AND value IS NOT NULL"
    ):
        text = str(date_local)
        daily[dt.date(int(text[:4]), int(text[4:6]), int(text[6:8]))] = float(value)
    connection.close()

    # Exclude the days around the channel swap, and any sample with no reading.
    buffer = np.timedelta64(CHANNEL_SWAP_BUFFER_DAYS * 86400, "s")
    near_swap = (times >= swap - buffer) & (times < swap + buffer)
    complete = np.isfinite(ac_w) & np.all([np.isfinite(v) for v in dc_by_array.values()], axis=0)
    valid = complete & ~near_swap

    return Actuals(
        times=times,
        dc_by_array=dc_by_array,
        ac_w=ac_w,
        soc_pct=soc,
        valid=valid,
        daily_ac_kwh=daily,
        curtailment_soc_pct=curtailment_soc_pct,
    )


class Archive:
    """Read-only access to the fetched forecast archive, with lead enforced."""

    def __init__(self, path: Path) -> None:
        """Load every feed into memory; the whole cache is only a few million rows."""
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

        self.nwp: dict[tuple[int, str], dict[np.datetime64, float]] = {}
        for stamp, lead, variable, value in connection.execute("SELECT valid_utc, lead_day, variable, value FROM nwp"):
            self.nwp.setdefault((lead, variable), {})[np.datetime64(stamp[:19], "s")] = value

        self.satellite: dict[str, dict[np.datetime64, float]] = {}
        for stamp, variable, value in connection.execute("SELECT valid_utc, variable, value FROM satellite"):
            self.satellite.setdefault(variable, {})[np.datetime64(stamp[:19], "s")] = value

        self.leads = sorted({lead for lead, _ in self.nwp})
        connection.close()

    def hourly(self, variable: str, lead: int, times: TimeArray) -> tuple[TimeArray, FloatArray] | None:
        """Return the hourly series for a variable at a lead, over ``times``."""
        table = self.nwp.get((min(lead, max(self.leads)) if self.leads else lead, variable))
        if not table:
            return None
        # Pad an hour either side. A lead block can be shorter than an hour --
        # lead changes on a 24 h boundary that need not align with the grid --
        # and an unpadded range would then hold too few points to interpolate
        # between, or to form a single interval from.
        first = times[0].astype("datetime64[h]").astype("datetime64[s]") - np.timedelta64(3600, "s")
        last = times[-1].astype("datetime64[h]").astype("datetime64[s]") + np.timedelta64(2 * 3600, "s")
        hours = np.arange(first, last, np.timedelta64(3600, "s")).astype("datetime64[s]")
        values = np.array([table.get(hour, np.nan) for hour in hours])
        if not np.isfinite(values).any():
            return None
        return hours, values

    def observed(self, variable: str, times: TimeArray) -> FloatArray | None:
        """Return satellite observations sampled onto ``times``, or None."""
        table = self.satellite.get(variable)
        if not table:
            return None
        stamps = np.array(sorted(table), dtype="datetime64[s]")
        values = np.array([table[stamp] for stamp in stamps], dtype=np.float64)
        seconds = stamps.astype(np.int64).astype(np.float64)
        wanted = times.astype("datetime64[s]").astype(np.int64).astype(np.float64)
        return np.asarray(np.interp(wanted, seconds, values, left=np.nan, right=np.nan), dtype=np.float64)


def lead_days(issue: np.datetime64, valid: TimeArray) -> FloatArray:
    """Whole-day lead of each valid time relative to an issue time.

    Rounded up, so the harness never uses a model run that had not been issued.
    A run labelled N days before valid time T was issued around ``T - N days``;
    requiring ``T - N days <= issue`` gives ``N >= (T - issue) / 24 h``.
    """
    delta = (valid - issue).astype("timedelta64[s]").astype(np.float64)
    return np.ceil(np.clip(delta, 0.0, None) / 86400.0)


@dataclass(frozen=True, slots=True)
class Variant:
    """One configuration of the model, scored end to end."""

    name: str
    description: str
    driver: str  # "forecast" or "observed"
    shading: bool = True
    transposition: str = "perez"
    incumbent: bool = False
    # Use the hard-horizon files instead of the graded transmittance grids, so
    # the value of grading can be separated from the value of shading at all.
    hard_horizon: bool = False
    # Blend recent satellite observations into the forecast by persistence.
    nowcast: bool = False


VARIANTS: tuple[Variant, ...] = (
    Variant("soular", "Perez + graded shading, forecast irradiance", "forecast"),
    Variant("no-shading", "same, with shading disabled", "forecast", shading=False),
    Variant("no-perez", "isotropic transposition instead of Perez", "forecast", transposition="isotropic"),
    Variant("incumbent", "open_meteo_solar_forecast emulation", "forecast", incumbent=True),
    Variant("soular-observed", "Perez + graded shading, satellite irradiance", "observed"),
    Variant("no-shading-observed", "shading disabled, satellite irradiance", "observed", shading=False),
    Variant("no-perez-observed", "isotropic sky, satellite irradiance", "observed", transposition="isotropic"),
    Variant("hard-horizon", "Soular optics with a hard horizon", "forecast", hard_horizon=True),
    Variant("hard-horizon-observed", "hard horizon, satellite irradiance", "observed", hard_horizon=True),
    Variant("nowcast", "forecast irradiance plus a satellite nowcast", "forecast", nowcast=True),
)


def build_system(
    site: Site,
    shading: dict[str, TransmittanceGrid],
    horizons: dict[str, TransmittanceGrid],
    variant: Variant,
) -> SystemSpec:
    """Assemble the system spec a variant runs under."""
    spec = SiteSpec(
        latitude=site.spec.latitude,
        longitude=site.spec.longitude,
        elevation_m=site.spec.elevation_m,
        albedo=INCUMBENT_ALBEDO if variant.incumbent else site.spec.albedo,
        transposition_model="isotropic" if variant.transposition == "isotropic" else "perez",  # type: ignore[arg-type]
    )
    if not variant.shading:
        applied: dict[str, TransmittanceGrid] = {}
    elif variant.hard_horizon:
        applied = horizons
    else:
        applied = shading
    return SystemSpec(site=spec, arrays=site.arrays, inverters={"default": site.inverter}, shading=applied)


def incumbent_forecast(
    site: Site,
    horizons: dict[str, TransmittanceGrid],
    grid: TimeGrid,
    weather: WeatherSeries,
) -> FloatArray:
    """Emulate ``open_meteo_solar_forecast``, for a like-for-like comparison.

    Isotropic sky with a fixed albedo of 0.2, no incidence-angle modifier, a Ross
    thermal coefficient with no wind term, and a hard horizon that substitutes
    *horizontal* diffuse when the sun is blocked. Clipping is applied to hourly
    means, as upstream does.
    """
    geometry = solar_geometry(grid.times, site.spec)
    dni = weather.dni if weather.dni is not None else decompose(grid.times, weather.ghi, geometry)[0]
    dhi = weather.dhi if weather.dhi is not None else decompose(grid.times, weather.ghi, geometry)[1]

    zenith = np.radians(geometry.apparent_zenith)
    total_dc = np.zeros(grid.times.size, dtype=np.float64)

    for array in site.arrays:
        tilt = np.radians(array.tilt_deg)
        cos_aoi = np.clip(
            np.cos(zenith) * np.cos(tilt)
            + np.sin(zenith) * np.sin(tilt) * np.cos(np.radians(geometry.azimuth - array.azimuth_deg)),
            0.0,
            None,
        )
        sky_view = (1.0 + np.cos(tilt)) / 2.0
        gti = dni * cos_aoi + dhi * sky_view + weather.ghi * INCUMBENT_ALBEDO * (1.0 - sky_view)

        horizon = horizons.get(array.name)
        if horizon is not None:
            # Upstream blocks all-or-nothing and falls back to horizontal diffuse,
            # with no sky-view scaling and no ground term.
            lit = horizon.lookup(geometry.azimuth, geometry.apparent_elevation) > HORIZON_LIT_THRESHOLD
            gti = np.where(lit, gti, dhi)

        cell = weather.temp_air + gti * INCUMBENT_ROSS_COEFFICIENT
        power = array.dc_capacity_w * (gti / 1000.0) * (1.0 + INCUMBENT_ALPHA_TEMP * (cell - 25.0))
        total_dc += np.clip(power * INCUMBENT_EFFICIENCY, 0.0, None)

    # Clip hourly means rather than instantaneous power, as upstream does.
    per_hour = 3600 // STEP_SECONDS
    usable = (total_dc.size // per_hour) * per_hour
    hourly = total_dc[:usable].reshape(-1, per_hour).mean(axis=1)
    clipped = np.clip(hourly, 0.0, site.inverter.ac_limit_w)
    expanded = np.repeat(clipped, per_hour)
    return np.concatenate([expanded, np.clip(total_dc[usable:], 0.0, site.inverter.ac_limit_w)])


def weather_for(
    archive: Archive,
    site: Site,
    grid: TimeGrid,
    issue: np.datetime64,
    driver: str,
    *,
    nowcast: bool = False,
) -> WeatherSeries | None:
    """Assemble the weather a variant sees, respecting lead time."""
    geometry = solar_geometry(grid.times, site.spec)
    clearsky = clear_sky(grid.times, site.spec, geometry)

    leads = lead_days(issue, grid.times)

    def resample(variable: str) -> FloatArray:
        """Bring one hourly interval-mean radiation variable onto the grid."""
        values = np.full(grid.times.size, np.nan)
        # One resample per distinct lead, so each block comes from a single run.
        for lead in np.unique(leads).astype(int):
            block = leads == lead
            hourly = archive.hourly(variable, int(lead), grid.times[block])
            if hourly is None:
                continue
            hours, series = hourly
            # Open-Meteo labels an interval mean with the end of its interval.
            values[block] = resample_to_grid(
                hours[:-1], hours[1:], series[1:], grid.times[block], clearsky.ghi[block], site.spec
            )
        return values

    ghi = resample("shortwave_radiation")
    # The NWP forecasts its own beam/diffuse split, which is model-informed and
    # better than re-deriving one from GHI with a correlation. Not using it would
    # repeat the incumbent's mistake of downloading data and discarding it.
    direct = resample("direct_radiation")
    diffuse = resample("diffuse_radiation")

    if driver == "observed":
        observed = archive.observed("shortwave_radiation", grid.times)
        if observed is not None:
            ghi = np.where(np.isfinite(observed), observed, ghi)
            # The satellite product carries shortwave radiation only, so its
            # split has to be derived. Dropping the NWP's split here keeps beam,
            # diffuse and total mutually consistent with the driving GHI.
            direct = np.full(grid.times.size, np.nan)
            diffuse = np.full(grid.times.size, np.nan)

    if not np.isfinite(ghi).any():
        return None
    ghi = np.nan_to_num(ghi, nan=0.0)

    temp = _ambient(archive, grid, leads, "temperature_2m", default=20.0)
    wind = _ambient(archive, grid, leads, "wind_speed_10m", default=2.0)

    if np.isfinite(direct).any() and np.isfinite(diffuse).any():
        # direct_radiation is on the horizontal; the model wants normal incidence.
        cos_zenith = np.clip(np.cos(np.radians(geometry.apparent_zenith)), 0.05, None)
        dni = np.nan_to_num(direct, nan=0.0) / cos_zenith
        dhi = np.nan_to_num(diffuse, nan=0.0)
    else:
        dni, dhi = decompose(grid.times, ghi, geometry)

    series = WeatherSeries(ghi=ghi, dni=dni, dhi=dhi, temp_air=temp, wind_speed_10m=wind)
    if nowcast:
        observations = _satellite_observations(archive, site, issue)
        if observations:
            series, _ = apply_to_weather(series, grid.times, clearsky.ghi, geometry, observations)
    return series


def _satellite_observations(archive: Archive, site: Site, issue: np.datetime64) -> list[Observation]:
    """Satellite observations that had actually been published at the issue time.

    Only observations at or before ``issue - latency`` are eligible. The archive
    holds the whole record, so without that cutoff the nowcast would be reading
    cloud that had not been published yet -- the single easiest way to
    manufacture skill that does not exist.
    """
    cutoff = issue - np.timedelta64(SATELLITE_LATENCY_MINUTES * 60, "s")
    table = archive.satellite.get("shortwave_radiation")
    if not table:
        return []

    stamps = np.array(sorted(stamp for stamp in table if stamp <= cutoff), dtype="datetime64[s]")
    if stamps.size == 0:
        return []
    recent = stamps[-SATELLITE_OBSERVATION_COUNT:]

    # Clear sky at the observation instants, so the ratio is well posed there
    # rather than being borrowed from the forecast grid.
    geometry = solar_geometry(recent, site.spec)
    observed_clearsky = clear_sky(recent, site.spec, geometry).ghi

    observations: list[Observation] = []
    for stamp, clear in zip(recent, observed_clearsky, strict=True):
        k = observation_from_irradiance(stamp, table[stamp], float(clear))
        if k is not None:
            observations.append(satellite_observation(stamp, k))
    return observations


def _ambient(archive: Archive, grid: TimeGrid, leads: FloatArray, variable: str, default: float) -> FloatArray:
    """Interpolate an hourly ambient variable onto the grid, at honest lead."""
    values = np.full(grid.times.size, np.nan)
    for lead in np.unique(leads).astype(int):
        block = leads == lead
        hourly = archive.hourly(variable, int(lead), grid.times[block])
        if hourly is None:
            continue
        hours, series = hourly
        finite = np.isfinite(series)
        if not finite.any():
            continue
        values[block] = np.interp(
            grid.times[block].astype(np.int64).astype(np.float64),
            hours[finite].astype(np.int64).astype(np.float64),
            series[finite],
        )
    return np.nan_to_num(values, nan=default)


@dataclass
class Accumulator:
    """Sufficient statistics for scoring a series under any scalar gain.

    Storing cross-moments rather than errors means the gain can be chosen after
    the replay, which is what makes the ablations fair. Comparing two
    transposition models without re-fitting a gain does not compare the models:
    it compares their biases. Perez delivers more plane-of-array irradiance than
    isotropic, so against an efficiency that was never calibrated, the *worse*
    model wins whenever the forecast already over-predicts. The site analysis hit
    this exact trap and had to withdraw a set of numbers over it.
    """

    n: int = 0
    sum_p: float = 0.0
    sum_a: float = 0.0
    sum_pp: float = 0.0
    sum_pa: float = 0.0
    sum_aa: float = 0.0

    def add(self, predicted: FloatArray, actual: FloatArray) -> None:
        """Accumulate cross-moments over the finite samples of a pair of series."""
        usable = np.isfinite(predicted) & np.isfinite(actual)
        if not usable.any():
            return
        p, a = predicted[usable], actual[usable]
        self.n += int(usable.sum())
        self.sum_p += float(np.sum(p))
        self.sum_a += float(np.sum(a))
        self.sum_pp += float(np.sum(p * p))
        self.sum_pa += float(np.sum(p * a))
        self.sum_aa += float(np.sum(a * a))

    @property
    def gain(self) -> float:
        """Least-squares scalar that best maps prediction onto actual."""
        return self.sum_pa / self.sum_pp if self.sum_pp > 0.0 else 1.0

    def rmse(self, gain: float = 1.0) -> float:
        """Root mean squared error, watts, after applying ``gain``."""
        if not self.n:
            return math.nan
        total = gain * gain * self.sum_pp - 2.0 * gain * self.sum_pa + self.sum_aa
        return math.sqrt(max(total, 0.0) / self.n)

    def mbe(self, gain: float = 1.0) -> float:
        """Mean bias, watts. Positive means the model over-predicts."""
        return (gain * self.sum_p - self.sum_a) / self.n if self.n else math.nan


# Every third local day calibrates the scalar gain; the rest are scored. One
# free parameter would barely overfit in-sample, but splitting costs nothing and
# removes the question entirely.
FIT_DAY_MODULUS = 3


def is_fit_day(local_day: int) -> bool:
    """Report whether a yyyymmdd-encoded local day belongs to the fitting set."""
    return (local_day % 100 + local_day // 100 % 100) % FIT_DAY_MODULUS == 0


# Night samples are perfectly predicted by every variant, so including them in a
# lead-time bucket dilutes any real difference toward nothing. A nowcast that
# changes daylight irradiance by 10% can look like it changed nothing at all.
DAYLIGHT_POWER_W = 1000.0

# Fine near the issue, coarse beyond. A nowcast built on persistence only acts
# for a couple of hours, so a 0-6h bucket buries its effect under four hours of
# samples it never touched -- which is exactly how a real 27% improvement in
# short-lead irradiance first measured as 0.1%.
LEAD_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0-1h", 0.0, 1.0),
    ("1-2h", 1.0, 2.0),
    ("2-6h", 2.0, 6.0),
    ("6-24h", 6.0, 24.0),
    ("24-48h", 24.0, 48.0),
)


@dataclass
class VariantScore:
    """Everything measured for one variant."""

    # Gains are fitted on one set of days and every number is reported on the
    # other, so a variant cannot be rewarded for a gain tuned to the same data.
    fit_site: Accumulator
    fit_per_array: dict[str, Accumulator]
    site: Accumulator
    per_array: dict[str, Accumulator]
    per_lead: dict[str, Accumulator]
    # The same buckets, restricted to samples where the site was actually
    # generating. This is where a near-term intervention has to show up.
    per_lead_daylight: dict[str, Accumulator]
    # Keyed on (issue time, local day as yyyymmdd) so a day-ahead forecast can be
    # selected afterwards rather than whichever issue happened to write last.
    daily_predicted: dict[tuple[np.datetime64, int], float]

    @classmethod
    def empty(cls, array_names: Sequence[str]) -> VariantScore:
        """Create a zeroed score."""
        return cls(
            fit_site=Accumulator(),
            fit_per_array={name: Accumulator() for name in array_names},
            site=Accumulator(),
            per_array={name: Accumulator() for name in array_names},
            per_lead={label: Accumulator() for label, _, _ in LEAD_BUCKETS},
            per_lead_daylight={label: Accumulator() for label, _, _ in LEAD_BUCKETS},
            daily_predicted={},
        )


def issue_times(start: dt.date, end: dt.date, every_hours: int) -> Iterator[np.datetime64]:
    """Yield forecast issue times across the replay period."""
    cursor = np.datetime64(f"{start.isoformat()}T00:00:00", "s")
    last = np.datetime64(f"{end.isoformat()}T00:00:00", "s")
    step = np.timedelta64(every_hours * 3600, "s")
    while cursor <= last:
        yield cursor
        cursor = cursor + step


def run(
    site: Site,
    shading: dict[str, TransmittanceGrid],
    horizons: dict[str, TransmittanceGrid],
    archive: Archive,
    actuals: Actuals,
    variants: Sequence[Variant],
    start: dt.date,
    end: dt.date,
    every_hours: int,
    timezone: ZoneInfo,
) -> dict[str, VariantScore]:
    """Replay every issue time through every variant and accumulate scores."""
    names = [array.name for array in site.arrays]
    scores = {variant.name: VariantScore.empty(names) for variant in variants}
    position = {stamp: i for i, stamp in enumerate(actuals.times)}

    issues = list(issue_times(start, end, every_hours))
    for number, issue in enumerate(issues, start=1):
        times = np.arange(
            issue, issue + np.timedelta64(HORIZON_HOURS * 3600, "s"), np.timedelta64(STEP_SECONDS, "s")
        ).astype("datetime64[s]")
        rows = np.array([position.get(stamp, -1) for stamp in times])
        inside = rows >= 0
        if inside.sum() < times.size // 2:
            continue

        grid = TimeGrid(times=times, step_seconds=np.full(times.size, float(STEP_SECONDS)))
        usable = inside & actuals.valid[np.clip(rows, 0, None)] & ~actuals.curtailed()[np.clip(rows, 0, None)]
        if not usable.any():
            continue

        lead_hours = (times - issue).astype("timedelta64[s]").astype(np.float64) / 3600.0
        local_days = to_local_dates(times, timezone)

        for variant in variants:
            weather = weather_for(archive, site, grid, issue, variant.driver, nowcast=variant.nowcast)
            if weather is None:
                continue

            score = scores[variant.name]
            fitting = is_fit_day(local_days[0])
            if variant.incumbent:
                predicted_site = incumbent_forecast(site, horizons, grid, weather)
                per_array_dc: dict[str, FloatArray] = {}
            else:
                result = forecast(build_system(site, shading, horizons, variant), grid, weather)
                predicted_site = result.ac_power_w
                per_array_dc = {entry.name: entry.dc_power_w for entry in result.arrays}

            actual_site = np.where(usable, actuals.ac_w[np.clip(rows, 0, None)], np.nan)

            if fitting:
                score.fit_site.add(predicted_site, actual_site)
                for name, predicted_dc in per_array_dc.items():
                    actual_dc = np.where(usable, actuals.dc_by_array[name][np.clip(rows, 0, None)], np.nan)
                    score.fit_per_array[name].add(predicted_dc, actual_dc)
            else:
                score.site.add(predicted_site, actual_site)
                generating = np.nan_to_num(actual_site, nan=0.0) > DAYLIGHT_POWER_W
                for label, low, high in LEAD_BUCKETS:
                    window = usable & (lead_hours >= low) & (lead_hours < high)
                    if window.any():
                        score.per_lead[label].add(predicted_site[window], actual_site[window])
                    lit = window & generating
                    if lit.any():
                        score.per_lead_daylight[label].add(predicted_site[lit], actual_site[lit])
                for name, predicted_dc in per_array_dc.items():
                    actual_dc = np.where(usable, actuals.dc_by_array[name][np.clip(rows, 0, None)], np.nan)
                    score.per_array[name].add(predicted_dc, actual_dc)

            # Bin predicted energy into local calendar days. The inverter's own
            # daily counters are local-day totals, and this site is 10-11 hours
            # east of UTC, so binning by UTC day would compare two different days.
            if not fitting:
                # Only whole local days. The window starts and ends mid-day in
                # local time, so its first and last days are partial -- scoring
                # those as whole days reads as a massive under-prediction. Since
                # the window is contiguous, every other day is fully covered.
                for day in np.unique(local_days[1:-1]):
                    if day in (local_days[0], local_days[-1]):
                        continue
                    window = local_days == day
                    energy = float(np.sum(predicted_site[window]) * STEP_SECONDS / 3.6e6)
                    score.daily_predicted[issue, int(day)] = energy

        if number % 50 == 0:
            print(f"  {number}/{len(issues)} issue times", file=sys.stderr)

    return scores


def to_local_dates(times: TimeArray, timezone: ZoneInfo) -> NDArray[np.int64]:
    """Map UTC instants to local calendar days, encoded as ``yyyymmdd``.

    Vectorised because a full replay converts the better part of a million
    timestamps, and DST-aware because this site shifts by an hour twice a year --
    a fixed offset would misfile a day's energy either side of each transition,
    and the inverter's own counters are local-day totals.
    """
    local = pd.DatetimeIndex(pd.to_datetime(times, utc=True)).tz_convert(timezone)
    return np.asarray(local.strftime("%Y%m%d"), dtype=np.int64)


def daily_errors(score: VariantScore, actuals: Actuals, timezone: ZoneInfo, gain: float) -> list[float]:
    """Day-ahead daily energy error, as a mean absolute percentage.

    "Day-ahead" is pinned down rather than left to whichever issue wrote last:
    for each local day, the forecast used is the one issued most recently at or
    before that day's local midnight. Anything else mixes forecast horizons and
    reports a number nobody could have acted on.
    """
    by_day: dict[int, list[tuple[np.datetime64, float]]] = {}
    for (issue, day), energy in score.daily_predicted.items():
        by_day.setdefault(int(day), []).append((issue, energy))

    errors: list[float] = []
    for stamp, entries in by_day.items():
        day = dt.date(stamp // 10000, (stamp // 100) % 100, stamp % 100)
        actual = actuals.daily_ac_kwh.get(day, 0.0)
        if actual <= 1.0:
            continue
        midnight = np.datetime64(
            dt.datetime.combine(day, dt.time(0, 0), timezone).astimezone(dt.UTC).replace(tzinfo=None).isoformat(), "s"
        )
        # The issue must precede the day, and must be recent enough that its
        # 48-hour horizon still covers all of it.
        eligible = [
            (issue, energy)
            for issue, energy in entries
            if issue <= midnight and (midnight - issue) <= np.timedelta64(24 * 3600, "s")
        ]
        if not eligible:
            continue
        _, predicted = max(eligible, key=lambda pair: pair[0])
        errors.append(abs(predicted * gain - actual) / actual)

    return errors


def daily_mape(score: VariantScore, actuals: Actuals, timezone: ZoneInfo, gain: float) -> tuple[float, float]:
    """Mean and median absolute percentage error on day-ahead daily energy.

    Both, because the mean is not robust here. A heavily overcast winter day can
    produce a couple of kilowatt-hours, and a two-kilowatt-hour miss on it is a
    100% error that swamps a whole month of good days. The median says what a
    typical day looks like; the mean says what the tail costs.
    """
    errors = daily_errors(score, actuals, timezone, gain)
    if not errors:
        return math.nan, math.nan
    return float(np.mean(errors)) * 100.0, float(np.median(errors)) * 100.0


def report(
    scores: dict[str, VariantScore],
    actuals: Actuals,
    site: Site,
    variants: Sequence[Variant],
    timezone: ZoneInfo,
) -> str:
    """Render the results as markdown, with every variant fairly re-scaled."""
    names = [array.name for array in site.arrays]
    capacity = sum(array.dc_capacity_w for array in site.arrays)
    caps = {array.name: array.dc_capacity_w for array in site.arrays}

    # One scalar per variant, least-squares fitted on the fitting days only.
    gains = {name: score.fit_site.gain for name, score in scores.items()}
    array_gains = {name: {array: score.fit_per_array[array].gain for array in names} for name, score in scores.items()}

    lines: list[str] = [
        "# Soular backtest",
        "",
        "Every variant carries its own scalar efficiency, fitted by least squares on",
        f"one local day in {FIT_DAY_MODULUS}, with all numbers below reported on the others. Without",
        "that, an ablation measures which variant happened to have less bias rather",
        "than which one is more skilful.",
        "",
        "## Site AC power, five-minute samples (holdout days)",
        "",
        "| variant | gain | RMSE (W) | nRMSE | bias (W) | energy MAPE | energy median APE | n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in variants:
        score = scores[variant.name]
        if not score.site.n:
            continue
        gain = gains[variant.name]
        mean_ape, median_ape = daily_mape(score, actuals, timezone, gain)
        lines.append(
            f"| {variant.name} | {gain:.3f} | {score.site.rmse(gain):,.0f} "
            f"| {score.site.rmse(gain) / capacity:.1%} | {score.site.mbe(gain):+,.0f} "
            f"| {mean_ape:.1f}% | {median_ape:.1f}% | {score.site.n:,} |"
        )

    lines += [
        "",
        "## Site AC power by lead time (RMSE, W)",
        "",
        "| variant | " + " | ".join(label for label, _, _ in LEAD_BUCKETS) + " |",
        "|---|" + "---:|" * len(LEAD_BUCKETS),
    ]
    for variant in variants:
        score = scores[variant.name]
        if not score.site.n:
            continue
        gain = gains[variant.name]
        cells = " | ".join(
            f"{score.per_lead[label].rmse(gain):,.0f}" if score.per_lead[label].n else "-"
            for label, _, _ in LEAD_BUCKETS
        )
        lines.append(f"| {variant.name} | {cells} |")

    lines += [
        "",
        f"## Site AC power by lead time, generating hours only (RMSE, W, above {DAYLIGHT_POWER_W:,.0f} W)",
        "",
        "| variant | " + " | ".join(label for label, _, _ in LEAD_BUCKETS) + " | n (0-6h) |",
        "|---|" + "---:|" * (len(LEAD_BUCKETS) + 1),
    ]
    for variant in variants:
        score = scores[variant.name]
        if not score.site.n:
            continue
        gain = gains[variant.name]
        cells = " | ".join(
            f"{score.per_lead_daylight[label].rmse(gain):,.0f}" if score.per_lead_daylight[label].n else "-"
            for label, _, _ in LEAD_BUCKETS
        )
        first = score.per_lead_daylight[LEAD_BUCKETS[0][0]].n
        lines.append(f"| {variant.name} | {cells} | {first:,} |")

    lines += [
        "",
        "## Per-array DC power (RMSE as a fraction of array capacity)",
        "",
        "| variant | " + " | ".join(names) + " |",
        "|---|" + "---:|" * len(names),
    ]
    for variant in variants:
        score = scores[variant.name]
        if not any(score.per_array[name].n for name in names):
            continue
        cells = " | ".join(
            f"{score.per_array[name].rmse(array_gains[variant.name][name]) / caps[name]:.1%}"
            if score.per_array[name].n
            else "-"
            for name in names
        )
        lines.append(f"| {variant.name} | {cells} |")

    lines += ["", "## Ablations", ""]
    for driver, base, comparison, what in (
        ("forecast", "no-shading", "soular", "graded shading"),
        ("forecast", "no-perez", "soular", "Perez transposition"),
        ("forecast", "incumbent", "soular", "everything, vs the incumbent"),
        ("observed", "no-shading-observed", "soular-observed", "graded shading, weather removed"),
        ("observed", "no-perez-observed", "soular-observed", "Perez transposition, weather removed"),
        ("forecast", "no-shading", "hard-horizon", "a hard horizon, vs no shading at all"),
        ("forecast", "hard-horizon", "soular", "grading the shading, vs a hard horizon"),
        ("observed", "hard-horizon-observed", "soular-observed", "grading the shading, weather removed"),
        ("forecast", "soular", "nowcast", "a satellite nowcast, on top of everything"),
    ):
        if base not in scores or comparison not in scores:
            continue
        before, after = scores[base], scores[comparison]
        if not (before.site.n and after.site.n):
            continue
        before_rmse = before.site.rmse(gains[base])
        after_rmse = after.site.rmse(gains[comparison])
        delta = 1.0 - after_rmse / before_rmse
        lines.append(f"- **{what}** ({driver} driver): RMSE {before_rmse:,.0f} W -> {after_rmse:,.0f} W, {delta:+.1%}")

    if "nowcast" in scores and "soular" in scores:
        lines += ["", "Nowcast effect by lead time, generating hours only:", ""]
        for label, _, _ in LEAD_BUCKETS:
            before = scores["soular"].per_lead_daylight[label]
            after = scores["nowcast"].per_lead_daylight[label]
            if before.n and after.n:
                before_rmse = before.rmse(gains["soular"])
                after_rmse = after.rmse(gains["nowcast"])
                lines.append(
                    f"- {label}: {before_rmse:,.0f} W -> {after_rmse:,.0f} W, "
                    f"{1.0 - after_rmse / before_rmse:+.1%} (n={after.n:,})"
                )

    if "no-shading-observed" in scores and "soular-observed" in scores:
        lines += ["", "Per-array shading benefit, weather removed:", ""]
        for name in names:
            before = scores["no-shading-observed"].per_array[name]
            after = scores["soular-observed"].per_array[name]
            if before.n and after.n:
                before_rmse = before.rmse(array_gains["no-shading-observed"][name])
                after_rmse = after.rmse(array_gains["soular-observed"][name])
                lines.append(f"- {name}: {1.0 - after_rmse / before_rmse:+.1%} RMSE")

    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the backtest and write a report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, required=True, help="the Sigen extract, solar.db")
    parser.add_argument("--arrays", type=Path, required=True, help="arrays.toml")
    parser.add_argument("--shading", type=Path, help="transmittance_grid.npz")
    parser.add_argument("--horizons", type=Path, help="directory holding horizon-<array>.txt")
    parser.add_argument("--cache", type=Path, default=Path("backtest_cache.db"))
    parser.add_argument("--start", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--end", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--issue-every", type=int, default=6, help="hours between forecast issue times")
    # No AC clipping is visible anywhere in this site's record -- output reaches
    # 25.8 kW and the distribution decays smoothly with no pileup -- so the
    # default has to sit above anything observed. A limit set too low silently
    # truncates predictions, and truncates the variants that predict most the
    # hardest, which quietly rigs an ablation.
    parser.add_argument("--ac-limit-w", type=float, default=100000.0)
    parser.add_argument(
        "--curtailment-soc",
        type=float,
        default=DEFAULT_CURTAILMENT_SOC_PCT,
        help="exclude samples at or above this battery state of charge",
    )
    parser.add_argument("--variants", help="comma-separated subset of variant names to run")
    parser.add_argument("--out", type=Path, default=Path("backtest_out/report.md"))
    args = parser.parse_args(argv)

    site = load_site(args.arrays, ac_limit_w=args.ac_limit_w)
    names = [array.name for array in site.arrays]
    print(f"Site: {site.spec.latitude:.5f}, {site.spec.longitude:.5f}  arrays: {', '.join(names)}")

    shading: dict[str, TransmittanceGrid] = {}
    if args.shading and args.shading.exists():
        raw = args.shading.read_bytes()
        shading = {name: from_npz(raw, name) for name in names}
        print(f"Shading: graded grids for {', '.join(shading)}")

    horizons: dict[str, TransmittanceGrid] = {}
    if args.horizons:
        for name in names:
            path = args.horizons / f"horizon-{name}.txt"
            if path.exists():
                horizons[name] = from_horizon(path.read_text())
        print(f"Horizons (for the incumbent baseline): {', '.join(horizons) or 'none'}")

    times = uniform_grid(args.start, args.end)
    actuals = load_actuals(args.db, times, names, curtailment_soc_pct=args.curtailment_soc)
    excluded = int((actuals.valid & actuals.curtailed()).sum())
    print(
        f"Actuals: {int(actuals.valid.sum()):,} usable of {times.size:,} five-minute samples; "
        f"{excluded:,} further excluded as curtailed (SOC >= {args.curtailment_soc:.0f}%)"
    )

    archive = Archive(args.cache)
    print(f"Archive: NWP leads {archive.leads}, satellite {'present' if archive.satellite else 'absent'}")

    variants = [v for v in VARIANTS if v.driver == "forecast" or archive.satellite]
    if args.variants:
        wanted = {name.strip() for name in args.variants.split(",") if name.strip()}
        unknown = wanted - {v.name for v in VARIANTS}
        if unknown:
            parser.error(f"unknown variants: {sorted(unknown)}")
        variants = [v for v in variants if v.name in wanted]
    print(f"Replaying {len(variants)} variants from {args.start} to {args.end}")
    scores = run(
        site, shading, horizons, archive, actuals, variants, args.start, args.end, args.issue_every, site.timezone
    )

    text = report(scores, actuals, site, variants, site.timezone)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                variant.name: {
                    "gain": scores[variant.name].fit_site.gain,
                    "site": {
                        "rmse": scores[variant.name].site.rmse(scores[variant.name].fit_site.gain),
                        "bias": scores[variant.name].site.mbe(scores[variant.name].fit_site.gain),
                        "n": scores[variant.name].site.n,
                    },
                    "daily_energy_ape": dict(
                        zip(
                            ("mean", "median"),
                            daily_mape(
                                scores[variant.name], actuals, site.timezone, scores[variant.name].fit_site.gain
                            ),
                            strict=True,
                        )
                    ),
                    "per_array": {
                        name: scores[variant.name].per_array[name].rmse(scores[variant.name].fit_per_array[name].gain)
                        for name in names
                        if scores[variant.name].per_array[name].n
                    },
                }
                for variant in variants
            },
            indent=2,
        )
    )
    print(text)
    print(f"Written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
