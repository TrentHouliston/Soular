"""Turn measured power into per-array shading samples.

A sample is one array at one five-minute instant: where the sun was, what the
array would have produced unshaded split into beam and diffuse, and what it
actually produced. Only samples that can say something about shading survive:
verified-clear sky, sun up, nothing curtailing the system.
"""

from dataclasses import dataclass
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from custom_components.soular.core.clearsky import clear_sky
from custom_components.soular.core.geometry import solar_geometry
from custom_components.soular.core.irradiance import plane_of_array
from custom_components.soular.core.types import FloatArray, TimeArray
from tools.site import Site

STEP = pd.Timedelta(minutes=5)

# Below this the clear-sky model is unreliable and the shadow is mostly the
# ground itself.
MIN_ELEVATION_DEG = 3.0

# Clear-sky selection from the satellite's observed clear-sky index. The small
# upper allowance admits cloud enhancement and the clear-sky model's mild
# conservatism without letting real cloud in.
CLEAR_K_LOW = 0.95
CLEAR_K_HIGH = 1.15
MIN_CLEARSKY_GHI = 60.0

# Curtailment. A full battery has nowhere to put the power; a site total pinned
# flat for fifteen minutes while the modelled sky is still moving is being held,
# not shaded. Either looks exactly like a shadow covering the whole sky.
SOC_FULL_PCT = 97.0
FLAT_TOP_WINDOW = 3
FLAT_TOP_MAX_STEP_KW = 0.05
FLAT_TOP_MIN_MODEL_STEP_KW = 0.25
FLAT_TOP_MIN_POWER_KW = 1.0

# Whole days that look like outages or comms gaps, judged against modelled
# clear-sky energy so a genuinely dark winter day survives.
MIN_DAILY_FRACTION_OF_CLEARSKY = 0.02
MIN_DAYLIGHT_SAMPLES_PER_DAY = 60

# Every fifth local day is held out of the fit and used to report skill.
# Whole days, never samples: neighbouring samples share a sky cell and would leak.
HOLDOUT_EVERY_NTH_DAY = 5


@dataclass(frozen=True, slots=True)
class Samples:
    """Everything the map builder needs, one frame per array.

    Each frame has columns ``az``, ``el``, ``beam``, ``diffuse`` (unshaded output
    in kW, split by component) and ``p`` (measured kW), restricted to verified
    clear, uncurtailed daylight. ``holdout`` marks the reporting days.
    """

    arrays: dict[str, pd.DataFrame]
    first: np.datetime64
    last: np.datetime64
    usable_count: int
    clear_count: int


def read_power(path: Path, site: Site) -> pd.DataFrame:
    """Read the power CSV onto a regular five-minute UTC grid, in watts."""
    frame = pd.read_csv(path)
    if "time" not in frame.columns:
        msg = f"{path} has no 'time' column"
        raise ValueError(msg)
    names = [array.name for array in site.arrays]
    missing = [name for name in names if name not in frame.columns]
    if missing:
        msg = f"{path} is missing a column for each array in the site file: {missing}"
        raise ValueError(msg)

    stamps = pd.to_datetime(frame["time"], utc=True, format="ISO8601")
    frame = frame.drop(columns="time").set_index(stamps).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    # Average onto the grid rather than sampling it, so a one-minute export and a
    # five-minute one describe the same interval.
    grid = frame.resample(STEP).mean()
    columns = [*names, "soc"] if "soc" in grid.columns else names
    return grid[columns].astype(float)


def read_satellite(cache: Path, times: pd.DatetimeIndex) -> pd.Series:
    """Observed GHI from the archive cache, interpolated onto ``times``."""
    with sqlite3.connect(f"file:{cache}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT valid_utc, value FROM satellite WHERE variable = 'shortwave_radiation' ORDER BY valid_utc"
        ).fetchall()
    if not rows:
        return pd.Series(np.nan, index=times)
    observed = pd.Series(
        [value for _, value in rows],
        index=pd.to_datetime([stamp for stamp, _ in rows], utc=True, format="ISO8601"),
    )
    return observed.reindex(observed.index.union(times)).interpolate("time", limit=3).reindex(times)


def _flat_top(measured_kw: pd.Series, expected_kw: pd.Series) -> pd.Series:
    """Output pinned while the model says it should be moving."""
    flat = (
        (measured_kw.diff().abs() <= FLAT_TOP_MAX_STEP_KW)
        .rolling(FLAT_TOP_WINDOW, min_periods=FLAT_TOP_WINDOW)
        .min()
        .fillna(0)
        .astype(bool)
    )
    moving = expected_kw.diff().abs() >= FLAT_TOP_MIN_MODEL_STEP_KW
    return flat & moving & (measured_kw >= FLAT_TOP_MIN_POWER_KW)


def _outage_days(measured_kw: pd.Series, expected_kw: pd.Series, days: np.ndarray) -> np.ndarray:
    """Flag every sample on a day to drop whole: dead inverters, comms gaps."""
    measured = np.nan_to_num(measured_kw.to_numpy(dtype=np.float64))
    expected = expected_kw.to_numpy(dtype=np.float64)
    bad = np.zeros(days.size, dtype=bool)
    for day in np.unique(days):
        on_day = days == day
        modelled = float(expected[on_day].sum())
        ratio = float(measured[on_day].sum()) / modelled if modelled > 0.0 else 0.0
        producing = int((measured[on_day] > 0.05).sum())  # noqa: PLR2004 - 50 W: "producing at all"
        bad[on_day] = ratio < MIN_DAILY_FRACTION_OF_CLEARSKY or producing < MIN_DAYLIGHT_SAMPLES_PER_DAY
    return bad


def _temperature_factor(poa_global: FloatArray, gamma_pdc: float) -> FloatArray:
    """Irradiance-driven cell-temperature derate.

    There is no ambient temperature in the inputs, so this captures only the part
    that scales with irradiance -- which is the part correlated with sun
    elevation, and would otherwise leak into the map as a fake low-sun shadow.
    """
    cell = 25.0 + 30.0 * poa_global / 800.0
    return np.clip(1.0 + gamma_pdc * (cell - 25.0), 0.5, 1.0)


def build_samples(site: Site, power: pd.DataFrame, satellite_ghi: pd.Series) -> Samples:
    """Compute geometry once, mask, and split into per-array sample frames."""
    index = pd.DatetimeIndex(power.index)
    times: TimeArray = index.tz_convert("UTC").tz_localize(None).to_numpy().astype("datetime64[s]")
    geometry = solar_geometry(times, site.spec)
    clearsky = clear_sky(times, site.spec, geometry)

    unshaded: dict[str, tuple[FloatArray, FloatArray]] = {}
    for array in site.arrays:
        poa = plane_of_array(array, site.spec, geometry, clearsky.ghi, clearsky.dni, clearsky.dhi)
        scale = array.dc_capacity_w / 1000.0 / 1000.0 * _temperature_factor(poa.poa_global, array.gamma_pdc)
        unshaded[array.name] = (poa.poa_beam * scale, poa.poa_diffuse * scale)

    names = [array.name for array in site.arrays]
    measured_kw = power[names] / 1000.0
    total_kw = measured_kw.sum(axis=1, min_count=len(names))
    expected_kw = pd.Series(sum(beam + diffuse for beam, diffuse in unshaded.values()), index=index)
    # Local calendar days as integers, because every day boundary here -- outages,
    # the holdout split -- is a day the site lived through, not a UTC one.
    days = index.tz_convert(site.timezone).tz_localize(None).to_numpy().astype("datetime64[D]").astype(np.int64)

    daylight = pd.Series(geometry.apparent_elevation > MIN_ELEVATION_DEG, index=index)
    curtailed = _flat_top(total_kw, expected_kw)
    if "soc" in power.columns:
        curtailed |= power["soc"] >= SOC_FULL_PCT
    outage = _outage_days(total_kw, expected_kw, days)
    usable = daylight & ~curtailed & ~outage & total_kw.notna()

    with np.errstate(invalid="ignore", divide="ignore"):
        k = satellite_ghi.to_numpy() / np.where(clearsky.ghi > MIN_CLEARSKY_GHI, clearsky.ghi, np.nan)
    clear = pd.Series((k >= CLEAR_K_LOW) & (k <= CLEAR_K_HIGH), index=index)
    keep = usable & clear

    # Days since 2025-01-01, so the residue class matches the one the rev-4 map held out.
    holdout = ((days - np.datetime64("2025-01-01", "D").astype(np.int64)) % HOLDOUT_EVERY_NTH_DAY) == 2  # noqa: PLR2004

    arrays: dict[str, pd.DataFrame] = {}
    for name in names:
        beam, diffuse = unshaded[name]
        frame = pd.DataFrame(
            {
                "az": geometry.azimuth,
                "el": geometry.apparent_elevation,
                "beam": beam,
                "diffuse": diffuse,
                "p": measured_kw[name].to_numpy(),
                "holdout": holdout,
            },
            index=index,
        )
        # Only samples where the beam is a meaningful part of the signal: in the
        # rest the shadow has nothing to bite on.
        wanted = keep.to_numpy() & measured_kw[name].notna().to_numpy() & (beam > 0.05)  # noqa: PLR2004 - 50 W
        arrays[name] = frame.loc[wanted]

    return Samples(
        arrays=arrays,
        first=times[0],
        last=times[-1],
        usable_count=int(usable.sum()),
        clear_count=int(keep.sum()),
    )
