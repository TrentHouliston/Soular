r"""Fit the weather correction's starting point from a year of record.

The live weather correction learns how far the forecast's sky misses, one ratio
per band of forecast clearness, over the last sixty days. From a fresh install
it has nothing, and a week of history is easily one unrepresentative week. The
bias is steady enough across a year that a year's fit is most of the answer, so
it is fitted here, offline, and written into the shading map, where the
integration starts each band from it and lets the live record follow the season.

It replays the record through exactly the forecast the backtest scores -- the
integration's own model at honest lead times, with the map's panel efficiency
applied -- and compares each instant's first forecast with what was measured,
leaving out curtailed, unread and switched-off instants as the integration does::

    uv run weather-prior --db solar.db --arrays arrays.toml \
        --shading transmittance_grid.npz --cache backtest_cache.db \
        --start 2025-10-24 --end 2026-10-21 --out transmittance_grid.npz
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import datetime as dt
import io
from pathlib import Path

import numpy as np

from custom_components.soular.core import weather_bias
from custom_components.soular.core.clearsky import clear_sky, clear_sky_index
from custom_components.soular.core.geometry import solar_geometry
from custom_components.soular.core.mask import outage
from custom_components.soular.core.pipeline import forecast
from custom_components.soular.core.shading import from_npz
from custom_components.soular.core.types import TimeGrid
from tools.backtest import (
    HORIZON_HOURS,
    STEP_SECONDS,
    VARIANTS,
    Archive,
    build_system,
    issue_times,
    load_actuals,
    uniform_grid,
    weather_for,
)
from tools.site import load_site

# A bin's ratio needs this much measured energy behind it, in watt-hours, before
# it is written; a thinner bin is left at 1 for the live record to fill.
MIN_BIN_WH = 50_000.0


def fit(
    predicted_w: np.ndarray,
    actual_w: np.ndarray,
    clearness: np.ndarray,
    usable: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return each bin's measured-over-forecast ratio and the forecast energy behind it."""
    keep = usable & np.isfinite(predicted_w) & np.isfinite(actual_w) & (predicted_w > 0.0)
    bins = weather_bias.bin_of(clearness[keep])
    hours = STEP_SECONDS / 3600.0
    predicted = np.bincount(bins, weights=predicted_w[keep] * hours, minlength=weather_bias.BINS)
    actual = np.bincount(bins, weights=actual_w[keep] * hours, minlength=weather_bias.BINS)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(predicted >= MIN_BIN_WH, actual / predicted, 1.0)
    return np.clip(ratio, weather_bias.MIN_SCALE, weather_bias.MAX_SCALE), predicted


def main(argv: Sequence[str] | None = None) -> int:
    """Fit the prior and write it into a copy of the map."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, required=True, help="the Sigen extract, solar.db")
    parser.add_argument("--arrays", type=Path, required=True, help="arrays.toml")
    parser.add_argument("--shading", type=Path, required=True, help="the map to read, and to copy")
    parser.add_argument("--cache", type=Path, default=Path("backtest_cache.db"))
    parser.add_argument("--start", type=dt.date.fromisoformat, required=True, help="first day with every array on")
    parser.add_argument("--end", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--issue-every", type=int, default=6, help="hours between forecast issue times")
    parser.add_argument("--out", type=Path, required=True, help="where to write the map carrying the prior")
    args = parser.parse_args(argv)

    site = load_site(args.arrays, ac_limit_w=100000.0)
    names = [array.name for array in site.arrays]
    raw = args.shading.read_bytes()
    shading = {name: from_npz(raw, name) for name in names}
    times = uniform_grid(args.start, args.end)
    actuals = load_actuals(args.db, times, names)
    archive = Archive(args.cache)
    system = build_system(site, shading, {}, next(v for v in VARIANTS if v.name == "soular"))
    position = {stamp: index for index, stamp in enumerate(actuals.times)}
    capacities = [array.dc_capacity_w for array in site.arrays]

    predicted = np.full(times.size, np.nan)
    per_array = np.full((len(names), times.size), np.nan)
    clearness = np.full(times.size, np.nan)
    for issue in issue_times(args.start, args.end, args.issue_every):
        horizon = np.arange(
            issue, issue + np.timedelta64(HORIZON_HOURS * 3600, "s"), np.timedelta64(STEP_SECONDS, "s")
        ).astype("datetime64[s]")
        rows = np.array([position.get(stamp, -1) for stamp in horizon])
        inside = rows >= 0
        if not inside.any():
            continue
        grid = TimeGrid(times=horizon, step_seconds=np.full(horizon.size, float(STEP_SECONDS)))
        weather = weather_for(archive, site, grid, issue, "forecast")
        if weather is None:
            continue
        result = forecast(system, grid, weather)
        sky = clear_sky(horizon, site.spec, solar_geometry(horizon, site.spec)).ghi
        k = np.nan_to_num(clear_sky_index(weather.ghi, sky), nan=1.0)
        # The first forecast made for each instant, as the live record keeps.
        target = rows[inside]
        unset = np.isnan(predicted[target])
        positions = target[unset]
        arrays = np.array([result.array(name).dc_power_w[inside][unset] for name in names])
        predicted[positions] = arrays.sum(axis=0)
        per_array[:, positions] = arrays
        clearness[positions] = k[inside][unset]

    measured = np.array([actuals.dc_by_array[name] for name in names])
    usable = actuals.valid & ~actuals.curtailed() & np.isfinite(measured).all(axis=0) & np.isfinite(predicted)
    for index in np.flatnonzero(usable):
        if outage(measured[:, index].tolist(), per_array[:, index].tolist(), capacities):
            usable[index] = False

    prior, energy = fit(predicted, measured.sum(axis=0), clearness, usable)
    print(f"Replayed {args.start} .. {args.end}: {int(usable.sum()):,} usable instants")
    edges = (0.0, *weather_bias.BIN_EDGES, float("inf"))
    for index, (ratio, wh) in enumerate(zip(prior, energy, strict=True)):
        print(f"  clearness {edges[index]:.2f}-{edges[index + 1]:.2f}: {ratio:.3f}  ({wh / 1000.0:,.0f} kWh forecast)")

    with np.load(io.BytesIO(raw), allow_pickle=False) as archive_file:
        payload = {key: archive_file[key] for key in archive_file.files}
    payload["weather_prior"] = prior
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **payload)  # pyright: ignore[reportArgumentType] - numpy types **kwds as its keyword options
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(buffer.getvalue())
    # Only worth writing if the integration can read it.
    for name in names:
        from_npz(args.out.read_bytes(), name)
    print(f"Written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
