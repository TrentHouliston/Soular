"""Command line: power CSV in, shading map out."""

import argparse
from collections.abc import Sequence
import datetime as dt
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from custom_components.soular.core.shading import from_npz
from tools.fetch_archive import connect, fetch_satellite
from tools.shading import grid as grid_module
from tools.shading.horizon import Horizon, fit, read_horizon
from tools.shading.samples import Samples, build_samples, read_power, read_satellite
from tools.site import load_site


def _npz_bytes(fields: dict[str, np.ndarray]) -> bytes:
    """Serialise the grids in the format the integration reads."""
    arrays = {
        "azimuth_deg": grid_module.AZIMUTH_AXIS,
        "elevation_deg": grid_module.ELEVATION_AXIS,
        **{f"T_{name}": values for name, values in fields.items()},
    }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)  # pyright: ignore[reportArgumentType] - numpy types **kwds as its keyword options
    return buffer.getvalue()


def _holdout_skill(samples: pd.DataFrame, lookup: object, capacity_w: float) -> tuple[float, float, int]:
    """RMSE on held-out clear samples, without and with the map, % of capacity.

    Each side gets its own least-squares gain from the training days, so this
    measures shading shape rather than which side happened to be better
    calibrated.
    """
    held = samples["holdout"].to_numpy(dtype=bool)
    train, test = samples.loc[~held], samples.loc[held]
    if test.empty or train.empty:
        return float("nan"), float("nan"), 0

    def rmse(transmittance: object) -> float:
        def predicted(frame: pd.DataFrame) -> np.ndarray:
            t = transmittance(frame["az"].to_numpy(), frame["el"].to_numpy())  # type: ignore[operator]
            return frame["diffuse"].to_numpy() + t * frame["beam"].to_numpy()

        x, y = predicted(train), train["p"].to_numpy()
        gain = float(np.dot(x, y) / np.dot(x, x))
        error = gain * predicted(test) - test["p"].to_numpy()
        return float(np.sqrt(np.mean(error**2))) * 1000.0 / capacity_w * 100.0

    return rmse(lambda az, _el: np.ones_like(az)), rmse(lookup), len(test)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the maps and write them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True, help="site TOML (see tools/site.py)")
    parser.add_argument("--power", type=Path, required=True, help="per-array power CSV, watts")
    parser.add_argument("--out", type=Path, required=True, help="directory to write the map into")
    parser.add_argument("--cache", type=Path, default=Path("backtest_cache.db"), help="weather archive cache")
    parser.add_argument("--seed", type=Path, help="directory of horizon-<array>.txt to start the fit from")
    parser.add_argument("--start", type=dt.date.fromisoformat, help="ignore data before this date")
    parser.add_argument("--end", type=dt.date.fromisoformat, help="ignore data on or after this date")
    parser.add_argument(
        "--satellite-model",
        default="jma_jaxa_himawari",
        help="Open-Meteo satellite model; the default covers Asia and Oceania only",
    )
    args = parser.parse_args(argv)

    site = load_site(args.site, ac_limit_w=1.0)
    power = read_power(args.power, site)
    times = pd.DatetimeIndex(power.index)
    keep = np.ones(len(times), dtype=bool)
    if args.start:
        keep &= times >= pd.Timestamp(args.start, tz=site.timezone)
    if args.end:
        keep &= times < pd.Timestamp(args.end, tz=site.timezone)
    power, times = power.loc[keep], times[keep]
    if power.empty:
        parser.error("no power data in the selected range")
    print(f"Power: {times[0]} .. {times[-1]}, arrays {[a.name for a in site.arrays]}")

    connection = connect(args.cache)
    fetch_satellite(
        connection,
        site.spec.latitude,
        site.spec.longitude,
        times.tz_convert("UTC").to_numpy().astype("datetime64[D]")[0].item(),
        times.tz_convert("UTC").to_numpy().astype("datetime64[D]")[-1].item(),
        model=args.satellite_model,
    )
    connection.commit()
    connection.close()

    samples: Samples = build_samples(site, power, read_satellite(args.cache, times))
    print(f"Usable daylight samples: {samples.usable_count:,}; verified clear: {samples.clear_count:,}")

    fields: dict[str, np.ndarray] = {}
    horizons: dict[str, Horizon] = {}
    for array in site.arrays:
        frame = samples.arrays[array.name]
        train = frame.loc[~frame["holdout"].to_numpy(dtype=bool)]
        seed_path = args.seed / f"horizon-{array.name}.txt" if args.seed else None
        seed = read_horizon(seed_path.read_text()) if seed_path and seed_path.exists() else None
        horizons[array.name] = fit(train, seed)
        fields[array.name] = grid_module.build(train, horizons[array.name])

    payload = _npz_bytes(fields)
    # The file is only worth writing if the integration can read it.
    grids = {array.name: from_npz(payload, array.name) for array in site.arrays}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "transmittance_grid.npz").write_bytes(payload)
    report: dict[str, dict[str, float | int | bool]] = {}
    print(f"\n{'array':10} {'clear samples':>13} {'edge width':>10}   holdout RMSE, % of capacity")
    for array in site.arrays:
        horizon = horizons[array.name]
        (args.out / f"horizon-{array.name}.txt").write_text(horizon.to_text())
        without, with_map, n = _holdout_skill(samples.arrays[array.name], grids[array.name].lookup, array.dc_capacity_w)
        report[array.name] = {
            "clear_samples": len(samples.arrays[array.name]),
            "edge_width_deg": round(horizon.width, 2),
            "fit_converged": horizon.converged,
            "holdout_samples": n,
            "holdout_rmse_pct_without_map": round(without, 2),
            "holdout_rmse_pct_with_map": round(with_map, 2),
        }
        print(
            f"{array.name:10} {len(samples.arrays[array.name]):13,} {horizon.width:9.1f}°"
            f"   {without:5.1f}% -> {with_map:5.1f}%  (n={n:,})"
        )

    (args.out / "build.json").write_text(
        json.dumps(
            {
                "data_from": str(samples.first),
                "data_to": str(samples.last),
                "usable_samples": samples.usable_count,
                "clear_samples": samples.clear_count,
                "seeded_from": str(args.seed) if args.seed else None,
                "arrays": report,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"\nWrote {args.out / 'transmittance_grid.npz'} and horizon-<array>.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
