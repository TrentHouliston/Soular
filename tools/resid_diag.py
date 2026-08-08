"""Residual diagnostic for the forward model.

Reports soular's own no-shading residual (actual/predicted) against sun
elevation and incidence angle, restricted to open-sky directions, per array.
Isolates forward-model error from tree shading.

Run from the repo root: ``uv run python tools/resid_diag.py``
"""

import datetime as dt
from pathlib import Path

import numpy as np

from custom_components.soular.core.geometry import angle_of_incidence, solar_geometry
from custom_components.soular.core.pipeline import forecast
from custom_components.soular.core.shading import from_npz
from custom_components.soular.core.types import TimeGrid
from tools.backtest import (
    HORIZON_HOURS,
    STEP_SECONDS,
    VARIANTS,
    Archive,
    build_system,
    is_fit_day,
    issue_times,
    load_actuals,
    load_site,
    to_local_dates,
    uniform_grid,
    weather_for,
)

SD = Path("/Users/trenthouliston/Code/solar-data")
site = load_site(SD / "config/arrays.toml", ac_limit_w=30000)
names = [a.name for a in site.arrays]
times = uniform_grid(dt.date(2025, 10, 17), dt.date(2026, 8, 7))
actuals = load_actuals(SD / "data/solar.db", times, names, curtailment_soc_pct=97.0)
archive = Archive(SD.parent / "soular/backtest_cache.db")
tz = site.timezone
mapraw = (SD / "outputs/transmittance_grid.npz").read_bytes()
shmap = {n: from_npz(mapraw, n) for n in names}
noshade = next(v for v in VARIANTS if v.name == "no-shading-observed")

pos = {s: i for i, s in enumerate(actuals.times)}
# accumulate per array: elevation bin, aoi bin -> sum ratio, count ; open-sky only
EL = np.arange(0, 60, 5.0)
AOI = np.arange(0, 90, 10.0)
acc = {
    n: {"el_s": np.zeros(len(EL)), "el_n": np.zeros(len(EL)), "ao_s": np.zeros(len(AOI)), "ao_n": np.zeros(len(AOI))}
    for n in names
}
issues = list(issue_times(dt.date(2025, 10, 17), dt.date(2026, 8, 7), 12))
for issue in issues:
    t = np.arange(issue, issue + np.timedelta64(HORIZON_HOURS * 3600, "s"), np.timedelta64(STEP_SECONDS, "s")).astype(
        "datetime64[s]"
    )
    rows = np.array([pos.get(s, -1) for s in t])
    inside = rows >= 0
    if inside.sum() < t.size // 2:
        continue
    grid = TimeGrid(times=t, step_seconds=np.full(t.size, float(STEP_SECONDS)))
    usable = inside & actuals.valid[np.clip(rows, 0, None)] & ~actuals.curtailed()[np.clip(rows, 0, None)]
    if not usable.any():
        continue
    days = to_local_dates(t, tz)
    if is_fit_day(days[0]):
        continue  # holdout only
    weather = weather_for(archive, site, grid, issue, "observed")
    if weather is None:
        continue
    res = forecast(build_system(site, {}, {}, noshade), grid, weather)
    geo = solar_geometry(t, site.spec)
    elev = geo.apparent_elevation
    for entry, arr in zip(res.arrays, site.arrays):
        pred = entry.dc_power_w
        act = np.where(usable, actuals.dc_by_array[arr.name][np.clip(rows, 0, None)], np.nan)
        aoi = angle_of_incidence(arr.azimuth_deg, arr.tilt_deg, geo)
        T = shmap[arr.name].lookup(geo.azimuth, elev)  # open-sky filter
        m = usable & np.isfinite(act) & (pred > 200) & (elev > 3) & (T > 0.9)
        if not m.any():
            continue
        r = act[m] / pred[m]
        e = elev[m]
        a = aoi[m]
        ei = np.clip((e // 5).astype(int), 0, len(EL) - 1)
        ai = np.clip((a // 10).astype(int), 0, len(AOI) - 1)
        for i in range(len(r)):
            acc[arr.name]["el_s"][ei[i]] += r[i]
            acc[arr.name]["el_n"][ei[i]] += 1
            acc[arr.name]["ao_s"][ai[i]] += r[i]
            acc[arr.name]["ao_n"][ai[i]] += 1

print("=== soular no-shading residual actual/pred in OPEN SKY, by ELEVATION ===")
print("elev   " + "  ".join(f"{n[:5]:>6}" for n in names))
for j, e in enumerate(EL):
    cells = []
    for n in names:
        s, c = acc[n]["el_s"][j], acc[n]["el_n"][j]
        cells.append(f"{s / c:6.3f}" if c > 200 else "   -- ")
    print(f"{int(e):2d}-{int(e) + 5:2d}  " + "  ".join(cells))
print("\n=== same, by AOI ===")
print("aoi    " + "  ".join(f"{n[:5]:>6}" for n in names))
for j, a in enumerate(AOI):
    cells = []
    for n in names:
        s, c = acc[n]["ao_s"][j], acc[n]["ao_n"][j]
        cells.append(f"{s / c:6.3f}" if c > 200 else "   -- ")
    print(f"{int(a):2d}-{int(a) + 10:2d}  " + "  ".join(cells))
