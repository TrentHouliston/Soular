# Making a shading map

A shading map tells Soular how much direct sunlight reaches each array from every
part of the sky. It is built from your own measured output: on a clear day the only
thing between what an array should have produced and what it did produce is
whatever blocked the sun, so a year of sun positions paints a picture of the sky
each array can see.

On the site Soular was built against, a measured map cut forecast error by about
a quarter against no shading at all, and beats a hand-drawn horizon line because
real trees are porous rather than solid.

## What you need

- **About a year of per-array DC power at five-minute resolution.** A year lets
    the sun cross every part of the sky it ever will. Less still works, but the
    map will fall back to a smooth guess in directions the sun has not visited
    yet.
- **The site file** describing each array, the same geometry you enter in
    Home Assistant.
- **A Soular checkout**, with [uv](https://docs.astral.sh/uv/) installed:
    `uv sync`.

### The power CSV

```csv
time,east,west,north,south,soc
2026-01-01T00:00:00Z,0,0,0,0,54
2026-01-01T00:05:00Z,0,0,0,0,54
```

- `time` is ISO 8601. Include a timezone; times without one are read as UTC.
- One column per array, named exactly as in the site file, in **watts**.
    Finer data is fine — it is averaged onto five minutes.
- `soc` is optional battery state of charge, in percent. A full battery
    throttles the panels at the sunniest part of the day, which looks exactly like
    a tree in the sky; with `soc` those samples are dropped. Without it the
    builder still catches most curtailment from the output going flat.

Per-array (per-MPPT) power is the part most inverters hide. Look for "PV1/PV2
power", "string power" or "MPPT power" in your inverter's app, API or Modbus
registers. If your strings were ever re-plugged into different inputs, remap the
columns so each one always means the same physical array.

### The site file

```toml
[site]
latitude = -33.12
longitude = 151.53
altitude = 10                 # metres; optional
timezone = "Australia/Sydney"
albedo = 0.20                 # optional; match the ground surface set in Home Assistant

[arrays.east]
azimuth = 84    # compass degrees: 0 north, 90 east, 180 south, 270 west
tilt = 25
dc_watts = 7920
```

## Building it

```bash
uv run shading-map --site site.toml --power power.csv --out my-maps/
```

It downloads observed sky brightness from Open-Meteo's satellite archive for the
same dates, which is how it knows which samples were genuinely clear, and caches
it in `backtest_cache.db`. That takes a minute the first time.

The default satellite (Himawari) covers **Asia and Oceania only**. Elsewhere, pass
the Open-Meteo satellite model that covers you with `--satellite-model`.

Output, in `my-maps/`:

| File                     |                                                                                     |
| ------------------------ | ----------------------------------------------------------------------------------- |
| `transmittance_grid.npz` | The map. One entry per array. This is the file Soular wants.                        |
| `horizon-<array>.txt`    | The fitted horizon line per array, for anything that only takes a horizon file.     |
| `build.json`             | What went in, and how well the map predicts days it was not built from (see below). |

Copy `transmittance_grid.npz` into `config/soular/` on your Home Assistant and
give its filename as each array's **Shading file**. One file serves every array:
Soular looks up the entry named after the array, so the array names in Home
Assistant must match the ones in the site file.

## Is it any good?

The builder holds back every fifth day, builds from the rest, and reports how
well each array's clear-sky output on the held-back days is predicted with and
without the map:

```
array      clear samples edge width   holdout RMSE, % of capacity
east              15,743       3.5°    22.7% ->  12.9%  (n=3,149)
north             16,458       2.4°    22.6% ->  13.4%  (n=3,302)
```

The second number should be clearly lower than the first. If an array barely
improves, either it is genuinely unshaded or its column is not the array you
think it is — two arrays with swapped columns both get worse.

## Rebuilding

Trees grow and get pruned. Rebuilding once a year is plenty. Pass `--seed` with
the directory of a previous build to start the horizon fit from the old horizons;
without it the fit starts from a horizon read straight off the data, which works
just as well.

`--start` and `--end` restrict the dates, which is also how to check a map
honestly: build it with `--end` set to some date, and score it on what came after.
