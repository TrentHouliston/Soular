# Soular

Solar production forecasting for Home Assistant, built for accuracy rather than convenience.

## Why

Most Home Assistant solar forecasts ask a weather API for plane-of-array irradiance and
scale it by a nameplate rating. That hides several approximations that matter:

- **Binary shading.** A horizon polyline says "blocked" or "not blocked". Real trees are
    porous and shade partially across a band of elevations. This is the big one: on the site
    Soular was built against, a measured graded transmittance map cuts site RMSE by around
    15% once weather error is taken out of the comparison, and by more than 20% on the two
    worst-affected arrays.
- **No feedback.** Your inverter knows exactly how wrong yesterday's forecast was, and
    nothing reads it.
- **A single number.** A point estimate can't tell a battery optimiser how much to hedge.
- **Isotropic sky.** Open-Meteo's `global_tilted_irradiance` uses an isotropic diffuse model
    with a hardcoded albedo of 0.2 — no circumsolar brightening, no horizon band.

That last one deserves an honest caveat, because it is the one most often oversold. Perez
transposition really does collect more irradiance than isotropic — measured here as +1.9%
annually at 25° tilt, rising to +5.0% at 90° — but most of that is a constant scale factor,
and any fitted or learned efficiency absorbs a constant. What survives calibration is the
*shape* error, which is 0.75% at 25° tilt and 1.8% at 90°. So Perez is the right default and
it matters for getting absolute kWh right without calibration, but at a typical roof pitch it
is not where the accuracy comes from. Soular uses it because it is free and correct, not
because it is the headline.

Soular does its own Perez transposition, applies a measured graded transmittance map, blends
satellite observations and your own array into the near-term forecast, produces P10/P50/P90
from NWP ensembles, and continuously bias-corrects against your measured output.

Every one of those claims is checked by an offline harness (`tools/backtest.py`) that replays
the model against measured production, at honest lead times, with each variant separately
re-scaled so an ablation measures skill rather than bias.

## Does it actually do better?

Against `ha-open-meteo-solar-forecast` on 297 days of measured five-minute production,
with that integration emulated from its own source and fed the same tilted irradiance it
really consumes: **27.7% lower RMSE and 85% less bias as served**, narrowing to 4.8% once
both sides are given an oracle-fitted scalar. Most of the practical gap is calibration,
which soular learns and the incumbent expects you to type in. Full numbers, decomposition
and caveats in [docs/vs-open-meteo-solar-forecast.md](docs/vs-open-meteo-solar-forecast.md).

## Status

Early development. See `docs/` for the design and the measured skill numbers.

## Installation

Not yet released. Once tagged, install through HACS as a custom repository.

### A note on size

Soular depends on [pvlib](https://pvlib-python.readthedocs.io/), which brings in pandas,
SciPy and h5py — roughly 150 MB installed, and about 80–120 MB of resident memory once
loaded. On a Raspberry Pi that is a real cost. It buys a validated, peer-reviewed
implementation of every irradiance and PV model Soular uses, numerically identical to the
analysis that produced the shading maps. If your hardware is tight, the incumbent
`open_meteo_solar_forecast` integration is much lighter.

## Development

```bash
uv sync --all-extras --dev
uv run pytest
uv run ruff check && uv run ruff format --check
uv run pyright
uv run lint-imports
```

`custom_components/soular/core/` is Home Assistant free by construction — an import-linter
contract limits it to numpy, pandas and pvlib. That boundary is load-bearing: the offline
backtest harness in `tools/` replays the exact same code path the integration runs, so the
skill numbers it reports describe the shipped model rather than an approximation of it.

## Licence

MIT.
