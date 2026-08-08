# Soular

Solar production forecasting for Home Assistant, built for accuracy rather than convenience.

## Why

Most Home Assistant solar forecasts ask a weather API for plane-of-array irradiance and
scale it by a nameplate rating. That hides several approximations that matter:

- **Isotropic sky.** Open-Meteo's `global_tilted_irradiance` uses an isotropic diffuse model
    with a hardcoded albedo of 0.2. It has no circumsolar brightening and no horizon band, so
    it under-reads clear-sky plane-of-array irradiance on a tilted plane.
- **Binary shading.** A horizon polyline says "blocked" or "not blocked". Real trees are
    porous and shade partially across a band of elevations.
- **No feedback.** Your inverter knows exactly how wrong yesterday's forecast was, and
    nothing reads it.
- **A single number.** A point estimate can't tell a battery optimiser how much to hedge.

Soular does its own Perez transposition, applies a measured graded transmittance map, blends
satellite observations and your own array into the near-term forecast, produces P10/P50/P90
from NWP ensembles, and continuously bias-corrects against your measured output.

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
