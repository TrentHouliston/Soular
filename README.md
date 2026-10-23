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

Soular does its own Perez transposition, applies a measured graded transmittance map, and
continuously bias-corrects against your measured output. That is all it does, on purpose.
Earlier versions also blended in a satellite nowcast and published a P10/P90 band from the
weather ensemble. Neither earned its keep: the nowcast helped the first two hours and nothing
after, and the band could never be checked against reality, because Open-Meteo keeps no
archive of past ensemble runs. Both were removed.

Every one of those claims is checked by an offline harness (`tools/backtest.py`) that replays
the model against measured production, at honest lead times, with each variant separately
re-scaled so an ablation measures skill rather than bias.

## Does it actually do better?

Replayed against a year of measured five-minute production (October 2025 to October 2026,
about a million samples on days held out from any fitting), at honest lead times:

|                                                    | RMSE as served | bias as served | RMSE re-scaled |
| -------------------------------------------------- | -------------: | -------------: | -------------: |
| **Soular**                                         |    **2,119 W** |     **+161 W** |        2,092 W |
| Soular without the learned correction              |        2,209 W |         +424 W |        2,100 W |
| Soular without shading                             |        3,077 W |       +1,220 W |        2,530 W |
| `ha-open-meteo-solar-forecast`, same horizon files |        2,675 W |         +773 W |        2,140 W |

That is 21% lower error and 79% less bias than the incumbent as a user would see it, on a
27.3 kWp site (2,119 W is 7.8% of capacity). "Re-scaled" gives every model its own best
constant gain, fitted on other days, which isolates the *shape* of the forecast from its
calibration: there the gap narrows to 2%. Most of the practical difference is calibration,
which Soular learns from your inverter and the incumbent expects you to type in.

One caveat on that table: the shading map was built from this same year, so for the
map those numbers are partly in-sample. The clean test is a map built only from data up to
2026-08-08 and scored on the ten weeks after it, which serves 1,936 W.

The rest of the error is mostly the weather. Given the sky as the satellite actually observed
it, rather than as it was forecast, the same model scores 1,580 W. No amount of modelling
the roof better touches that difference.

The incumbent emulation and the earlier, longer comparison are in
[docs/vs-open-meteo-solar-forecast.md](docs/vs-open-meteo-solar-forecast.md); its numbers
predate the shading-map tool and a channel-mapping fix, so the table above supersedes them.

## Status

Everything described above is built and tested. It has not yet been through a
tagged release or a HACS submission, and every measured number on this page comes
from one site.

Known gaps, listed because they are the ones a user would notice: the learner
starts cold rather than warm-starting from the recorder, so a fresh install
spends about a week reaching its calibration, and there are no skill diagnostics
reporting the forecast's own error against a baseline.

## Installation

Add `https://github.com/TrentHouliston/soular` to HACS as a custom repository,
restart, then add the integration. Add one array per roof plane; each becomes its
own device under the site. Full setup, entity list, wiring and troubleshooting are
in [docs/configuration.md](docs/configuration.md).

## Making a shading map

The shading map is where most of the accuracy comes from, and it is built from your
own array's measured output. With about a year of five-minute per-array power in a
CSV, `uv run shading-map` turns it into the `.npz` the integration reads. See
[docs/shading-maps.md](docs/shading-maps.md).

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

| Path                             |                                                                            |
| -------------------------------- | -------------------------------------------------------------------------- |
| `custom_components/soular/`      | The integration. Only this ships to HACS.                                  |
| `custom_components/soular/core/` | The physics and the learner. Home Assistant free by construction.          |
| `tools/shading/`                 | Builds a shading map from measured per-array power (`uv run shading-map`). |
| `tools/backtest.py`              | Replays the model against measured production (`uv run backtest`).         |
| `tools/fetch_archive.py`         | Builds the weather archive both tools read (`uv run fetch-archive`).       |

`core/` is limited by an import-linter contract to numpy, pandas and pvlib, and both tools
reach the model only through it. That boundary is load-bearing: the backtest replays the
exact code path the integration runs, and the shading map is measured through the same
optics that will consume it.

## Licence

MIT.
