# Soular versus `ha-open-meteo-solar-forecast`

A like-for-like replay of both integrations against 297 days of measured
five-minute production from a 27.28 kWp site at Morisset Park, NSW — roughly
812,000 scored samples on holdout days.

The incumbent is not approximated. `tools/incumbent.py` is written from
[rany2/open-meteo-solar-forecast](https://github.com/rany2/open-meteo-solar-forecast)'s
source and fed the irradiance that integration actually consumes: it does not
transpose, it asks Open-Meteo for `global_tilted_irradiance` with a tilt and
azimuth and uses the answer. The archive holds that variable per array at fixed
lead times, requested with the same `azimuth - 180` conversion the upstream
coordinator performs. Both sides see forecasts of the same age.

## Headline

| variant                |    raw RMSE |   raw bias | re-scaled RMSE | fitted scalar | daily energy (median APE) |
| ---------------------- | ----------: | ---------: | -------------: | ------------: | ------------------------: |
| **soular, everything** | **2,249 W** | **+161 W** |    **2,228 W** |     **0.941** |                 **18.9%** |
| soular, no learning    |     2,337 W |     +452 W |        2,228 W |         0.864 |                     20.8% |
| incumbent              |     3,112 W |   +1,079 W |        2,339 W |         0.695 |                     21.8% |
| soular, no shading     |     3,094 W |   +1,223 W |        2,523 W |         0.718 |                     34.5% |

Two numbers, and they answer different questions.

**As served: 27.7% lower RMSE and 85% less bias.** `raw` applies no scalar to
either side — it is what each integration actually publishes. This is the number
that describes what a user experiences.

**Re-scaled: 4.8% lower RMSE.** Give each side its own least-squares scalar,
fitted on held-out days, and the gap narrows sharply. This isolates the *shape*
of the forecast from its calibration.

The distance between those two figures is the whole story, and it deserves to be
stated plainly rather than buried.

## Most of the practical gap is calibration

The incumbent needs a scalar of **0.695** to be unbiased at this site. It has a
place to put one — the `efficiency_factor` option — so a user who measured their
own output, worked out the right value and typed it in would recover most of the
difference. Worth knowing: upstream's README suggests "typically around 0.93",
and the default is 1.0, so anyone following that guidance is left with a forecast
over-predicting by more than 40%.

Soular does not ask. Its online correction learns the same constant from the
array's own output and lands at **0.941**, and because it forgets exponentially
it tracks soiling, pruning and degradation rather than freezing a number typed in
once. That is a difference in kind, not in accuracy: a constant is a maintenance
task, and a learned one is not.

So the fair summary is: **the incumbent could close most of this gap with manual
tuning, and cannot close any of it without.**

## What is left when both are calibrated

The 4.8% that survives re-scaling is model shape, and it shows up where it should
— at short lead, where the satellite nowcast operates. Generating hours only,
each variant carrying its own fitted scalar:

| lead   | incumbent |  soular | difference |
| ------ | --------: | ------: | ---------: |
| 0–1h   |   3,747 W | 3,329 W |     +11.2% |
| 1–2h   |   3,826 W | 3,414 W |     +10.8% |
| 2–6h   |   3,878 W | 3,679 W |      +5.1% |
| 6–24h  |   3,841 W | 3,668 W |      +4.5% |
| 24–48h |   3,968 W | 3,812 W |      +3.9% |

Better everywhere, and roughly twice as much better inside two hours. That is the
nowcast, which the incumbent has no equivalent of.

## The labelling offset is not the reason

Upstream accumulates power computed from irradiance at time `T` under the key
`T - 15min`, so its instantaneous power series leads reality by about a quarter
hour. That is a genuine defect and it is reproduced here — but disabling it makes
upstream **worse**, 2,339 W to 2,375 W. The offset is not what is costing it.
Reporting it as the cause would have been an easy and wrong story.

## What actually accounts for the difference

In rough order of contribution, on this site:

1. **Calibration** — 0.695 versus 0.941. Dominates everything else, and is the
    difference between a learned correction and a manual constant.
2. **Graded shading** — worth 11.4% on its own against no shading at all. The
    incumbent supports hard-horizon files and uses them here, so it gets most of
    this; the graded map's advantage over a hard horizon is a few percent.
3. **The satellite nowcast** — 7.4% at 0–1h and 5.0% at 1–2h, fading to nothing
    by two hours.
4. **Everything else** — Perez transposition, Faiman thermal with wind, inverter
    efficiency curve, clipping on a fine grid. Individually small at a 25° roof
    pitch. Perez in particular is worth about 0.75% of *shape* here, which is
    honest to say given how often it is oversold.

## Caveats

- **One site.** A 27.28 kWp array with heavy tree shading at 33°S. The shading
    terms will matter less on an unobstructed roof, and the calibration term will
    matter wherever the model over-predicts, which appears to be generally.
- **The incumbent's efficiency factor is unset here**, matching this site's real
    configuration. Setting it correctly is the single change that would most
    improve it.
- **Lead times are conservative for both sides.** The archive resolves runs to
    whole days and the harness rounds up, so short-lead figures are a floor.
- **Curtailed samples are excluded**, since a full battery is lost production
    neither integration could predict.
- **Sub-hourly resolution is generous to the incumbent.** It serves a
    quarter-hour step function; this replay interpolates it linearly rather than
    stepping, which removes an artefact that is about resolution rather than skill.

## Reproducing

```bash
uv run fetch-archive \
  --latitude -33.11915471966274 --longitude 151.53401076793673 \
  --start 2025-10-16 --end 2026-08-08 --feeds nwp,satellite,gti \
  --arrays /path/to/arrays.toml --cache backtest_cache.db

uv run backtest \
  --db /path/to/solar.db --arrays /path/to/arrays.toml \
  --shading /path/to/transmittance_grid.npz --horizons /path/to/outputs \
  --cache backtest_cache.db --start 2025-10-17 --end 2026-08-07 \
  --ac-limit-w 30000 --issue-every 3 \
  --variants soular,nowcast,learning,upstream,upstream-no-offset,no-shading \
  --out report.md
```
