# What the backtest says

Measured by `tools/backtest.py` replaying the model against 297 days of
five-minute production from a 27.28 kWp site at Morisset Park, NSW
(−33.119, 151.534), October 2025 to August 2026. Roughly 406,000 five-minute
samples on holdout days.

Read the harness docstring before the numbers: forecasts come from archived model
runs at honest lead times, every variant is separately re-scaled before
comparison, and DC strings are scored against modelled DC while the AC meter is
scored against modelled AC.

## The short version

1. **Shading is the whole story.** Adding shading of any kind cuts site RMSE by
    12–15% against no shading, and by 23% once weather error is removed from the
    comparison. Per array it is 19–27%.
2. **Perez transposition does not measurably help at this roof pitch.** It is
    still the right default, for reasons below.
3. **The graded transmittance map currently loses to the hard horizon**, which
    contradicts what the analysis handoff expected. This is the finding worth
    acting on.
4. **Weather error dominates everything else.** The gap between forecast-driven
    and observation-driven runs is about 24% of RMSE. That gap is the budget for
    nowcasting, ensembles and online learning.

## Perez versus isotropic

Perez collects genuinely more plane-of-array irradiance than an isotropic sky —
measured on this site's clear-sky geometry:

| tilt | annual POA gain | residual after a fitted scalar |
| ---: | --------------: | -----------------------------: |
|   0° |           −0.0% |                          0.03% |
|  10° |           +0.9% |                          0.38% |
|  25° |           +1.9% |                          0.75% |
|  40° |           +2.7% |                          1.02% |
|  60° |           +3.7% |                          1.31% |
|  90° |           +5.0% |                          1.79% |

Almost all of the gain is a constant, and any fitted or learned efficiency
absorbs a constant. What survives calibration is the shape error in the last
column: 0.75% at this site's 25° pitch, buried under a forecast RMSE of about 8%
of capacity. The replay agrees — Perez scores within a percent of isotropic
either way, which is noise.

Perez stays the default because it costs nothing, because it is what gets
absolute kWh right for a user who never calibrates anything, and because the
advantage roughly triples on a steep or façade-mounted array. It is simply not
where accuracy comes from on a normal roof, and claiming otherwise would be
overselling it.

## The graded map versus the hard horizon

This is the surprise. The analysis handoff recommends the graded transmittance
grid over the hard horizon files, on the grounds that a porous canopy shades
partially and a step function cannot express that. The replay finds the opposite,
consistently and on both drivers:

- With forecast irradiance: hard horizon 2,163 W RMSE, graded grid 2,225 W —
    grading is **2.9% worse**.
- With observed irradiance: hard horizon 1,585 W, graded grid 1,680 W — grading
    is **6.0% worse**.
- Per array, weather removed, the hard horizon matches or beats the graded grid
    on all four.

Both had shading, both were separately re-scaled, so this is not a bias artefact.

### Why

The graded field applies substantial attenuation to sky that is demonstrably
open. Sampling both fields along the sun's actual annual track above 10°
elevation, and restricting to directions the hard horizon calls fully clear of
the skyline:

| array | mean graded T where the horizon says "lit" |
| ----- | -----------------------------------------: |
| east  |                                      0.881 |
| west  |                                      0.872 |
| north |                                      0.823 |
| south |                                      0.853 |

So in directions with nothing in the way, the graded map still removes 12–18% of
the beam. And it is not a uniform offset that a gain could soak up: taking sky at
least 20° above the skyline, 26–33% of cells sit below T = 0.95 and 14–20% below
T = 0.9, with minima around 0.52 in wide-open sky.

Renormalising does not fix it. The 98th percentile of transmittance along the sun
track is already 0.99–1.00 for every array, so the field does reach unity at its
brightest — the attenuation is spread across direction, not applied as a scale.
Position-dependent error is exactly what a scalar cannot absorb.

### What probably causes it

The classifier fit reports `edge_width_deg = 10.0` for all four arrays, sitting
exactly on the optimiser's upper bound. A parameter pinned at its bound is a
parameter the fit wanted to push further, and a 10° edge smears each shadow
boundary well into open sky. That is consistent with both the magnitude and the
spatial pattern above.

### What this does not mean

It does not mean the shading work was wrong — shading is the single largest
accuracy win in the whole system, and it came from that work. It means the
*classifier* output specifically has a soft-boundary artefact that costs more
than the extra fidelity buys, on this evaluation.

It is also not a clean contradiction of the original validation. That comparison
used a clear-sky irradiance model on the clearest third of holdout days, and
found grading ahead by 1–5%. This one uses archived forecasts and satellite
observations across all weather. Different driver, different population; the
honest summary is that the two methods are within a few percent of each other and
the sign depends on how you ask.

### Suggested follow-up

- Re-derive the classifier with a smaller upper bound on `edge_width_deg`, or with
    the edge width free per azimuth rather than shared.
- Try the empirical field (`transmittance_empirical.npz`) gated on sample count,
    falling back to the classifier where counts are thin — the handoff's own
    Option B, which this replay has not yet tested.
- Until then, either input is defensible. Soular reads both and does not prefer
    one in code.

## Daily energy

Day-ahead daily energy, scored against the inverter's own local-day counters,
using for each day the forecast issued most recently before that day began:

| variant                             | median APE | mean APE |
| ----------------------------------- | ---------: | -------: |
| no shading                          |      35.3% |    47.2% |
| graded shading                      |      20.8% |    42.2% |
| hard horizon                        |      21.1% |    40.7% |
| incumbent emulation                 |      21.2% |    41.3% |
| graded shading, observed irradiance |      17.7% |    36.7% |
| hard horizon, observed irradiance   |      13.1% |    33.5% |

Shading cuts the typical day's energy error roughly in half, 35% to 21%. The
mean stays high everywhere because a two kilowatt-hour miss on a dark winter day
is a 100% error; the median is the number that describes a normal day.

For scale, the original analysis reported site daily energy error of 19.4%
falling to 15.4% with shading, on clear days only. The 13.1% median here is
across all weather with observed irradiance, which is consistent.

## Where the remaining error is

| driver                      | site RMSE | as % of capacity |
| --------------------------- | --------: | ---------------: |
| forecast irradiance, shaded |   2,163 W |             7.9% |
| observed irradiance, shaded |   1,585 W |             5.8% |

Replacing forecast irradiance with satellite observation removes about 27% of the
RMSE. That is the ceiling on what better weather input can buy, and it is far
larger than anything left in the optics. It is the reason the next phases are
satellite nowcasting, ensemble spread and online bias correction rather than more
physics.

The residual 5.8% under observed irradiance is the floor set by everything else:
sub-pixel cloud, the 10-minute satellite cadence against five-minute production,
module-level effects the model does not represent, and the shading map's own
±25% absolute accuracy.

## Caveats

- **Lead time is conservative.** The archive resolves runs to whole days and the
    harness rounds up, so a forecast for one hour ahead is scored using a run up to
    a day old. In production the freshest run is usually a few hours old, so the
    0–6h column is a floor rather than an estimate.
- **Ensemble spread could not be backtested at all.** Open-Meteo's ensemble
    endpoint accepts historical date ranges and returns correctly shaped responses
    full of nulls: it serves the current run, not an archive. Quantile calibration
    will have to be measured online or accumulated forward.
- **Daily energy percentage errors are inflated by dark days.** A two kilowatt-hour
    miss on a heavily overcast winter day is a 100% error. The median is reported
    alongside the mean for that reason.
- **Curtailed samples are excluded**, since a full battery is lost production no
    forecast can predict, but the exclusion rule here is state-of-charge only — the
    flat-top and export-limit rules land with the online learning work.
