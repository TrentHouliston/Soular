# Configuring Soular

## Installing

Soular is not yet in the HACS default list. Add it as a custom repository:

1. HACS → three dots → **Custom repositories**
2. Repository `https://github.com/TrentHouliston/soular`, type **Integration**
3. Install, then restart Home Assistant
4. **Settings → Devices & services → Add integration → Soular**

Soular pulls in pvlib, which brings pandas, SciPy and h5py — roughly 150 MB
installed and 80–120 MB of resident memory. On constrained hardware that is a
real cost; the README explains what it buys and what the lighter alternative is.

## Setting up a site

The site holds everything shared between roof planes.

| Field                       | Meaning                                                                                                                                                                                                                                                                        |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Name**                    | Names the site device and its entities.                                                                                                                                                                                                                                        |
| **Latitude / Longitude**    | Decimal degrees, north and east positive. Defaults to Home Assistant's own location.                                                                                                                                                                                           |
| **Elevation**               | Metres above sea level. Only weakly affects the clear-sky model — being out by fifty metres does not matter.                                                                                                                                                                   |
| **Ground surface**          | Sets the reflectance used for the ground-reflected component. Grass, the default, is 0.20.                                                                                                                                                                                     |
| **Inverter AC limit**       | Optional, and **leave it empty unless your inverter actually clips.** A limit set too low truncates the forecast at exactly the sunniest part of the day, and because the learned correction will then try to compensate, the damage spreads to hours that were never clipped. |
| **Battery state of charge** | Optional. A full battery curtails the array at midday. Telling Soular about it keeps those samples out of the calibration.                                                                                                                                                     |

You can come back to any of these later: **Devices & services → Soular →
Configure**. The one thing that cannot change is the location. A site that moves
is a different sky, and the weeks of learned correction describing this roof
under these trees would silently follow it to the new one.

## Adding arrays

One array per roof plane. Panels facing different directions, or at different
tilts, need separate arrays — that is what lets the model account for east
waking earlier than west, and for one plane being shaded while another is not.

| Field                       | Meaning                                                                                                                                    |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Name**                    | Names this array's device and entities.                                                                                                    |
| **Azimuth**                 | Compass degrees the panels face: 0 north, 90 east, 180 south, 270 west.                                                                    |
| **Tilt**                    | Degrees from horizontal. 0 is flat, 90 is vertical.                                                                                        |
| **DC capacity**             | Total nameplate rating of the panels on this plane, in watts.                                                                              |
| **Temperature coefficient** | From the module datasheet, typically about −0.35 %/°C.                                                                                     |
| **DC losses**               | Soiling, wiring, mismatch and connections combined. The 14% default suits most systems, and the learned correction re-estimates it anyway. |
| **Measured power**          | Optional but strongly recommended — see below.                                                                                             |
| **Shading file**            | Optional. See below.                                                                                                                       |

### Measured power

Point this at the sensor reporting this array's actual output. It does two
things, and both are worth more than any physics option on this page.

It **calibrates the forecast.** Soular fits a small recursive regression against
your measured output and applies the result as a multiplicative correction. On
the site this was built against, that correction settles around 0.94 — meaning
the uncalibrated physics over-predicts by about 6%, which is typical. Because
the fit forgets exponentially over about thirty days of daylight, it tracks
soiling, pruning and panel degradation rather than freezing a number.

It also **feeds the nowcast.** A working array is a well-sited pyranometer you
already own, with about a minute of latency, and its reading of how bright it is
right now beats any weather model at that horizon.

Without it the array runs on physics alone. That still works — it is what every
other solar forecast integration does — but the first thing to check when the
numbers look consistently high is whether this field is set.

### Shading files

Put the file in `config/soular/` and give the filename here. Three formats are
accepted:

- **`.npz`** — a graded transmittance grid over azimuth and elevation, containing
    an entry named after this array. This is the good one.
- **`.csv`** — the same thing in long form.
- **`.txt`** — a two-column horizon file, azimuth and elevation. Converted to a
    grid with a 2° soft edge, because a hard step aliases badly against a
    five-minute sun track.

A horizon file says "blocked" or "not blocked". Real trees are porous and shade
partially across a band of elevations, and the difference is not small: on the
site Soular was built against, a measured graded map cuts site RMSE by around
15% against no shading at all, and by more than 20% on the two worst-affected
arrays. If you only have a horizon polyline, use it — it is much better than
nothing — but a graded map is where the accuracy is.

The file is parsed when you submit the form, so a wrong filename or an
unreadable file is reported there rather than becoming a silently unshaded array.

## What you get

On the site device:

| Entity                                                               |                                                                                        |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **Estimated power production - now**                                 | The current forecast, and the entity carrying the full 48-hour series as an attribute. |
| **Estimated power production - now (P10 / P90)**                     | The confidence band on that number.                                                    |
| **Estimated energy production - today / remaining today / tomorrow** | Daily totals in kWh.                                                                   |
| **Estimated energy production - today / tomorrow (P10 / P90)**       | The band on those totals.                                                              |
| **Estimated energy production - this hour / next hour**              | Disabled by default.                                                                   |
| **Highest power peak time - today / tomorrow**                       | When the peak lands.                                                                   |

Each array gets the same power and energy sensors for its own plane, plus
diagnostics that start disabled: plane-of-array irradiance, cell temperature,
shading transmittance, the learned correction and how many samples are behind it.

Site diagnostics, also disabled by default, report the nowcast contribution and
when each source last updated.

### The confidence band

P10 and P90 come from the ECMWF ensemble — fifty-one runs of the weather model
that disagree with each other, which is the only honest measure of how uncertain
tomorrow is. When the ensemble is unavailable these report **unknown** rather
than falling back to the median, because a band that quietly collapses onto the
point forecast reads as certainty.

The day-ahead energy bounds are not the pointwise power bounds added up. Summing
pointwise P90s would assume every hour of the day hits its ninetieth percentile
at once; the totals here integrate whole plausible days and take quantiles of
those, which is a factor of about two narrower and is the number that means what
it says.

## Wiring it up

**Energy dashboard.** Settings → Dashboards → Energy → Solar panels → add your
production sensor, and pick Soular under "forecast production". Soular supplies
the site's hourly P50.

**haeo.** Point a solar element at
`sensor.<site>_estimated_power_production_now`. Soular emits haeo's own richest
forecast format — the series with units and interpolation mode — starting at
now and spanning exactly 48 hours, which is what haeo's forecast cycle expects.
If you would rather your battery plan hedged, point it at the P10 sensor instead;
that is a risk posture, not a correction.

## How often it updates

| Source         | Interval                | What is lost if it fails                                                                           |
| -------------- | ----------------------- | -------------------------------------------------------------------------------------------------- |
| Weather model  | 30 minutes              | Everything. This is the backbone, and a failure here marks the integration unavailable.            |
| Satellite      | 10 minutes              | The nowcast. The next couple of hours revert to the weather model, which is what they were before. |
| Ensemble       | 1 hour                  | The P10/P90 band, which reports unknown.                                                           |
| Your own array | Every coordinator cycle | A minute-latency observation, and the calibration signal.                                          |

The two optional sources fail independently and neither can take the other down.
**Settings → System → Repairs → System information** reports which of the three
endpoints is reachable — worth checking before concluding the forecast is wrong,
because a forecast still being produced says nothing about whether the satellite
is up.

## When it looks wrong

**Consistently high, especially in the first week.** Expected. The correction
ramps in over roughly its first 500 samples so a fresh install is pure physics,
and physics over-predicts. Check the **Learned correction** diagnostic on each
array: it starts at 100% and should settle somewhere near 90–95%.

**The correction is pinned at 75% or 130%.** Those are the clamps, and hitting
one means something structural is wrong rather than that the array is that
inefficient — usually a DC capacity that is out by a lot, or an inverter limit
truncating the forecast.

**Zero all day.** Check the inverter AC limit. Setting it below what the array
actually produces is the one configuration error that can flatten the whole
forecast.

**Two identical roof planes producing differently.** If two arrays share azimuth
and tilt and their learned corrections diverge in opposite directions, their
shading files are probably swapped.

**Learning never accumulates samples.** The curtailment mask is doing its job too
well, or the measured-power sensor is not set. A battery that is full most of the
day genuinely does leave few usable samples.

For anything else, **Devices & services → Soular → three dots → Download
diagnostics** dumps the geometry, the health of each source and a summary of the
current forecast, with coordinates redacted.

## Removing it

Delete the config entry. Devices and entities go with it, including one device
per array. Shading files in `config/soular/` are yours and are left alone.

The learner's state lives in `.storage/soular.learning` and is removed with the
entry, so reinstalling starts from pure physics again and spends another week
learning what it already knew. If you are reinstalling deliberately, copying that
file aside first will save you the wait.
