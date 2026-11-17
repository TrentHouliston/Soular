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
| **Name**                    | Prefixes every array's device and entity names.                                                                                                                                                                                                                                |
| **Latitude / Longitude**    | Decimal degrees, north and east positive. Defaults to Home Assistant's own location.                                                                                                                                                                                           |
| **Elevation**               | Metres above sea level. Only weakly affects the clear-sky model — being out by fifty metres does not matter.                                                                                                                                                                   |
| **Ground surface**          | Sets the reflectance used for the ground-reflected component. Grass, the default, is 0.20.                                                                                                                                                                                     |
| **Inverter AC limit**       | Optional, and **leave it empty unless your inverter actually clips.** A limit set too low truncates the forecast at exactly the sunniest part of the day, and because the weather correction will then try to compensate, the damage spreads to hours that were never clipped. |
| **Battery state of charge** | Optional. A full battery curtails the array at midday. Telling Soular about it keeps those samples out of the calibration.                                                                                                                                                     |

You can come back to any of these later: **Devices & services → Soular →
Configure**. The one thing that cannot change is the location. A site that moves
is a different sky, and the weeks of weather correction describing this sky
would silently follow it to the new one.

## Adding arrays

One array per roof plane. Panels facing different directions, or at different
tilts, need separate arrays — that is what lets the model account for east
waking earlier than west, and for one plane being shaded while another is not.

| Field                       | Meaning                                                                                                                                |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Name**                    | Names this array's device and entities.                                                                                                |
| **Azimuth**                 | Compass degrees the panels face: 0 north, 90 east, 180 south, 270 west.                                                                |
| **Tilt**                    | Degrees from horizontal. 0 is flat, 90 is vertical.                                                                                    |
| **DC capacity**             | Total nameplate rating of the panels on this plane, in watts.                                                                          |
| **Temperature coefficient** | From the module datasheet, typically about −0.35 %/°C.                                                                                 |
| **DC losses**               | Soiling, wiring, mismatch and connections combined. The 14% default suits most systems; a graded shading map measures the real figure. |
| **Measured power**          | Optional but strongly recommended — see below.                                                                                         |
| **Shading file**            | Optional. See below.                                                                                                                   |

### Measured power

Point this at the sensor reporting this array's actual output. It is worth more
than any physics option on this page, because it lets Soular **correct the
weather forecast.**

The forecast's weather is not unbiased. At the site this was built against,
Open-Meteo promises more sun than arrives, and by a margin that depends on how
much it promised: a clear forecast is usually right, a broken-cloud one usually
overshoots. Soular keeps a record, over the last 60 days, of how much of the
forecast energy actually arrived at each level of forecast cloudiness, and
scales each future instant by the ratio for its level. It is one correction for
the site, measured on the arrays that have a sensor, because cloud covers the
whole roof at once.

It moves slowly on purpose. Each level is held at "no correction" until it has
seen about two good days' worth of its own energy, so one cloudy week cannot
teach it that every day is cloudy. Instants when the battery is full are left
out, because the inverter is throttling the panels then rather than the sky
under-delivering.

You do not have to wait for it either. On a fresh install Soular replays the
last week of these sensors' history from the recorder against the weather for
the same days. That needs recorder statistics on the sensors (a `state_class` of
`measurement`, which inverter power sensors almost always have), and happens
once, not on every restart.

How efficient the panels themselves are is a different question, and it is not
learned here: see the shading file below. Without a power sensor the forecast
runs on the weather as forecast, which is what every other solar forecast
integration does.

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

A graded map built by Soular's own tool also carries each array's **panel
efficiency**: what the array delivered under a satellite-verified clear sky,
over what the model predicted. Soiling, module tolerance, wiring and the
clear-sky model's own bias all land in it. It is a property of the hardware, so
it is measured once, from a year of record, and applied to every forecast
rather than learned live, where a week of weather the forecast got wrong would
drag it about. Rebuild the map after cleaning the panels or when something new
casts a shadow.

To make a graded map from your own array's measured output, see
[shading-maps.md](shading-maps.md). It needs about a year of five-minute
per-array power, so the sun has visited every part of the sky it ever will.

## What you get

Every sensor belongs to an array: there is no site-wide total. The energy
dashboard and the recorder both want one entity per array, and haeo adds up
several sensors itself, so a total would only repeat what the arrays say.

On each array's device:

| Entity                                                                 |                                                                                                |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Estimated power production - now**                                   | The current forecast, and the full eight-day series as an attribute.                           |
| **Estimated power production - now (P10) / (P90)**                     | A poor and a good version of the forecast, each with its own eight-day series. See below.      |
| **Estimated energy production - today / remaining today / tomorrow**   | Daily totals in kWh, with the day's P10 and P90 as attributes.                                 |
| **Estimated energy production - day after tomorrow … 7 days from now** | The week ahead, one daily total each, as `open_meteo_solar_forecast` has. Disabled by default. |
| **Estimated energy production - this hour / next hour**                | Disabled by default.                                                                           |
| **Highest power peak time - today / tomorrow**                         | When the peak lands. Disabled by default.                                                      |

The series is five-minute steps for the first six hours, quarter-hours out to two
days, and half-hours beyond that. It is kept out of the recorder, which drops
every attribute of a row over 16 kB; the sensors' states are recorded as usual,
and the power sensors keep long-term statistics you can set against the
inverter's own.

Diagnostics, disabled by default: plane-of-array irradiance, cell temperature,
shading transmittance, panel efficiency (from the shading map), the weather
correction applied right now, and when the weather was last fetched.

### The P10/P90 band

Each daily total carries `estimate10` and `estimate90` attributes,
named as Solcast names them. On nine days in ten the day's production should come
in above `estimate10`, and on nine in ten below `estimate90`. `band_calibrated`
says whether the band comes from this site's own record yet.

The band comes from how far this site's actual production has landed from
Soular's forecast, not from a weather ensemble. Open-Meteo keeps no archive of
its ensemble runs, so a band built that way could never be checked. This one can.
In a year-long replay at the reference site, using only results known before each
forecast, 10% of days fell below P10 and 11% above P90, from one to seven days
ahead.

How it works:

- Each midnight Soular keeps the forecast it has just made for the next eight
    days.
- Through each day it compares that forecast with what the arrays actually
    produce. Instants when the battery is full are left out on both sides, because
    the inverter is throttling the panels then.
- The band for each lead is the 10th and 90th percentile of the last 60 days'
    ratios. A lead is pooled with the ones either side of it for a usable sample
    size.

It needs a **Measured power** sensor on at least one array, and a **Battery state
of charge** sensor if there is a battery. Until a lead has 15 results, it uses a
starting band from the reference site, which is about −40% to +35% for tomorrow.

The P10 and P90 power sensors scale each day's forecast by that day's ratio.
Their series is one coherent poor (or good) day, not every interval at its worst
at once, so integrating the P10 series over a day gives that day's
`estimate10`. The band is measured on the arrays' output together and applied to
each array alike, so the arrays' P10s add up to the site's: one poor day for the
whole roof, which is what weather does.

## Wiring it up

**Energy dashboard.** Settings → Dashboards → Energy → Solar panels → add each
array's production sensor, and pick Soular under "forecast production". Soular
supplies the whole site's hourly forecast for its entry.

**haeo.** Point a solar element at every array's
`sensor.<site>_<array>_estimated_power_production_now`; haeo adds them up. Soular
emits haeo's own richest forecast format — the series with units and
interpolation mode — starting at now and spanning a whole number of days
(eight), which is what haeo's forecast cycle expects.

To plan against a poor day instead, use the arrays'
`..._estimated_power_production_now_p10` sensors, which have the same format.

## How often it updates

| Source         | Interval        | What is lost if it fails                                                                                                                     |
| -------------- | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Weather model  | 30 minutes      | Nothing at first: the last forecast keeps being served. After six hours without a fresh one the sensors go unavailable rather than go stale. |
| Your own array | Every 5 minutes | The weather correction's record. The correction holds its last value.                                                                        |

The forecast itself is recomputed every five minutes regardless, because the sun
keeps moving even when the weather does not. **Settings → System → Repairs →
System information** reports whether Open-Meteo is reachable — worth checking
before concluding the forecast is wrong, because a cached forecast keeps being
served for hours after the source has gone.

## When it looks wrong

**Consistently high or low on clear days.** That is the panels, not the
weather: check the **Panel efficiency** diagnostic. If it is missing, the array's
shading file is a plain horizon or an older map without an efficiency, and the
physics runs uncalibrated; rebuild the map with Soular's tool.

**Consistently high on cloudy days.** That is the weather correction still
learning. Check the **Weather correction** diagnostic, and **Download
diagnostics** for how many days are behind it. If it never moves off 100%, the
measured-power sensors are not set, or the battery is full whenever the sun is
out.

**Zero all day.** Check the inverter AC limit. Setting it below what the array
actually produces is the one configuration error that can flatten the whole
forecast.

**Two identical roof planes producing differently.** If two arrays share azimuth
and tilt and their panel efficiencies sit either side of each other, their
power sensors or shading map entries are probably swapped.

For anything else, **Devices & services → Soular → three dots → Download
diagnostics** dumps the geometry, the weather fetch status and a summary of the
current forecast, with coordinates redacted.

## Removing it

Delete the config entry. Devices and entities go with it, including one device
per array. Shading files in `config/soular/` are yours and are left alone.

The weather correction's record lives in `.storage/soular.learning` and is
removed with the entry. Reinstalling catches it up from the recorder's last week
again, so little is lost.
