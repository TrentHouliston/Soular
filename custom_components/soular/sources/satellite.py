"""Observed irradiance from geostationary satellite.

Not a forecast. Open-Meteo's satellite endpoint is an observation archive: for
Australia it serves JMA/JAXA Himawari at ten-minute cadence and about half an
hour of latency. The nowcast is built from it in :mod:`core.blend`, by
persistence.

Two things about it that are easy to get wrong:

* The resolution parameter is ``temporal_resolution=native``. Without it the
  response is resampled to hourly, discarding exactly the sub-hourly variability
  the nowcast exists to capture.
* The model identifier the API accepts is ``jma_jaxa_himawari``, which is not the
  name the documentation lists.
"""

from dataclasses import dataclass
import datetime as dt
import logging
from typing import Any

from aiohttp import ClientError, ClientSession, ClientTimeout
import numpy as np

_LOGGER = logging.getLogger(__name__)

ARCHIVE_URL = "https://satellite-api.open-meteo.com/v1/archive"

# Regional coverage differs by satellite; seamless picks whichever serves the
# site. Himawari covers Australia, Asia, India and New Zealand.
DEFAULT_MODEL = "satellite_radiation_seamless"

REQUEST_TIMEOUT = ClientTimeout(total=30)

# Enough history to survive a gap and still have something to persist from,
# without asking for a day of data every ten minutes.
LOOKBACK_HOURS = 6


class SatelliteError(Exception):
    """The satellite archive could not be reached, or returned nothing usable."""


@dataclass(frozen=True, slots=True)
class SatelliteObservations:
    """Observed global horizontal irradiance, most recent last."""

    times: np.ndarray
    ghi: np.ndarray

    def __len__(self) -> int:
        """Return the number of observations."""
        return int(self.times.size)

    def latest(self) -> tuple[np.datetime64, float] | None:
        """Return the most recent non-null observation, or nothing."""
        finite = np.isfinite(self.ghi)
        if not finite.any():
            return None
        index = int(np.flatnonzero(finite)[-1])
        return self.times[index], float(self.ghi[index])


async def fetch(
    session: ClientSession,
    latitude: float,
    longitude: float,
    *,
    now: dt.datetime,
    model: str = DEFAULT_MODEL,
) -> SatelliteObservations:
    """Fetch recent observed irradiance for a site."""
    end = now.date()
    start = (now - dt.timedelta(hours=LOOKBACK_HOURS)).date()
    params = {
        "latitude": f"{latitude:.6f}",
        "longitude": f"{longitude:.6f}",
        "hourly": "shortwave_radiation",
        "models": model,
        "temporal_resolution": "native",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": "UTC",
    }

    try:
        async with session.get(ARCHIVE_URL, params=params, timeout=REQUEST_TIMEOUT) as response:
            if response.status != 200:  # noqa: PLR2004
                body = (await response.text())[:200]
                msg = f"satellite archive returned HTTP {response.status}: {body}"
                raise SatelliteError(msg)
            payload: dict[str, Any] = await response.json()
    except ClientError as err:
        msg = f"could not reach the satellite archive: {err}"
        raise SatelliteError(msg) from err
    except TimeoutError as err:
        msg = "the satellite archive timed out"
        raise SatelliteError(msg) from err

    if payload.get("error"):
        msg = f"satellite archive rejected the request: {payload.get('reason', 'no reason given')}"
        raise SatelliteError(msg)

    return parse(payload)


def parse(payload: dict[str, Any]) -> SatelliteObservations:
    """Convert a satellite response into arrays."""
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        msg = f"satellite response carried no series; keys were {sorted(payload)}"
        raise SatelliteError(msg)

    values = hourly.get("shortwave_radiation")
    if values is None:
        msg = "satellite response is missing shortwave_radiation"
        raise SatelliteError(msg)

    times = np.array(hourly["time"], dtype="datetime64[s]")
    # Nulls are ordinary here: the archive is published in arrears, so the most
    # recent slots are routinely empty until the next pass lands.
    ghi = np.array([np.nan if value is None else float(value) for value in values], dtype=np.float64)
    return SatelliteObservations(times=times, ghi=ghi)
