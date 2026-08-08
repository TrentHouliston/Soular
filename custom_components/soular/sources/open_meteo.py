"""The deterministic numerical weather prediction source.

Requests horizontal irradiance components rather than Open-Meteo's
``global_tilted_irradiance``, because the plane-of-array transposition happens
here, with a real anisotropic sky model and the site's own ground reflectance.

Radiation is requested hourly and resampled in the core rather than asked for at
15-minute resolution. Open-Meteo interpolates sub-hourly values itself for most
models, and our resampling works in clear-sky index, which does not shave the
diurnal peak the way interpolating irradiance directly does. It is also the exact
path the offline backtest measured, so the shipped behaviour is the measured one.
"""

from dataclasses import dataclass
import logging
from typing import Any

from aiohttp import ClientError, ClientSession, ClientTimeout
import numpy as np

_LOGGER = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Everything the forward model consumes, requested together so one response
# carries a self-consistent set rather than a mixture of runs.
HOURLY_VARIABLES = (
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "temperature_2m",
    "wind_speed_10m",
)

REQUEST_TIMEOUT = ClientTimeout(total=30)
# Two days of forecast plus a day of slack, so the 48-hour horizon is always
# covered even when a run is late.
FORECAST_DAYS = 3


class OpenMeteoError(Exception):
    """Open-Meteo could not be reached, or returned something unusable."""


@dataclass(frozen=True, slots=True)
class HourlyWeather:
    """Hourly forecast series, as interval means labelled with interval ends.

    Open-Meteo's radiation variables are means over the *preceding* hour. Keeping
    that in the type name is deliberate: treating them as instantaneous shifts the
    whole series half an hour, which is a bias that does not cancel between
    morning and evening.
    """

    times: np.ndarray
    ghi: np.ndarray
    direct_horizontal: np.ndarray
    diffuse: np.ndarray
    temperature: np.ndarray
    wind_speed_10m: np.ndarray

    def __len__(self) -> int:
        """Return the number of hourly samples."""
        return int(self.times.size)


async def fetch(
    session: ClientSession,
    latitude: float,
    longitude: float,
    *,
    model: str | None = None,
) -> HourlyWeather:
    """Fetch the hourly forecast for a site."""
    params: dict[str, str] = {
        "latitude": f"{latitude:.6f}",
        "longitude": f"{longitude:.6f}",
        "hourly": ",".join(HOURLY_VARIABLES),
        "forecast_days": str(FORECAST_DAYS),
        "timezone": "UTC",
        # The default is km/h. Feeding that to a convective cooling model would
        # over-cool the array by a factor of 3.6.
        "wind_speed_unit": "ms",
    }
    if model:
        params["models"] = model

    try:
        async with session.get(FORECAST_URL, params=params, timeout=REQUEST_TIMEOUT) as response:
            if response.status != 200:  # noqa: PLR2004
                body = (await response.text())[:200]
                msg = f"Open-Meteo returned HTTP {response.status}: {body}"
                raise OpenMeteoError(msg)
            payload: dict[str, Any] = await response.json()
    except ClientError as err:
        msg = f"could not reach Open-Meteo: {err}"
        raise OpenMeteoError(msg) from err
    except TimeoutError as err:
        msg = "Open-Meteo timed out"
        raise OpenMeteoError(msg) from err

    if payload.get("error"):
        msg = f"Open-Meteo rejected the request: {payload.get('reason', 'no reason given')}"
        raise OpenMeteoError(msg)

    return parse(payload)


def parse(payload: dict[str, Any]) -> HourlyWeather:
    """Convert a forecast response into arrays, rejecting anything incomplete."""
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        msg = f"Open-Meteo response carried no hourly series; keys were {sorted(payload)}"
        raise OpenMeteoError(msg)

    times = np.array(hourly["time"], dtype="datetime64[s]")
    if times.size == 0:
        msg = "Open-Meteo returned an empty forecast"
        raise OpenMeteoError(msg)

    def column(name: str) -> np.ndarray:
        values = hourly.get(name)
        if values is None:
            msg = f"Open-Meteo response is missing {name}"
            raise OpenMeteoError(msg)
        # Nulls appear at the edges of a model's coverage. Zero is right for
        # radiation and wrong for temperature, so callers get NaN and decide.
        return np.array([np.nan if value is None else float(value) for value in values], dtype=np.float64)

    return HourlyWeather(
        times=times,
        ghi=column("shortwave_radiation"),
        direct_horizontal=column("direct_radiation"),
        diffuse=column("diffuse_radiation"),
        temperature=column("temperature_2m"),
        wind_speed_10m=column("wind_speed_10m"),
    )
