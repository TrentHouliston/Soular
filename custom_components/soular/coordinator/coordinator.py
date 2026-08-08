"""Fetch weather, run the model, and hold the result.

One timer drives three cadences. The weather model is refetched every half hour,
because Open-Meteo publishes new runs hourly at best and polling faster only
burns quota. Satellite observations are refetched every ten minutes, which is
their publication cadence. The forecast itself is recomputed every five minutes
regardless, because the sun keeps moving even when the weather does not -- and on
a shaded site the sun's position is most of what changes minute to minute.

Losing the satellite is never fatal. It costs the nowcast and leaves the weather
model, which is what the forecast was before the nowcast existed.
"""

from datetime import datetime, timedelta
import logging

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util
import numpy as np

from custom_components.soular.const import DOMAIN, RECOMPUTE_INTERVAL, SATELLITE_INTERVAL, UPDATE_INTERVAL
from custom_components.soular.core.blend import (
    Observation,
    apply_to_weather,
    observation_from_irradiance,
    satellite_observation,
)
from custom_components.soular.core.clearsky import clear_sky
from custom_components.soular.core.geometry import solar_geometry
from custom_components.soular.core.nwp import weather_from_hourly
from custom_components.soular.core.pipeline import ForecastResult, SystemSpec, WeatherSeries, build_time_grid, forecast
from custom_components.soular.sources import satellite as satellite_source
from custom_components.soular.sources.open_meteo import HourlyWeather, OpenMeteoError, fetch

_LOGGER = logging.getLogger(__name__)

# Past this, a cached forecast is describing weather that has moved on. Entities
# go unavailable rather than quietly serving a stale number, because a battery
# optimiser acting on yesterday's sky is worse than one that knows it is blind.
MAX_WEATHER_AGE = timedelta(hours=6)

# How many recent satellite observations to persist from. They are ten minutes
# apart and highly correlated, so a handful adds precision and more adds nothing.
SATELLITE_OBSERVATION_COUNT = 3


class SoularCoordinator(DataUpdateCoordinator[ForecastResult]):
    """Keeps a current forecast for one site."""

    def __init__(self, hass: HomeAssistant, name: str, system: SystemSpec) -> None:
        """Set up the coordinator for a configured system."""
        super().__init__(hass, _LOGGER, name=f"{DOMAIN} {name}", update_interval=RECOMPUTE_INTERVAL)
        self.system = system
        self._weather: HourlyWeather | None = None
        self._weather_at: datetime | None = None
        self._weather_error: str | None = None
        self._satellite: satellite_source.SatelliteObservations | None = None
        self._satellite_at: datetime | None = None
        self._satellite_error: str | None = None
        # Share of the near-term forecast that came from observation rather than
        # from the weather model. Surfaced so "is this a nowcast?" is answerable.
        self.observed_share: float = 0.0

    @property
    def weather_fetched_at(self) -> datetime | None:
        """When the underlying forecast was last successfully retrieved."""
        return self._weather_at

    @property
    def weather_error(self) -> str | None:
        """The last weather fetch failure, if the most recent attempt failed."""
        return self._weather_error

    @property
    def satellite_fetched_at(self) -> datetime | None:
        """When satellite observations were last successfully retrieved."""
        return self._satellite_at

    @property
    def satellite_error(self) -> str | None:
        """The last satellite fetch failure, if the most recent attempt failed."""
        return self._satellite_error

    async def _async_update_data(self) -> ForecastResult:
        """Refresh weather if it is due, then recompute the forecast."""
        await self._refresh_weather()
        # Never fatal. Losing the satellite costs the nowcast and leaves the
        # weather model, which is what the forecast was before it existed.
        await self._refresh_satellite()

        weather = self._weather
        if weather is None:
            msg = self._weather_error or "no weather has been retrieved yet"
            raise UpdateFailed(msg)

        age = dt_util.utcnow() - (self._weather_at or dt_util.utcnow())
        if age > MAX_WEATHER_AGE:
            msg = (
                f"weather is {age.total_seconds() / 3600:.1f} hours old; {self._weather_error or 'source unavailable'}"
            )
            raise UpdateFailed(msg)

        # numpy and pvlib over a few hundred timesteps: milliseconds, but it is
        # still CPU work and belongs off the event loop.
        return await self.hass.async_add_executor_job(self._compute, weather)

    async def _refresh_weather(self) -> None:
        """Fetch new weather when the cached copy is due for replacement."""
        now = dt_util.utcnow()
        if self._weather is not None and self._weather_at is not None and now - self._weather_at < UPDATE_INTERVAL:
            return

        session = async_get_clientsession(self.hass)
        try:
            self._weather = await fetch(session, self.system.site.latitude, self.system.site.longitude)
        except OpenMeteoError as err:
            # Keep serving the cached forecast. A dropped request is common and a
            # half-hour-old sky is far better than no forecast at all; the age
            # check above is what eventually gives up.
            self._weather_error = str(err)
            if self._weather is None:
                raise UpdateFailed(str(err)) from err
            _LOGGER.debug("Keeping cached weather: %s", err)
        else:
            self._weather_at = now
            self._weather_error = None

    async def _refresh_satellite(self) -> None:
        """Fetch recent observed irradiance when the cached copy is due."""
        now = dt_util.utcnow()
        if self._satellite_at is not None and now - self._satellite_at < SATELLITE_INTERVAL:
            return

        session = async_get_clientsession(self.hass)
        try:
            self._satellite = await satellite_source.fetch(
                session, self.system.site.latitude, self.system.site.longitude, now=now
            )
        except satellite_source.SatelliteError as err:
            self._satellite_error = str(err)
            _LOGGER.debug("No satellite nowcast this cycle: %s", err)
        else:
            self._satellite_at = now
            self._satellite_error = None

    def _observations(self, times: np.ndarray, clearsky_ghi: np.ndarray) -> list[Observation]:
        """Build clear-sky-index observations from the satellite record."""
        del times, clearsky_ghi
        if self._satellite is None or len(self._satellite) == 0:
            return []

        finite = np.flatnonzero(np.isfinite(self._satellite.ghi))
        if finite.size == 0:
            return []
        recent = finite[-SATELLITE_OBSERVATION_COUNT:]
        stamps = self._satellite.times[recent]

        geometry = solar_geometry(stamps, self.system.site)
        observed_clearsky = clear_sky(stamps, self.system.site, geometry).ghi

        observations: list[Observation] = []
        for index, stamp, clear in zip(recent, stamps, observed_clearsky, strict=True):
            k = observation_from_irradiance(stamp, float(self._satellite.ghi[index]), float(clear))
            if k is not None:
                observations.append(satellite_observation(stamp, k))
        return observations

    def _nowcast(self, series: WeatherSeries, times: np.ndarray) -> WeatherSeries:
        """Blend recent observations into the forecast."""
        geometry = solar_geometry(times, self.system.site)
        clearsky = clear_sky(times, self.system.site, geometry).ghi

        observations = self._observations(times, clearsky)
        if not observations:
            self.observed_share = 0.0
            return series

        blended, share = apply_to_weather(series, times, clearsky, geometry, observations)
        self.observed_share = float(share[0]) if share.size else 0.0
        return blended

    def _compute(self, weather: HourlyWeather) -> ForecastResult:
        """Run the model. Blocking; called in an executor."""
        start = np.datetime64(int(dt_util.utcnow().timestamp() // 300 * 300), "s")
        grid = build_time_grid(start)
        series = weather_from_hourly(
            hourly_times=weather.times,
            grid=grid,
            site=self.system.site,
            ghi=weather.ghi,
            temperature=weather.temperature,
            wind_speed_10m=weather.wind_speed_10m,
            direct_horizontal=weather.direct_horizontal,
            diffuse=weather.diffuse,
        )
        series = self._nowcast(series, grid.times)
        return forecast(self.system, grid, series)
