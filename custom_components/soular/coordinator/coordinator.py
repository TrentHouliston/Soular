"""Fetch weather, run the model, and hold the result.

One timer drives two cadences. The weather is refetched every half hour, because
Open-Meteo publishes new runs hourly at best and polling faster only burns quota.
The forecast itself is recomputed every five minutes regardless, because the sun
keeps moving even when the weather does not -- and on a shaded site the sun's
position is most of what changes minute to minute.
"""

from datetime import datetime, timedelta
import logging

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util
import numpy as np

from custom_components.soular.const import DOMAIN, RECOMPUTE_INTERVAL, UPDATE_INTERVAL
from custom_components.soular.core.nwp import weather_from_hourly
from custom_components.soular.core.pipeline import ForecastResult, SystemSpec, build_time_grid, forecast
from custom_components.soular.sources.open_meteo import HourlyWeather, OpenMeteoError, fetch

_LOGGER = logging.getLogger(__name__)

# Past this, a cached forecast is describing weather that has moved on. Entities
# go unavailable rather than quietly serving a stale number, because a battery
# optimiser acting on yesterday's sky is worse than one that knows it is blind.
MAX_WEATHER_AGE = timedelta(hours=6)


class SoularCoordinator(DataUpdateCoordinator[ForecastResult]):
    """Keeps a current forecast for one site."""

    def __init__(self, hass: HomeAssistant, name: str, system: SystemSpec) -> None:
        """Set up the coordinator for a configured system."""
        super().__init__(hass, _LOGGER, name=f"{DOMAIN} {name}", update_interval=RECOMPUTE_INTERVAL)
        self.system = system
        self._weather: HourlyWeather | None = None
        self._weather_at: datetime | None = None
        self._weather_error: str | None = None

    @property
    def weather_fetched_at(self) -> datetime | None:
        """When the underlying forecast was last successfully retrieved."""
        return self._weather_at

    @property
    def weather_error(self) -> str | None:
        """The last weather fetch failure, if the most recent attempt failed."""
        return self._weather_error

    async def _async_update_data(self) -> ForecastResult:
        """Refresh weather if it is due, then recompute the forecast."""
        await self._refresh_weather()

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
        return forecast(self.system, grid, series)
