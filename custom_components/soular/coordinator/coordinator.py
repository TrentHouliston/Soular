"""Fetch weather, run the model, and hold the result.

One timer drives two cadences. The weather model is refetched every half hour,
because Open-Meteo publishes new runs hourly at best and polling faster only
burns quota. The forecast itself is recomputed every five minutes regardless,
because the sun keeps moving even when the weather does not -- and on a shaded
site the sun's position is most of what changes minute to minute.
"""

from dataclasses import replace
from datetime import datetime, timedelta
import logging

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util
import numpy as np

from custom_components.soular.const import DOMAIN, LEARNING_SAMPLE_INTERVAL, RECOMPUTE_INTERVAL, UPDATE_INTERVAL
from custom_components.soular.coordinator.actuals import (
    Learner,
    forecast_clearness,
    read_percentage,
    read_power,
    samples_from,
)
from custom_components.soular.coordinator.backfill import (
    BACKFILL_DAYS,
    arrays_to_backfill,
    async_read_history,
    backfill_window,
    replay,
)
from custom_components.soular.core.band import BAND_LEADS, BandTracker, Bounds, hourly_snapshot
from custom_components.soular.core.correction import build_features
from custom_components.soular.core.geometry import solar_geometry
from custom_components.soular.core.mask import CURTAILMENT_SOC_PCT
from custom_components.soular.core.nwp import weather_from_hourly
from custom_components.soular.core.pipeline import ArrayForecast, ForecastResult, SystemSpec, build_time_grid, forecast
from custom_components.soular.sources.open_meteo import HourlyWeather, OpenMeteoError, fetch

_LOGGER = logging.getLogger(__name__)

# Past this, a cached forecast is describing weather that has moved on. Entities
# go unavailable rather than quietly serving a stale number, because a battery
# optimiser acting on yesterday's sky is worse than one that knows it is blind.
MAX_WEATHER_AGE = timedelta(hours=6)


class SoularCoordinator(DataUpdateCoordinator[ForecastResult]):
    """Keeps a current forecast for one site."""

    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        system: SystemSpec,
        power_sensors: dict[str, str] | None = None,
        soc_sensor: str | None = None,
        learner: Learner | None = None,
        band: BandTracker | None = None,
    ) -> None:
        """Set up the coordinator for a configured system."""
        super().__init__(hass, _LOGGER, name=f"{DOMAIN} {name}", update_interval=RECOMPUTE_INTERVAL)
        self.system = system
        self.power_sensors = power_sensors or {}
        self.soc_sensor = soc_sensor
        self.learner = learner or Learner()
        self.band = band or BandTracker()
        self._bounds: tuple[Bounds, ...] = self.band.bounds(dt_util.now().date().toordinal())
        self._weather: HourlyWeather | None = None
        self._weather_at: datetime | None = None
        self._weather_error: str | None = None
        self._backfilled = False

    @property
    def weather_fetched_at(self) -> datetime | None:
        """When the underlying forecast was last successfully retrieved."""
        return self._weather_at

    @property
    def band_bounds(self) -> tuple[Bounds, ...]:
        """The P10/P90 band for today and each day after, as ratios to the forecast."""
        return self._bounds

    @property
    def band_samples(self) -> int:
        """How many of this site's own daily ratios are behind tomorrow's band."""
        return self._bounds[1].samples

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

        if not self._backfilled:
            self._backfilled = True
            await self._async_backfill()

        # Read measured output before recomputing, so the correction applied to
        # this forecast reflects everything known up to now and nothing after.
        measured = {name: read_power(self.hass, entity) for name, entity in self.power_sensors.items()}
        soc = read_percentage(self.hass, self.soc_sensor)

        # numpy and pvlib over a few hundred timesteps: milliseconds, but it is
        # still CPU work and belongs off the event loop.
        return await self.hass.async_add_executor_job(self._compute, weather, measured, soc)

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

    async def _async_backfill(self) -> None:
        """Catch arrays with nothing learned up from the recorder, once.

        A head start, never a requirement: any failure leaves the learner to
        sample live state as it would have anyway.
        """
        arrays = arrays_to_backfill(self.learner, self.power_sensors)
        if not arrays:
            return
        try:
            start, end = backfill_window(dt_util.utcnow())
            entities = set(arrays.values()) | ({self.soc_sensor} if self.soc_sensor else set())
            history = await async_read_history(self.hass, entities, start, end)
            if not any(history.get(entity) for entity in arrays.values()):
                return
            weather = await fetch(
                async_get_clientsession(self.hass),
                self.system.site.latitude,
                self.system.site.longitude,
                past_days=BACKFILL_DAYS + 1,
                forecast_days=1,
            )
            soc = history.get(self.soc_sensor) if self.soc_sensor else None
            offered = await self.hass.async_add_executor_job(
                replay, self.learner, self.system, weather, arrays, history, soc
            )
        except Exception:  # a failed head start must not fail setup
            _LOGGER.warning("Could not learn from recorded history; learning from live output only", exc_info=True)
            return
        _LOGGER.info(
            "Learned from %d recorded samples for %s",
            offered,
            ", ".join(sorted(arrays)),
        )

    def _learn(
        self, result: ForecastResult, ghi: np.ndarray, measured: dict[str, float | None], soc: float | None
    ) -> None:
        """Fold the present instant's measured output into the correction."""
        now = dt_util.utcnow()
        if self.learner.last_sample_at is not None and now - self.learner.last_sample_at < LEARNING_SAMPLE_INTERVAL:
            return

        samples = samples_from(result, self.system, ghi, np.array([result.now_index]))
        for name, (sample,) in samples.items():
            value = measured.get(name)
            if value is not None:
                self.learner.observe(name, sample, value, soc)

    def _correct(self, result: ForecastResult, clearness: np.ndarray) -> ForecastResult:
        """Apply the learned correction to every array, and to the site total."""
        if not self.learner.states:
            return result

        geometry = solar_geometry(result.times, self.system.site)
        day_of_year = np.full(result.times.size, float(dt_util.utcnow().timetuple().tm_yday))

        corrected: list[ArrayForecast] = []
        scaled = np.zeros(result.times.size)
        original = np.zeros(result.times.size)
        for entry in result.arrays:
            state = self.learner.states.get(entry.name)
            if state is None or state.samples == 0:
                corrected.append(entry)
                scaled += entry.ac_power_w
                original += entry.ac_power_w
                continue
            factor = state.correction(
                build_features(
                    elevation_deg=geometry.apparent_elevation,
                    clearness=clearness,
                    day_of_year=day_of_year,
                    transmittance=entry.transmittance,
                    temperature_c=entry.cell_temperature,
                )
            )
            corrected.append(
                ArrayForecast(
                    name=entry.name,
                    dc_power_w=entry.dc_power_w * factor,
                    ac_power_w=entry.ac_power_w * factor,
                    poa_global=entry.poa_global,
                    poa_beam=entry.poa_beam,
                    cell_temperature=entry.cell_temperature,
                    transmittance=entry.transmittance,
                )
            )
            scaled += entry.ac_power_w * factor
            original += entry.ac_power_w

        del original
        return ForecastResult(
            times=result.times,
            arrays=tuple(corrected),
            ac_power_w=scaled,
            clearsky=result.clearsky,
            now_index=result.now_index,
        )

    def _compute(
        self,
        weather: HourlyWeather,
        measured: dict[str, float | None] | None = None,
        soc: float | None = None,
    ) -> ForecastResult:
        """Run the model. Blocking; called in an executor."""
        start = np.datetime64(int(dt_util.utcnow().timestamp() // 300 * 300), "s")
        # From local midnight, so the daily totals and the energy dashboard
        # describe the whole of today rather than only what is left of it.
        midnight = dt_util.start_of_local_day()
        grid = build_time_grid(start, history_start=np.datetime64(int(midnight.timestamp()), "s"))
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
        result = forecast(self.system, grid, series)
        result = replace(result, now_index=int(np.searchsorted(grid.times, start)))

        if measured:
            self._learn(result, series.ghi, measured, soc)

        result = self._correct(result, forecast_clearness(grid.times, self.system, series.ghi))
        self._track_band(result, measured or {}, soc)
        return result

    def _track_band(self, result: ForecastResult, measured: dict[str, float | None], soc: float | None) -> None:
        """Compare the day's output with the forecasts made for it, and refresh the band.

        Measured against the arrays that have a power sensor, on both sides, so an
        array without one neither inflates the forecast nor goes missing from the
        actual. Without any, the band stays at its starting width.
        """
        today = dt_util.now()
        day = today.date().toordinal()
        midnight = dt_util.start_of_local_day(today)
        start = int(midnight.timestamp())
        end = int(dt_util.start_of_local_day(today.date() + timedelta(days=1)).timestamp())

        sensed = [entry for entry in result.arrays if entry.name in self.power_sensors]
        if sensed:
            snapshot = None
            if day not in self.band.snapshots:
                power = np.sum([entry.dc_power_w for entry in sensed], axis=0)
                # One hour spare, for the day the clocks go back.
                snapshot = hourly_snapshot(day, start, result.times, power, hours=BAND_LEADS * 24 + 1)
            self.band.start_day(day, start, end, snapshot)

            readings = [measured.get(entry.name) for entry in sensed]
            usable = all(value is not None for value in readings) and (soc is None or soc < CURTAILMENT_SOC_PCT)
            total = float(sum(value for value in readings if value is not None)) if usable else None
            self.band.tick(int(dt_util.utcnow().timestamp()), total)

        self._bounds = self.band.bounds(day)
