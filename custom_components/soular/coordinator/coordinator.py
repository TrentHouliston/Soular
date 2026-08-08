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

from custom_components.soular.const import (
    DOMAIN,
    ENSEMBLE_INTERVAL,
    LEARNING_SAMPLE_INTERVAL,
    RECOMPUTE_INTERVAL,
    SATELLITE_INTERVAL,
    UPDATE_INTERVAL,
)
from custom_components.soular.coordinator.actuals import ArraySample, Learner, read_percentage, read_power
from custom_components.soular.core.blend import (
    Observation,
    apply_to_weather,
    observation_from_irradiance,
    satellite_observation,
)
from custom_components.soular.core.clearsky import clear_sky, clear_sky_index
from custom_components.soular.core.correction import build_features
from custom_components.soular.core.ensemble import (
    DEFAULT_QUANTILES,
    energy_quantiles,
    quantile_indices,
    spread_from_members,
    trajectories,
)
from custom_components.soular.core.geometry import solar_geometry
from custom_components.soular.core.nwp import weather_from_hourly
from custom_components.soular.core.pipeline import (
    ArrayForecast,
    ForecastResult,
    SystemSpec,
    WeatherSeries,
    build_time_grid,
    forecast,
)
from custom_components.soular.sources import ensemble as ensemble_source
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


# Below this the central clear-sky index is too small for a ratio against it to
# carry any information, and the quantile is left at the deterministic value.
MIN_SCALING_INDEX = 0.01


def _scale_to_index(power: np.ndarray, central: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Scale modelled power by a clear-sky index relative to the central one.

    Re-running the whole model per trajectory would be exact and about fifteen
    times the work. Power is close to linear in irradiance once geometry and
    shading are fixed, and both are identical across trajectories -- the sun is
    where it is regardless of the weather.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(central > MIN_SCALING_INDEX, index / np.maximum(central, 1e-9), 1.0)
    return np.asarray(np.clip(power * ratio, 0.0, None), dtype=np.float64)


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
    ) -> None:
        """Set up the coordinator for a configured system."""
        super().__init__(hass, _LOGGER, name=f"{DOMAIN} {name}", update_interval=RECOMPUTE_INTERVAL)
        self.system = system
        self.power_sensors = power_sensors or {}
        self.soc_sensor = soc_sensor
        self.learner = learner or Learner()
        self._weather: HourlyWeather | None = None
        self._weather_at: datetime | None = None
        self._weather_error: str | None = None
        self._satellite: satellite_source.SatelliteObservations | None = None
        self._satellite_at: datetime | None = None
        self._satellite_error: str | None = None
        # Share of the near-term forecast that came from observation rather than
        # from the weather model. Surfaced so "is this a nowcast?" is answerable.
        self.observed_share: float = 0.0
        self._ensemble: ensemble_source.EnsembleForecast | None = None
        self._ensemble_at: datetime | None = None
        self._ensemble_error: str | None = None
        self._share_series: np.ndarray | None = None
        # Site AC power at each published quantile, and day-ahead energy the same
        # way. Empty when no ensemble is available, which entities report as
        # unknown rather than silently equal to the median.
        self.quantile_power: dict[float, np.ndarray] = {}
        self.quantile_energy: dict[str, dict[float, float]] = {}

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

    @property
    def ensemble_fetched_at(self) -> datetime | None:
        """When ensemble members were last successfully retrieved."""
        return self._ensemble_at

    @property
    def ensemble_error(self) -> str | None:
        """The last ensemble fetch failure, if the most recent attempt failed."""
        return self._ensemble_error

    @property
    def uncertainty_available(self) -> bool:
        """Whether the forecast currently carries a spread at all."""
        return bool(self.quantile_power)

    async def _async_update_data(self) -> ForecastResult:
        """Refresh weather if it is due, then recompute the forecast."""
        await self._refresh_weather()
        # Neither is fatal. Losing the satellite costs the nowcast; losing the
        # ensemble costs the uncertainty band. Both leave a working forecast.
        await self._refresh_satellite()
        await self._refresh_ensemble()

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

    async def _refresh_ensemble(self) -> None:
        """Fetch ensemble members when the cached copy is due."""
        now = dt_util.utcnow()
        if self._ensemble_at is not None and now - self._ensemble_at < ENSEMBLE_INTERVAL:
            return

        session = async_get_clientsession(self.hass)
        try:
            self._ensemble = await ensemble_source.fetch(session, self.system.site.latitude, self.system.site.longitude)
        except ensemble_source.EnsembleError as err:
            self._ensemble_error = str(err)
            _LOGGER.debug("No ensemble spread this cycle: %s", err)
        else:
            self._ensemble_at = now
            self._ensemble_error = None

    def _quantiles(self, result: ForecastResult, series: WeatherSeries) -> None:
        """Derive quantile power and energy from the ensemble, if there is one."""
        self.quantile_power = {}
        self.quantile_energy = {}

        forecast_ensemble = self._ensemble
        if forecast_ensemble is None or not forecast_ensemble.usable:
            return

        times = result.times
        geometry = solar_geometry(times, self.system.site)
        clearsky = clear_sky(times, self.system.site, geometry).ghi

        member_geometry = solar_geometry(forecast_ensemble.times, self.system.site)
        member_clearsky = clear_sky(forecast_ensemble.times, self.system.site, member_geometry).ghi
        member_k = np.nan_to_num(
            np.vstack(
                [clear_sky_index(row, member_clearsky) for row in np.nan_to_num(forecast_ensemble.members, nan=0.0)]
            ),
            nan=1.0,
        )

        spread = spread_from_members(forecast_ensemble.times, member_k, DEFAULT_QUANTILES)
        median_k = np.nan_to_num(clear_sky_index(series.ghi, clearsky), nan=1.0)

        # Power at each quantile is the pointwise band; honest for "how bright
        # might it be at 2pm", and deliberately not what the energy figures use.
        indices = quantile_indices(spread, times, median_k, self._share_series)
        for level, row in zip(DEFAULT_QUANTILES, indices, strict=True):
            self.quantile_power[level] = _scale_to_index(result.ac_power_w, median_k, row)

        # Energy needs whole coherent days, so it integrates trajectories.
        paths = trajectories(spread, times, median_k, observed_share=self._share_series)
        powers = np.vstack([_scale_to_index(result.ac_power_w, median_k, path) for path in paths])

        local = dt_util.now()
        midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
        for label, offset in (("today", 0), ("tomorrow", 1)):
            start = np.datetime64(int((midnight + timedelta(days=offset)).timestamp()), "s")
            end = np.datetime64(int((midnight + timedelta(days=offset + 1)).timestamp()), "s")
            if (times >= start).any() and (times < end).any():
                self.quantile_energy[label] = energy_quantiles(times, powers, start, end, DEFAULT_QUANTILES)

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
        self._share_series = share
        return blended

    def _learn(self, result: ForecastResult, measured: dict[str, float | None], soc: float | None) -> None:
        """Fold the present instant's measured output into the correction."""
        now = dt_util.utcnow()
        if self.learner.last_sample_at is not None and now - self.learner.last_sample_at < LEARNING_SAMPLE_INTERVAL:
            return

        geometry = solar_geometry(result.times[:1], self.system.site)
        clearsky = clear_sky(result.times[:1], self.system.site, geometry).ghi
        day_of_year = float(now.timetuple().tm_yday)
        capacity = {array.name: array.dc_capacity_w for array in self.system.arrays}

        for entry in result.arrays:
            value = measured.get(entry.name)
            if value is None:
                continue
            sample = ArraySample(
                predicted_w=float(entry.dc_power_w[0]),
                capacity_w=capacity.get(entry.name, 0.0),
                transmittance=float(entry.transmittance[0]),
                elevation_deg=float(geometry.apparent_elevation[0]),
                # Clear-sky index of the forecast at this instant, which is the
                # same quantity the correction is applied against later.
                clearness=float(np.nan_to_num(clear_sky_index(entry.poa_global[:1], clearsky), nan=1.0)[0]),
                day_of_year=day_of_year,
                temperature_c=float(entry.cell_temperature[0]),
            )
            self.learner.observe(entry.name, sample, value, soc)

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
        )

    def _compute(
        self,
        weather: HourlyWeather,
        measured: dict[str, float | None] | None = None,
        soc: float | None = None,
    ) -> ForecastResult:
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
        result = forecast(self.system, grid, series)

        if measured:
            self._learn(result, measured, soc)

        geometry = solar_geometry(grid.times, self.system.site)
        clearsky = clear_sky(grid.times, self.system.site, geometry).ghi
        clearness = np.nan_to_num(clear_sky_index(series.ghi, clearsky), nan=1.0)
        corrected = self._correct(result, clearness)
        self._quantiles(corrected, series)
        return corrected
