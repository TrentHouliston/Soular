"""Tests for observation blending.

The invariants here are what stop a nowcast making things worse. A blend that can
leave the range of its inputs, or that keeps applying an observation past the
point where the weather model has become the better estimate, degrades the
forecast precisely in the conditions it was added to improve.
"""

import numpy as np
import pytest

from custom_components.soular.core.blend import (
    FORECAST_VARIANCE,
    MAX_GAP_MINUTES,
    Observation,
    blend,
    invert_power_to_index,
    observation_from_irradiance,
    pv_observation,
    satellite_observation,
)
from custom_components.soular.core.clearsky import K_MAX

NOW = np.datetime64("2026-01-15T02:00:00", "s")


def grid(minutes: int = 480, step: int = 5) -> np.ndarray:
    """Build a forecast grid starting at the observation instant."""
    return np.arange(NOW, NOW + np.timedelta64(minutes * 60, "s"), np.timedelta64(step * 60, "s")).astype(
        "datetime64[s]"
    )


def at_gap(minutes: float, observations: list[Observation], forecast_k: float = 0.9, observed_k: float = 0.2) -> float:
    """Return the observed share at a given gap from the observation."""
    times = np.array([NOW + np.timedelta64(int(minutes * 60), "s")], dtype="datetime64[s]")
    del forecast_k, observed_k
    _, share = blend(times, np.array([0.9]), observations)
    return float(share[0])


def test_no_observations_leaves_the_forecast_untouched() -> None:
    """With nothing observed, the model's own forecast passes straight through."""
    times = grid()
    forecast = np.full(times.size, 0.8)
    blended, share = blend(times, forecast, [])

    np.testing.assert_array_equal(blended, forecast)
    assert np.all(share == 0.0)


def test_a_fresh_observation_dominates() -> None:
    """At its own instant the satellite is far more precise than the model.

    Measured: persistence at zero gap has variance 0.008 against the model's
    0.070, so it should carry the large majority of the answer.
    """
    times = grid()
    blended, share = blend(times, np.full(times.size, 0.9), [satellite_observation(NOW, 0.2)])

    assert share[0] > 0.85
    assert blended[0] < 0.35


def test_the_crossover_lands_where_it_was_measured() -> None:
    """Persistence stops winning at around two and a half hours.

    That is not a tuned constant -- it falls out of the two variance models. The
    measurement it has to match: persistence RMSE reaches the model's 0.265
    somewhere between two and three hours.
    """
    assert at_gap(60, [satellite_observation(NOW, 0.2)]) > 0.5
    assert at_gap(150, [satellite_observation(NOW, 0.2)]) == pytest.approx(0.5, abs=0.12)
    assert at_gap(360, [satellite_observation(NOW, 0.2)]) < 0.45


def test_influence_is_monotone_in_the_gap() -> None:
    """Confidence in an observation only ever falls as the gap widens."""
    times = grid()
    _, share = blend(times, np.full(times.size, 0.9), [satellite_observation(NOW, 0.3)])
    assert np.all(np.diff(share) <= 1e-12)


def test_influence_stops_entirely_past_the_horizon() -> None:
    """Far enough ahead the forecast is exactly the model again.

    Exactly, not nearly: a weight that merely tends to zero leaves the long-range
    forecast permanently a hair from the model it came from.
    """
    times = grid(minutes=int(MAX_GAP_MINUTES) + 180)
    forecast = np.full(times.size, 0.9)
    blended, share = blend(times, forecast, [satellite_observation(NOW, 0.1)])

    far = (times - NOW).astype("timedelta64[m]").astype(int) > MAX_GAP_MINUTES
    assert far.any()
    assert np.all(share[far] == 0.0)
    np.testing.assert_allclose(blended[far], forecast[far])


def test_the_blend_stays_between_its_inputs() -> None:
    """Inverse-variance weighting is convex by construction.

    The additive alternative -- applying an observed bias to the forecast --
    fails exactly here, and fails worst under thick cloud where the forecast
    index is near zero and subtracting a bias makes it negative.
    """
    times = grid()
    for forecast_k, observed_k in [(0.9, 0.05), (0.05, 0.9), (0.1, 0.1), (1.2, 0.0)]:
        blended, _ = blend(times, np.full(times.size, forecast_k), [satellite_observation(NOW, observed_k)])
        lo, hi = min(forecast_k, observed_k), max(forecast_k, observed_k)
        assert np.all(blended >= lo - 1e-9)
        assert np.all(blended <= hi + 1e-9)


def test_stacked_observations_cannot_displace_the_forecast() -> None:
    """Several observations sharpen the estimate; they do not replace it.

    This is the regression that motivated inverse-variance weighting. Three
    stacked exponential weights summed past one, so the forecast was discarded
    outright near the observation and then over-applied past the crossover.
    """
    times = grid()
    observations = [satellite_observation(NOW - np.timedelta64(m * 60, "s"), 0.2) for m in (0, 10, 20)]
    _, share = blend(times, np.full(times.size, 0.9), observations)

    assert np.all(share <= 1.0)
    assert np.all(share >= 0.0)
    # More observations means more confidence, never more than total confidence.
    _, single = blend(times, np.full(times.size, 0.9), [satellite_observation(NOW, 0.2)])
    assert share[0] >= single[0]


def test_the_array_starts_sharper_and_fades_faster_than_the_satellite() -> None:
    """One roof beats a five-kilometre pixel now, and loses to it later."""
    assert at_gap(0, [pv_observation(NOW, 0.3)]) > at_gap(0, [satellite_observation(NOW, 0.3)])
    assert at_gap(240, [pv_observation(NOW, 0.3)]) < at_gap(240, [satellite_observation(NOW, 0.3)])


def test_latency_is_handled_by_the_same_expression_as_lead() -> None:
    """A half-hour-old observation is already a forecast; no special case."""
    times = grid()
    stale = satellite_observation(NOW - np.timedelta64(30 * 60, "s"), 0.2)
    _, stale_share = blend(times, np.full(times.size, 0.9), [stale])
    _, fresh_share = blend(times, np.full(times.size, 0.9), [satellite_observation(NOW, 0.2)])
    assert stale_share[0] < fresh_share[0]


def test_a_less_certain_forecast_yields_more_ground() -> None:
    """If the model is known to be worse, the observation should count for more."""
    times = grid()
    _, confident = blend(times, np.full(times.size, 0.9), [satellite_observation(NOW, 0.2)], FORECAST_VARIANCE / 4)
    _, unsure = blend(times, np.full(times.size, 0.9), [satellite_observation(NOW, 0.2)], FORECAST_VARIANCE * 4)
    assert unsure[0] > confident[0]


class TestObservationFromIrradiance:
    """Turning a measurement into an index."""

    def test_a_clear_reading_gives_an_index_near_one(self) -> None:
        """Observed equal to clear sky is an index of one."""
        assert observation_from_irradiance(NOW, 800.0, 800.0) == pytest.approx(1.0)

    def test_low_sun_is_refused(self) -> None:
        """A ratio of two small numbers is not a measurement of cloud."""
        assert observation_from_irradiance(NOW, 3.0, 4.0) is None

    def test_cloud_enhancement_is_capped(self) -> None:
        """Enhancement is real, but not unbounded."""
        assert observation_from_irradiance(NOW, 5000.0, 800.0) == K_MAX


class TestInvertPowerToIndex:
    """Reading the array as an irradiance sensor."""

    def test_full_output_reads_as_clear(self) -> None:
        """Producing what the model expects means the sky is as modelled."""
        assert invert_power_to_index(5000.0, 5000.0, min_modelled_w=500.0, transmittance=1.0) == pytest.approx(1.0)

    def test_half_output_reads_as_half(self) -> None:
        """Producing half of expectation implies half the irradiance."""
        assert invert_power_to_index(2500.0, 5000.0, min_modelled_w=500.0, transmittance=1.0) == pytest.approx(0.5)

    def test_shaded_directions_are_refused(self) -> None:
        """Behind a tree, low output means shade, not cloud.

        Reading it as cloud would darken the whole forecast for hours on the
        strength of a shadow the model already knows about.
        """
        assert invert_power_to_index(500.0, 5000.0, min_modelled_w=500.0, transmittance=0.4) is None

    def test_low_expected_output_is_refused(self) -> None:
        """Near sunrise the ratio is dominated by everything except cloud."""
        assert invert_power_to_index(10.0, 100.0, min_modelled_w=500.0, transmittance=1.0) is None

    def test_the_result_is_bounded(self) -> None:
        """An implausible over-production cannot become an implausible forecast."""
        assert invert_power_to_index(50_000.0, 5000.0, min_modelled_w=500.0, transmittance=1.0) == K_MAX
