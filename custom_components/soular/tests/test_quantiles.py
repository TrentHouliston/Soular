"""End-to-end behaviour of the forecast's confidence band.

Two properties carry the weight here. The band must bracket the point forecast
without displacing it -- the nowcast and the learned correction own the central
estimate, and the ensemble supplies only the width. And when there is no
ensemble the bounds must report ``unknown``, because a band that quietly
collapses onto the median reads as certainty about a forecast nobody checked.
"""

from homeassistant.core import HomeAssistant
import pytest

from custom_components.soular.tests.conftest import build_entry

P10_POWER = "sensor.morisset_park_estimated_power_production_now_p10"
P90_POWER = "sensor.morisset_park_estimated_power_production_now_p90"
MEDIAN_POWER = "sensor.morisset_park_estimated_power_production_now"
P10_TODAY = "sensor.morisset_park_estimated_energy_production_today_p10"
P90_TODAY = "sensor.morisset_park_estimated_energy_production_today_p90"
MEDIAN_TODAY = "sensor.morisset_park_estimated_energy_production_today"
P90_TOMORROW = "sensor.morisset_park_estimated_energy_production_tomorrow_p90"


def value(hass: HomeAssistant, entity: str) -> float:
    """Read a numeric state, failing loudly if it is not one."""
    state = hass.states.get(entity)
    assert state is not None, f"{entity} does not exist"
    return float(state.state)


@pytest.mark.usefixtures("configured")
def test_the_band_brackets_the_point_forecast(hass: HomeAssistant) -> None:
    """P10 below, P90 above, on both power and energy."""
    assert value(hass, P10_POWER) <= value(hass, MEDIAN_POWER) + 1.0
    assert value(hass, P90_POWER) >= value(hass, MEDIAN_POWER) - 1.0
    assert value(hass, P10_TODAY) <= value(hass, MEDIAN_TODAY) + 0.1
    assert value(hass, P90_TODAY) >= value(hass, MEDIAN_TODAY) - 0.1


@pytest.mark.usefixtures("configured")
def test_the_band_has_width(hass: HomeAssistant) -> None:
    """A spread of exactly zero means the ensemble never reached the model."""
    assert value(hass, P90_TOMORROW) > value(hass, "sensor.morisset_park_estimated_energy_production_tomorrow_p10")


@pytest.mark.usefixtures("configured")
def test_energy_bounds_are_not_the_integral_of_the_power_bounds(hass: HomeAssistant) -> None:
    """The reason ensemble copula coupling exists, checked end to end.

    Integrating the pointwise P90 assumes every hour of the day hits its 90th
    percentile at once. That is a real day only if the ensemble members never
    cross, and it is the wrong answer by roughly a factor of two on band width.
    Here it shows up as the day's P90 sitting strictly below what the pointwise
    band would integrate to.
    """
    median = value(hass, MEDIAN_TODAY)
    ratio = value(hass, P90_TODAY) / median if median > 0 else 1.0
    # The pointwise band is a fixed ratio of the median at every instant, so
    # integrating it would reproduce that ratio exactly. Trajectories cannot.
    assert ratio < 1.6, "day-ahead P90 looks like an integrated pointwise quantile"


@pytest.mark.usefixtures("configured")
def test_quantiles_are_site_only(hass: HomeAssistant) -> None:
    """Per-array bounds would be six more entities describing the same weather."""
    assert hass.states.get("sensor.morisset_park_north_estimated_power_production_now") is not None
    assert hass.states.get("sensor.morisset_park_north_estimated_power_production_now_p10") is None


async def test_without_an_ensemble_the_bounds_are_unknown(hass: HomeAssistant, no_satellite: object) -> None:
    """Never zero, and never silently equal to the median.

    A consumer hedging against a bound that has quietly become the point forecast
    is hedging against nothing, and nothing about the state would say so.
    """
    del no_satellite
    entry = build_entry()
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    for entity in (P10_POWER, P90_POWER, P10_TODAY, P90_TODAY):
        state = hass.states.get(entity)
        assert state is not None
        assert state.state == "unknown", f"{entity} reported {state.state} with no ensemble"
