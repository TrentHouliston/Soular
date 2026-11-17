"""Telling equipment that is off from weather that is dark."""

from custom_components.soular.core.mask import outage

CAPACITY = [7920.0, 7920.0, 5720.0, 5720.0]
FORECAST = [4000.0, 4200.0, 3000.0, 2900.0]


def test_a_working_site_is_not_an_outage() -> None:
    """Every array a little under its forecast is weather."""
    assert not outage([3600.0, 3700.0, 2700.0, 2500.0], FORECAST, CAPACITY)


def test_a_storm_is_not_an_outage() -> None:
    """Every array dark together, but still lit by diffuse light, is weather."""
    assert not outage([500.0, 520.0, 400.0, 380.0], FORECAST, CAPACITY)


def test_a_dead_array_is_an_outage() -> None:
    """Two strings reading nothing in full sun, as during commissioning, are off."""
    assert outage([0.0, 3800.0, 0.0, 2600.0], FORECAST, CAPACITY)


def test_a_whole_site_reading_nothing_is_an_outage() -> None:
    """No weather leaves a well-lit roof at zero."""
    assert outage([0.0, 0.0, 0.0, 0.0], FORECAST, CAPACITY)


def test_one_array_far_out_of_step_is_an_outage() -> None:
    """A string at a few per cent while the others run normally has tripped, even above the dead threshold."""
    assert outage([200.0, 3800.0, 2800.0, 2600.0], FORECAST, CAPACITY)


def test_night_is_never_an_outage() -> None:
    """With nothing forecast there is nothing to be out of step with."""
    assert not outage([0.0] * 4, [0.0] * 4, CAPACITY)
