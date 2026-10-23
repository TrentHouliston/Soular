"""The forecast attribute must be accepted by haeo's own extractor.

haeo's detector is all-or-nothing: one malformed point rejects the entire entity,
and it then falls back to reading the plain state, which drops the forecast with
no error logged anywhere. A contract that fails silently is exactly the kind that
needs an explicit test.

The detector is reimplemented here rather than imported. haeo is not a dependency
of this project, and vendoring the check means it keeps holding even if that repo
moves -- while still failing loudly here if the shape ever drifts.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from homeassistant.core import HomeAssistant
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

# One local day, so a series that ends mid-day is detectable.
SECONDS_PER_DAY = 86400


def haeo_detects(attributes: Mapping[str, Any]) -> bool:
    """Reimplementation of haeo's ``extractors.haeo.Parser.detect``."""
    if "forecast" not in attributes:
        return False

    forecast = attributes["forecast"]
    if not isinstance(forecast, Sequence) or isinstance(forecast, str):
        return False
    if not forecast:
        return False

    for item in forecast:
        if not isinstance(item, Mapping) or "time" not in item or "value" not in item:
            return False
        time = item["time"]
        if isinstance(time, str):
            try:
                datetime.fromisoformat(time)
            except ValueError:
                return False
        elif not isinstance(time, datetime):
            return False
        if not isinstance(item["value"], (int, float)) or isinstance(item["value"], bool):
            return False

    unit = attributes.get("unit_of_measurement")
    if not isinstance(unit, str) or not unit:
        return False

    device_class = attributes.get("device_class")
    return device_class is None or isinstance(device_class, str)


@pytest.fixture
def forecast_attributes(hass: HomeAssistant, configured: MockConfigEntry) -> Mapping[str, Any]:
    """Return the site power sensor's attributes."""
    state = hass.states.get("sensor.morisset_park_estimated_power_production_now")
    assert state is not None, "the site power sensor should exist"
    return state.attributes


def test_haeo_accepts_the_forecast_attribute(forecast_attributes: Mapping[str, Any]) -> None:
    """The emitted attribute passes haeo's detector."""
    assert haeo_detects(forecast_attributes)


def test_series_carries_units_and_interpolation(forecast_attributes: Mapping[str, Any]) -> None:
    """Units and interpolation mode travel with the series.

    Watts and ``power`` because that is haeo's base unit for power, so nothing
    has to be converted. Linear because the series is instantaneous samples of a
    continuous curve; ``previous`` would turn it into a staircase.
    """
    assert forecast_attributes["unit_of_measurement"] == "W"
    assert forecast_attributes["device_class"] == "power"
    assert forecast_attributes["interpolation_mode"] == "linear"


def test_series_starts_at_or_before_now(forecast_attributes: Mapping[str, Any]) -> None:
    """There is a point covering the present instant.

    When haeo's extractor matches it discards the sensor's own state and takes
    interval zero from the series. Without a leading point it extrapolates flat
    from the first one it has.
    """
    from homeassistant.util import dt as dt_util  # noqa: PLC0415

    points = forecast_attributes["forecast"]
    first = datetime.fromisoformat(points[0]["time"])
    assert first <= dt_util.utcnow()


def test_series_spans_whole_days(forecast_attributes: Mapping[str, Any]) -> None:
    """The series length is a whole number of days.

    haeo pads a forecast to whole days by wrapping its head onto its tail. A
    37-hour series would therefore get 11 hours of this morning spliced onto its
    end, which looks like a plausible forecast and is not one.
    """
    points = forecast_attributes["forecast"]
    first = datetime.fromisoformat(points[0]["time"])
    last = datetime.fromisoformat(points[-1]["time"])
    # The final point opens the last interval, so the span is one step short.
    span = (last - first).total_seconds()
    assert span % SECONDS_PER_DAY != 0 or span > 0
    covered = span + 15 * 60
    assert covered % SECONDS_PER_DAY == 0, f"series covers {covered / 3600:.2f} hours, not a whole number of days"


def test_series_is_dense_enough_near_term(forecast_attributes: Mapping[str, Any]) -> None:
    """Near-term points are five minutes apart.

    haeo trapezoid-integrates the series into its horizon intervals, and its
    finest tier is one minute. Point density buys accuracy directly there.
    """
    points = forecast_attributes["forecast"]
    first = datetime.fromisoformat(points[0]["time"])
    second = datetime.fromisoformat(points[1]["time"])
    assert (second - first).total_seconds() == 300


def test_values_are_plain_floats(forecast_attributes: Mapping[str, Any]) -> None:
    """Values are floats, not numpy scalars.

    A numpy float survives haeo's ``isinstance(value, (int, float))`` check but
    does not survive being written to the state machine and read back, so this
    catches a failure that would only appear in production.
    """
    for point in forecast_attributes["forecast"]:
        assert type(point["value"]) is float


def test_malformed_points_would_be_rejected() -> None:
    """The vendored detector is strict, so the test above means something."""
    good = {
        "forecast": [{"time": "2026-01-15T00:00:00+00:00", "value": 1.0}],
        "unit_of_measurement": "W",
    }
    assert haeo_detects(good)

    assert not haeo_detects({**good, "forecast": []})
    assert not haeo_detects({**good, "unit_of_measurement": ""})
    assert not haeo_detects({**good, "forecast": [{"time": "not a time", "value": 1.0}]})
    assert not haeo_detects({**good, "forecast": [{"time": "2026-01-15T00:00:00+00:00", "value": "1.0"}]})
    assert not haeo_detects({**good, "forecast": [{"value": 1.0}]})
