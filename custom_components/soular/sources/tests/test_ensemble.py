"""Tests for the ensemble source.

The response shape is the trap. Open-Meteo names the control run after the plain
variable and perturbs the rest into ``_memberNN`` columns, and it returns a full
set of correctly shaped columns even where it has nothing to say -- every value
null. A parser that trusts the shape reports fifty members and a spread of zero.
"""

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
import numpy as np
import pytest
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker

from custom_components.soular.sources.ensemble import ENSEMBLE_URL, EnsembleError, fetch, parse

LATITUDE = -33.11915471966274
LONGITUDE = 151.53401076793673


def payload(members: int = 4, steps: int = 4, *, null_from: int | None = None) -> dict[str, Any]:
    """Build a well-formed ensemble response with a control and perturbed members."""
    times = [f"2026-01-15T{hour:02d}:00" for hour in range(0, steps * 3, 3)]
    hourly: dict[str, Any] = {"time": times}
    for member in range(members):
        values: list[float | None] = [400.0 + 50.0 * member + 10.0 * step for step in range(steps)]
        if null_from is not None and member >= null_from:
            values = [None] * steps
        key = "shortwave_radiation" if member == 0 else f"shortwave_radiation_member{member:02d}"
        hourly[key] = values
    return {"latitude": LATITUDE, "longitude": LONGITUDE, "hourly": hourly}


def test_the_control_run_becomes_member_zero() -> None:
    """The unsuffixed column is the control, and it belongs in the ensemble."""
    forecast = parse(payload(members=4))

    assert len(forecast) == 4
    assert forecast.members.shape == (4, 4)
    # Members are ordered by index, so the control is first and the rest ascend.
    assert np.all(np.diff(forecast.members[:, 0]) > 0)


def test_members_are_ordered_numerically_not_lexically() -> None:
    """``member10`` sorts before ``member9`` as a string, which would shuffle ranks.

    Ranks are what ensemble copula coupling carries; scrambling the member axis
    would not change the quantiles at all, so nothing downstream would complain.
    """
    forecast = parse(payload(members=12))
    assert np.all(np.diff(forecast.members[:, 0]) > 0)


def test_null_members_are_dropped() -> None:
    """A correctly shaped all-null column is not a member.

    Counting it would report a spread the model never expressed, and outside the
    forecast horizon that is most of the response.
    """
    forecast = parse(payload(members=6, null_from=3))
    assert len(forecast) == 3
    assert np.isfinite(forecast.members).all()


def test_a_thin_ensemble_is_not_usable() -> None:
    """Two members cannot describe a distribution, so do not pretend they can."""
    assert not parse(payload(members=2)).usable
    assert parse(payload(members=6)).usable


def test_a_response_without_members_is_rejected() -> None:
    """The endpoint answers historical ranges with a shape and no data."""
    empty = {"hourly": {"time": ["2026-01-15T00:00"]}}
    with pytest.raises(EnsembleError, match="no members"):
        parse(empty)

    with pytest.raises(EnsembleError, match="no series"):
        parse({"latitude": LATITUDE})


async def test_fetch_asks_for_an_ensemble_model(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker) -> None:
    """The request has to name a model, or the API serves a single run."""
    aioclient_mock.get(ENSEMBLE_URL, json=payload(members=5))
    forecast = await fetch(async_get_clientsession(hass), LATITUDE, LONGITUDE)

    assert len(forecast) == 5
    query = aioclient_mock.mock_calls[0][1].query
    assert query["models"] == "ecmwf_ifs025"
    assert query["hourly"] == "shortwave_radiation"
    # UTC throughout: the model works in instants and localises only at the edge.
    assert query["timezone"] == "UTC"


async def test_an_error_response_raises(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker) -> None:
    """A rejected request must not read as an ensemble of zero members."""
    aioclient_mock.get(ENSEMBLE_URL, status=400, text="bad request")
    with pytest.raises(EnsembleError, match="HTTP 400"):
        await fetch(async_get_clientsession(hass), LATITUDE, LONGITUDE)


async def test_a_json_error_body_raises(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker) -> None:
    """Open-Meteo also reports errors with HTTP 200 and an ``error`` flag."""
    aioclient_mock.get(ENSEMBLE_URL, json={"error": True, "reason": "no such model"})
    with pytest.raises(EnsembleError, match="no such model"):
        await fetch(async_get_clientsession(hass), LATITUDE, LONGITUDE)
