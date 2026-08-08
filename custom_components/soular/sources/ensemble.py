"""Ensemble members, for forecast uncertainty.

Open-Meteo's ensemble endpoint serves the *current* run, not an archive: ask it
for a historical range and it returns a correctly shaped response full of nulls.
That is why this source could not be validated offline the way the deterministic
model and the satellite were, and why the quantiles it produces are checked by
unit tests on synthetic ensembles plus online coverage rather than by a replay.

The API names the control run after the plain variable and the perturbed members
``<variable>_memberNN``, so the control becomes member zero.
"""

import logging
import re
from typing import Any

from aiohttp import ClientError, ClientSession, ClientTimeout
import numpy as np

_LOGGER = logging.getLogger(__name__)

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

# ECMWF's 51-member ensemble has the broadest coverage and the longest horizon.
# BOM's ACCESS-GE is the regional alternative for Australia at 18 members.
DEFAULT_MODEL = "ecmwf_ifs025"

REQUEST_TIMEOUT = ClientTimeout(total=45)
FORECAST_DAYS = 3

MEMBER_SUFFIX = re.compile(r"_member(\d+)$")


class EnsembleError(Exception):
    """The ensemble API could not be reached, or returned nothing usable."""


class EnsembleForecast:
    """Member trajectories of shortwave radiation."""

    def __init__(self, times: np.ndarray, members: np.ndarray) -> None:
        """Store times and a (member, time) array."""
        self.times = times
        self.members = members

    def __len__(self) -> int:
        """Return the number of members."""
        return int(self.members.shape[0])

    @property
    def usable(self) -> bool:
        """Report whether enough members carry data to describe a spread."""
        minimum_members = 3
        return self.members.shape[0] >= minimum_members and bool(np.isfinite(self.members).any())


async def fetch(
    session: ClientSession,
    latitude: float,
    longitude: float,
    *,
    model: str = DEFAULT_MODEL,
) -> EnsembleForecast:
    """Fetch ensemble members for a site."""
    params = {
        "latitude": f"{latitude:.6f}",
        "longitude": f"{longitude:.6f}",
        "hourly": "shortwave_radiation",
        "models": model,
        "forecast_days": str(FORECAST_DAYS),
        "timezone": "UTC",
    }

    try:
        async with session.get(ENSEMBLE_URL, params=params, timeout=REQUEST_TIMEOUT) as response:
            if response.status != 200:  # noqa: PLR2004
                body = (await response.text())[:200]
                msg = f"ensemble API returned HTTP {response.status}: {body}"
                raise EnsembleError(msg)
            payload: dict[str, Any] = await response.json()
    except ClientError as err:
        msg = f"could not reach the ensemble API: {err}"
        raise EnsembleError(msg) from err
    except TimeoutError as err:
        msg = "the ensemble API timed out"
        raise EnsembleError(msg) from err

    if payload.get("error"):
        msg = f"ensemble API rejected the request: {payload.get('reason', 'no reason given')}"
        raise EnsembleError(msg)

    return parse(payload)


def parse(payload: dict[str, Any]) -> EnsembleForecast:
    """Convert an ensemble response into a (member, time) array."""
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        msg = f"ensemble response carried no series; keys were {sorted(payload)}"
        raise EnsembleError(msg)

    times = np.array(hourly["time"], dtype="datetime64[s]")
    rows: list[tuple[int, list[float]]] = []
    for column, values in hourly.items():
        if column == "time" or not column.startswith("shortwave_radiation"):
            continue
        match = MEMBER_SUFFIX.search(column)
        member = int(match.group(1)) if match else 0
        rows.append((member, [np.nan if value is None else float(value) for value in values]))

    if not rows:
        msg = "ensemble response carried no members"
        raise EnsembleError(msg)

    rows.sort(key=lambda item: item[0])
    members = np.array([values for _, values in rows], dtype=np.float64)

    # Drop members that are entirely null. The endpoint returns a full set of
    # correctly shaped columns even where it has nothing to say.
    keep = np.isfinite(members).any(axis=1)
    return EnsembleForecast(times=times, members=members[keep])
