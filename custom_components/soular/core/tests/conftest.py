"""Shared fixtures for the forecasting core.

The site described here is the one the shading maps in ``solar-data`` were
measured at, so tests exercise real geometry rather than a round-numbers site
that would hide sign and convention errors.
"""

import numpy as np
import pytest

from custom_components.soular.core.pipeline import SystemSpec
from custom_components.soular.core.types import ArraySpec, InverterSpec, SiteSpec, TimeArray

# Morisset Park, NSW. Southern hemisphere and east of Greenwich, which between
# them catch every azimuth and longitude sign error worth catching.
SITE = SiteSpec(latitude=-33.11915471966274, longitude=151.53401076793673, elevation_m=10.0)

# Four strings, 27.28 kWp of Jinko JKM440. "north" and "south" are building names
# rather than orientations: both face 354 degrees and have identical capacity, so
# only their shading tells them apart.
ARRAYS = (
    ArraySpec(name="east", azimuth_deg=84.0, tilt_deg=25.0, dc_capacity_w=7920.0),
    ArraySpec(name="west", azimuth_deg=264.0, tilt_deg=25.0, dc_capacity_w=7920.0),
    ArraySpec(name="north", azimuth_deg=354.0, tilt_deg=25.0, dc_capacity_w=5720.0),
    ArraySpec(name="south", azimuth_deg=354.0, tilt_deg=25.0, dc_capacity_w=5720.0),
)

INVERTER = InverterSpec(name="default", ac_limit_w=20000.0)


@pytest.fixture
def site() -> SiteSpec:
    """Return the reference site."""
    return SITE


@pytest.fixture
def system() -> SystemSpec:
    """Return the reference site's full four-array configuration."""
    return SystemSpec(site=SITE, arrays=ARRAYS, inverters={"default": INVERTER})


def day_times(date: str = "2026-01-15", step_minutes: int = 5) -> TimeArray:
    """Build a full UTC day sampled at ``step_minutes``."""
    start = np.datetime64(f"{date}T00:00:00", "s")
    end = start + np.timedelta64(24 * 3600, "s")
    return np.arange(start, end, np.timedelta64(step_minutes * 60, "s"))


@pytest.fixture
def summer_day() -> TimeArray:
    """Return a southern-hemisphere summer day at five-minute resolution."""
    return day_times("2026-01-15")


@pytest.fixture
def winter_day() -> TimeArray:
    """Return a southern-hemisphere winter day, when the sun tracks lowest and north."""
    return day_times("2026-06-21")
