"""Site and array geometry for the offline tools, read from a TOML file.

The same file drives the backtest and the shading-map builder, so a map is always
built against the geometry it will be scored and served with::

    [site]
    latitude = -33.12
    longitude = 151.53
    altitude = 10          # metres, optional
    timezone = "Australia/Sydney"
    albedo = 0.20          # optional; match what the integration is configured with

    [arrays.east]
    azimuth = 84           # compass degrees, 0 = north, clockwise
    tilt = 25
    dc_watts = 7920

Array names must match the names used in Home Assistant and in the power CSV.
"""

from dataclasses import dataclass
from pathlib import Path
import tomllib
from zoneinfo import ZoneInfo

from custom_components.soular.core.types import ArraySpec, InverterSpec, SiteSpec

# The integration's default ground reflectance ("grass").
DEFAULT_ALBEDO = 0.20


@dataclass(frozen=True, slots=True)
class Site:
    """A parsed site file."""

    spec: SiteSpec
    arrays: tuple[ArraySpec, ...]
    inverter: InverterSpec
    timezone: ZoneInfo


def load_site(path: Path, *, ac_limit_w: float) -> Site:
    """Read site and array geometry."""
    config = tomllib.loads(path.read_text())
    site = config["site"]
    arrays = tuple(
        ArraySpec(
            name=name,
            azimuth_deg=float(entry["azimuth"]),
            tilt_deg=float(entry["tilt"]),
            dc_capacity_w=float(entry["dc_watts"]),
        )
        for name, entry in sorted(config["arrays"].items())
    )
    return Site(
        spec=SiteSpec(
            latitude=float(site["latitude"]),
            longitude=float(site["longitude"]),
            elevation_m=float(site.get("altitude", 0.0)),
            albedo=float(site.get("albedo", DEFAULT_ALBEDO)),
        ),
        arrays=arrays,
        inverter=InverterSpec(name="default", ac_limit_w=ac_limit_w),
        timezone=ZoneInfo(str(site.get("timezone", "UTC"))),
    )
