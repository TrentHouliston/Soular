"""Turn a config entry into the core's system specification.

The boundary this file sits on is the point of the whole layout: everything to
its left is Home Assistant configuration, everything to its right is plain
numbers that the offline backtest can replay identically.
"""

from pathlib import Path

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE, CONF_NAME
from homeassistant.core import HomeAssistant

from custom_components.soular.const import (
    CONF_ALBEDO,
    CONF_AZIMUTH,
    CONF_DC_CAPACITY,
    CONF_DC_LOSS,
    CONF_ELEVATION,
    CONF_INVERTER_AC_LIMIT,
    CONF_SHADING_FILE,
    CONF_TEMPERATURE_COEFFICIENT,
    CONF_TILT,
    DEFAULT_ALBEDO,
    DEFAULT_DC_LOSS_PERCENT,
    DEFAULT_TEMPERATURE_COEFFICIENT,
    SHADING_DIRECTORY,
    SUBENTRY_TYPE_ARRAY,
)
from custom_components.soular.core.pipeline import SystemSpec
from custom_components.soular.core.shading import (
    ShadingFormatError,
    TransmittanceGrid,
    from_csv,
    from_horizon,
    from_npz,
)
from custom_components.soular.core.types import ArraySpec, InverterSpec, SiteSpec

INVERTER_NAME = "default"


def shading_directory(hass: HomeAssistant) -> Path:
    """Return the directory shading files are read from."""
    return Path(hass.config.path(SHADING_DIRECTORY))


def load_shading_file(path: Path, array_name: str) -> TransmittanceGrid:
    """Parse a shading file, dispatching on its extension.

    Blocking: reads from disk. Call from an executor.
    """
    if not path.exists():
        msg = f"no shading file at {path}"
        raise ShadingFormatError(msg)

    suffix = path.suffix.lower()
    if suffix == ".npz":
        return from_npz(path.read_bytes(), array_name)
    if suffix == ".csv":
        return from_csv(path.read_text(), array_name)
    if suffix in {".txt", ".tsv"}:
        return from_horizon(path.read_text())

    msg = f"unrecognised shading file type {suffix!r}; expected .npz, .csv, .txt or .tsv"
    raise ShadingFormatError(msg)


def build_arrays(entry: ConfigEntry) -> tuple[ArraySpec, ...]:
    """Build the array specs from the entry's subentries."""
    arrays: list[ArraySpec] = []
    for subentry in entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_TYPE_ARRAY:
            continue
        data = subentry.data
        arrays.append(
            ArraySpec(
                name=str(subentry.title),
                azimuth_deg=float(data[CONF_AZIMUTH]),
                tilt_deg=float(data[CONF_TILT]),
                dc_capacity_w=float(data[CONF_DC_CAPACITY]),
                # Stored as a percentage per degree because that is how datasheets
                # print it; the model wants a fraction.
                gamma_pdc=float(data.get(CONF_TEMPERATURE_COEFFICIENT, DEFAULT_TEMPERATURE_COEFFICIENT)) / 100.0,
                dc_loss_fraction=float(data.get(CONF_DC_LOSS, DEFAULT_DC_LOSS_PERCENT)) / 100.0,
                inverter=INVERTER_NAME,
            )
        )
    return tuple(sorted(arrays, key=lambda array: array.name))


def build_site(entry: ConfigEntry) -> SiteSpec:
    """Build the site spec from the entry's own data."""
    return SiteSpec(
        latitude=float(entry.data[CONF_LATITUDE]),
        longitude=float(entry.data[CONF_LONGITUDE]),
        elevation_m=float(entry.data.get(CONF_ELEVATION, 0.0)),
        albedo=float(entry.data.get(CONF_ALBEDO, DEFAULT_ALBEDO)),
    )


def build_system(
    entry: ConfigEntry,
    shading: dict[str, TransmittanceGrid] | None = None,
) -> SystemSpec:
    """Assemble the complete system spec the core runs on."""
    arrays = build_arrays(entry)
    limit = entry.data.get(CONF_INVERTER_AC_LIMIT)
    # With no limit configured, size the inverter to the array. "No limit" cannot
    # be represented as an enormous number: the efficiency curve is a function of
    # load *fraction*, so a nominally infinite inverter looks permanently idle and
    # the model returns zero everywhere. A DC/AC ratio of 1.0 is the neutral
    # assumption, and it means an unconfigured system simply never clips.
    default_limit = sum(array.dc_capacity_w for array in arrays) or 1.0
    inverter = InverterSpec(
        name=INVERTER_NAME,
        ac_limit_w=float(limit) if limit else default_limit,
    )
    return SystemSpec(
        site=build_site(entry),
        arrays=arrays,
        inverters={INVERTER_NAME: inverter},
        shading=shading or {},
    )


def load_all_shading(hass: HomeAssistant, entry: ConfigEntry) -> dict[str, TransmittanceGrid]:
    """Load every configured shading file.

    Blocking: reads from disk. Call from an executor.

    A file that fails to parse is skipped with a warning rather than failing
    setup. Losing shading degrades the forecast; failing setup loses it entirely,
    and the sensor that tells the user something is wrong along with it.
    """
    import logging  # noqa: PLC0415

    logger = logging.getLogger(__name__)
    directory = shading_directory(hass)
    grids: dict[str, TransmittanceGrid] = {}

    for subentry in entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_TYPE_ARRAY:
            continue
        filename = subentry.data.get(CONF_SHADING_FILE)
        if not filename:
            continue
        name = str(subentry.title)
        try:
            grids[name] = load_shading_file(directory / str(filename), name)
        except (ShadingFormatError, OSError) as err:
            logger.warning("Could not load shading for array %s: %s", name, err)

    return grids


def site_name(entry: ConfigEntry) -> str:
    """Return the site's display name."""
    return str(entry.data.get(CONF_NAME, "Solar"))
