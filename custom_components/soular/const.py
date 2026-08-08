"""Constants for the Soular integration."""

from typing import Final

DOMAIN: Final = "soular"

# Site configuration keys. These describe the location every array shares; per-array
# geometry lives on the array subentries instead.
CONF_ELEVATION: Final = "elevation"
CONF_ALBEDO: Final = "albedo"
CONF_GROUND_TYPE: Final = "ground_type"

# Ground reflectance presets. The incumbent integration hardcodes 0.2 for everyone;
# exposing the choice matters most for steep tilts and snowy or bright surroundings.
GROUND_TYPE_ALBEDO: Final[dict[str, float]] = {
    "urban": 0.14,
    "grass": 0.20,
    "concrete": 0.30,
    "snow": 0.65,
}
DEFAULT_GROUND_TYPE: Final = "grass"
DEFAULT_ALBEDO: Final = GROUND_TYPE_ALBEDO[DEFAULT_GROUND_TYPE]
