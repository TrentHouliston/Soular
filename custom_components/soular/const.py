"""Constants for the Soular integration."""

from datetime import timedelta
from typing import Final

DOMAIN: Final = "soular"

# Site configuration keys. These describe the location every array shares; per-array
# geometry lives on the array subentries instead.
CONF_ELEVATION: Final = "elevation"
CONF_ALBEDO: Final = "albedo"
CONF_GROUND_TYPE: Final = "ground_type"
CONF_INVERTER_AC_LIMIT: Final = "inverter_ac_limit"

# Array subentry keys.
SUBENTRY_TYPE_ARRAY: Final = "array"
CONF_AZIMUTH: Final = "azimuth"
CONF_TILT: Final = "tilt"
CONF_DC_CAPACITY: Final = "dc_capacity"
CONF_TEMPERATURE_COEFFICIENT: Final = "temperature_coefficient"
CONF_DC_LOSS: Final = "dc_loss"
CONF_SHADING_FILE: Final = "shading_file"
# Optional per-array measured output. Without one the array cannot be calibrated,
# which is a loss of accuracy rather than a failure.
CONF_POWER_SENSOR: Final = "power_sensor"

# Site-level inputs that sharpen curtailment detection. Both optional: the mask
# has rules that need no configuration at all.
CONF_BATTERY_SOC_SENSOR: Final = "battery_soc_sensor"

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

# Datasheet default for a modern mono-PERC module, -0.35 %/degC.
DEFAULT_TEMPERATURE_COEFFICIENT: Final = -0.35
# PVWatts' combined soiling, wiring, mismatch and connection default. Deliberately
# one number: the parts are not separately identifiable from production data.
DEFAULT_DC_LOSS_PERCENT: Final = 14.0

# Shading files live under <config>/soular/ so a user drops them somewhere obvious
# and no absolute path ends up in a config entry.
SHADING_DIRECTORY: Final = "soular"

# Open-Meteo publishes new runs hourly at best, so polling faster only burns quota.
# The forecast itself is recomputed far more often than this, because the sun moves
# continuously even when the weather does not.
UPDATE_INTERVAL: Final = timedelta(minutes=30)
RECOMPUTE_INTERVAL: Final = timedelta(minutes=5)
# The satellite publishes every ten minutes with about half an hour of latency,
# so polling faster than its cadence cannot learn anything new.
SATELLITE_INTERVAL: Final = timedelta(minutes=10)
# Ensembles run every six hours at best and are three-hourly in output, so
# hourly polling already outruns anything new they have to say.
ENSEMBLE_INTERVAL: Final = timedelta(hours=1)

# The learner samples measured output at the recompute cadence. Samples closer
# together than this are more correlated than informative.
LEARNING_SAMPLE_INTERVAL: Final = timedelta(minutes=5)
