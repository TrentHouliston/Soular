"""Diagnostics dump for a config entry.

Aimed at answering the question a bug report actually needs: what geometry was
configured, did the shading load, how fresh is the weather, and what does the
forecast currently look like.
"""

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
import numpy as np

from custom_components.soular import SoularConfigEntry
from custom_components.soular.const import SUBENTRY_TYPE_ARRAY
from custom_components.soular.system import build_system

# The full series is hundreds of points. A summary answers "is this plausible?"
# without making the report unreadable.
SUMMARY_KEYS = ("min", "max", "mean")


def _summary(values: np.ndarray) -> dict[str, float]:
    """Reduce a series to something a human can scan."""
    if values.size == 0:
        return dict.fromkeys(SUMMARY_KEYS, 0.0)
    return {
        "min": round(float(np.min(values)), 2),
        "max": round(float(np.max(values)), 2),
        "mean": round(float(np.mean(values)), 2),
    }


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant,  # noqa: ARG001 - fixed by the diagnostics platform signature
    entry: SoularConfigEntry,
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    runtime = entry.runtime_data
    system = build_system(entry, runtime.shading if runtime else {})

    arrays = [
        {
            "name": array.name,
            "azimuth_deg": array.azimuth_deg,
            "tilt_deg": array.tilt_deg,
            "dc_capacity_w": array.dc_capacity_w,
            "gamma_pdc": array.gamma_pdc,
            "dc_loss_fraction": array.dc_loss_fraction,
            "shading_loaded": array.name in system.shading,
            "panel_efficiency": system.shading[array.name].efficiency_for(array.dc_loss_fraction)
            if array.name in system.shading
            else None,
        }
        for array in system.arrays
    ]

    diagnostics: dict[str, Any] = {
        # Coordinates are the one thing here worth redacting: everything else is
        # equipment configuration.
        "site": {
            "elevation_m": system.site.elevation_m,
            "albedo": system.site.albedo,
            "transposition_model": system.site.transposition_model,
        },
        "arrays": arrays,
        "configured_array_count": sum(
            1 for subentry in entry.subentries.values() if subentry.subentry_type == SUBENTRY_TYPE_ARRAY
        ),
        "inverters": {
            name: {"ac_limit_w": inverter.ac_limit_w, "model": inverter.model}
            for name, inverter in system.inverters.items()
        },
    }

    coordinator = runtime.coordinator if runtime else None
    if coordinator is not None:
        diagnostics["coordinator"] = {
            "last_update_success": coordinator.last_update_success,
            "weather_fetched_at": coordinator.weather_fetched_at.isoformat()
            if coordinator.weather_fetched_at
            else None,
            "weather_error": coordinator.weather_error,
        }
        today = dt_util.now().date().toordinal()
        diagnostics["weather_correction"] = {
            "days": coordinator.weather_bias.day_count,
            "prior": coordinator.weather_bias.prior,
            "by_forecast_clearness": [round(float(r), 3) for r in coordinator.weather_bias.bin_ratios(today)],
        }
        diagnostics["band"] = [
            {"lead": lead, "p10": round(b.low, 3), "p90": round(b.high, 3), "samples": b.samples}
            for lead, b in enumerate(coordinator.band_bounds)
        ]
        result = coordinator.data
        if coordinator.last_update_success:
            diagnostics["forecast"] = {
                "steps": int(result.times.size),
                "from": str(result.times[0]),
                "to": str(result.times[-1]),
                "site_ac_power_w": _summary(result.ac_power_w),
                "arrays": {
                    entry_result.name: {
                        "dc_power_w": _summary(entry_result.dc_power_w),
                        "poa_global": _summary(entry_result.poa_global),
                        "transmittance": _summary(entry_result.transmittance),
                    }
                    for entry_result in result.arrays
                },
            }

    return diagnostics
