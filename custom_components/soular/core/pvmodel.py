"""Module temperature, DC production and the inverter.

Two details here are worth more than they look.

**Wind height.** Faiman's convective coefficients were fitted against wind at
module height, and every weather API reports it at 10 m. Feeding 10 m wind
straight in cools the array by 2-3 degrees too much at noon -- a systematic ~0.8%
over-prediction. That is small, but it is exactly the kind of bias the online
correction would silently absorb into its efficiency term, which is where soiling
is supposed to show up. Correcting it here keeps the learned term interpretable.

**Where clipping happens.** Clipping is a limit on instantaneous power, so
applying it to an interval mean understates the energy lost inside the interval.
The pipeline therefore evaluates this model on its fine grid and aggregates
afterwards, never the reverse.
"""

import numpy as np
import pvlib

from custom_components.soular.core.types import ArraySpec, FloatArray, InverterSpec

# Roughness length for suburban terrain, used to bring 10 m wind to module height.
SURFACE_ROUGHNESS_M = 0.25
MODULE_HEIGHT_M = 3.0
MEASUREMENT_HEIGHT_M = 10.0


def wind_at_module_height(wind_speed_10m: FloatArray, factor: float) -> FloatArray:
    """Scale 10 m wind down to module height by a fixed log-law factor."""
    return np.asarray(np.clip(wind_speed_10m, 0.0, None) * factor, dtype=np.float64)


def default_wind_height_factor(
    roughness_m: float = SURFACE_ROUGHNESS_M,
    module_height_m: float = MODULE_HEIGHT_M,
    measurement_height_m: float = MEASUREMENT_HEIGHT_M,
) -> float:
    """Log-law ratio between module-height and 10 m wind speed.

    Provided so the 0.67 default in :class:`ArraySpec` is derived rather than a
    magic number, and so an exposed or rooftop install can recompute it.
    """
    return float(np.log(module_height_m / roughness_m) / np.log(measurement_height_m / roughness_m))


def cell_temperature(
    poa_global: FloatArray,
    temp_air: FloatArray,
    wind_speed_10m: FloatArray,
    array: ArraySpec,
) -> FloatArray:
    """Faiman cell temperature, using real ambient temperature and wind.

    The incumbent integration uses a fixed Ross coefficient with no wind term,
    which cannot distinguish a still 35 degree afternoon from a breezy one --
    worth several degrees of cell temperature and about 1.5% of power.
    """
    wind = wind_at_module_height(wind_speed_10m, array.wind_height_factor)
    return np.asarray(
        # pvlib annotates wind_speed as float; the model is elementwise and takes arrays.
        pvlib.temperature.faiman(poa_global, temp_air, wind),  # pyright: ignore[reportArgumentType]
        dtype=np.float64,
    )


def dc_power(poa_effective: FloatArray, cell_temp: FloatArray, array: ArraySpec) -> FloatArray:
    """DC power in watts from effective plane-of-array irradiance.

    PVWatts form: linear in irradiance with a temperature coefficient. No
    explicit low-light efficiency term -- low-light loss is a smooth monotone
    function of irradiance, which the learned correction's clear-sky-index basis
    already spans. Parameterising it twice would set the physics and the learned
    term fighting over one effect.
    """
    power = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=poa_effective,
        temp_cell=cell_temp,
        pdc0=array.dc_capacity_w,
        gamma_pdc=array.gamma_pdc,
    )
    derated = np.asarray(power, dtype=np.float64) * (1.0 - array.dc_loss_fraction)
    return np.asarray(np.clip(np.nan_to_num(derated, nan=0.0), 0.0, None), dtype=np.float64)


def ac_power(dc_power_w: FloatArray, inverter: InverterSpec) -> FloatArray:
    """Convert DC to AC, applying the inverter's efficiency curve and its limit.

    A hard ``min(dc, limit)`` -- what the incumbent does -- ignores that inverter
    efficiency falls away at low load and is not flat at high load either. The
    PVWatts curve is one parameter and captures both ends.
    """
    dc = np.asarray(np.clip(dc_power_w, 0.0, None), dtype=np.float64)
    if inverter.dc_limit_w is not None:
        dc = np.clip(dc, None, inverter.dc_limit_w)

    if inverter.model == "constant":
        return np.asarray(np.clip(dc * inverter.eta_nom, 0.0, inverter.ac_limit_w), dtype=np.float64)

    # pvlib parameterises its inverter by DC input rating rather than AC output,
    # and its efficiency curve is a function of load *fraction*. That cuts both
    # ways, and both ways end at zero: a rating far below the array drives the
    # -0.0162*zeta term negative, and a rating far above it drives the
    # -0.0059/zeta term negative. Either way pvlib floors the result. So
    # ac_limit_w has to be within about an order of magnitude of the array it
    # serves -- see system.build_system, which defaults it to the array capacity
    # rather than to something nominally unlimited.
    pdc0 = inverter.ac_limit_w / inverter.eta_nom

    # Cap DC at that rating *before* the curve. pvlib's efficiency polynomial is
    # only valid up to full load: past it the -0.0162*zeta term drives efficiency
    # negative, and pvlib floors the result at zero. So an array oversized enough
    # to overdrive its inverter would report no output at all, exactly at the
    # sunniest moment of the day. Capping is also what really happens -- an
    # inverter at its limit walks the MPPT off the maximum-power point rather
    # than accepting DC it cannot convert.
    dc = np.clip(dc, None, pdc0)

    result = pvlib.inverter.pvwatts(dc, pdc0, eta_inv_nom=inverter.eta_nom)
    return np.asarray(
        np.clip(np.nan_to_num(np.asarray(result, dtype=np.float64), nan=0.0), 0.0, None), dtype=np.float64
    )
