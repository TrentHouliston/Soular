"""Tests for the thermal, DC and inverter models."""

import numpy as np
import pytest

from custom_components.soular.core.pvmodel import (
    ac_power,
    cell_temperature,
    dc_power,
    default_wind_height_factor,
    wind_at_module_height,
)
from custom_components.soular.core.types import ArraySpec, InverterSpec

ARRAY = ArraySpec(name="east", azimuth_deg=84.0, tilt_deg=25.0, dc_capacity_w=7920.0)
STC = np.array([1000.0])


def test_default_wind_factor_matches_the_spec_default() -> None:
    """The 0.67 in ArraySpec is a log-law result, not a magic number.

    Ignoring the 10 m to module-height conversion cools the array 2-3 degrees too
    much at noon, a systematic ~0.8% over-prediction that the online correction
    would otherwise absorb into its efficiency term -- which is where soiling is
    meant to show up.
    """
    assert default_wind_height_factor() == pytest.approx(ARRAY.wind_height_factor, abs=0.01)
    assert 0.0 < default_wind_height_factor() < 1.0


def test_wind_is_never_negative() -> None:
    """A bad reading cannot become negative convective cooling."""
    result = wind_at_module_height(np.array([-3.0, 0.0, 5.0]), 0.67)
    assert np.all(result >= 0.0)


def test_wind_cools_the_cell() -> None:
    """Faiman responds to wind, which a fixed Ross coefficient cannot."""
    poa = np.array([900.0])
    air = np.array([35.0])
    still = cell_temperature(poa, air, np.array([0.0]), ARRAY)
    breezy = cell_temperature(poa, air, np.array([8.0]), ARRAY)

    assert breezy[0] < still[0]
    assert still[0] - breezy[0] > 5.0, "8 m/s should be worth several degrees"


def test_cell_is_never_cooler_than_the_air_in_sun() -> None:
    """Irradiance heats the module; it cannot refrigerate it."""
    air = np.array([10.0, 20.0, 30.0])
    poa = np.array([200.0, 600.0, 1000.0])
    result = cell_temperature(poa, air, np.full(3, 2.0), ARRAY)
    assert np.all(result >= air)


def test_dc_power_at_stc_is_capacity_less_losses() -> None:
    """1000 W/m2 at 25 degrees gives nameplate, minus the DC loss fraction."""
    result = dc_power(STC, np.array([25.0]), ARRAY)
    assert result[0] == pytest.approx(ARRAY.dc_capacity_w * (1.0 - ARRAY.dc_loss_fraction))


def test_dc_power_falls_with_temperature() -> None:
    """The temperature coefficient has the sign and magnitude on the datasheet."""
    cool = dc_power(STC, np.array([25.0]), ARRAY)[0]
    hot = dc_power(STC, np.array([55.0]), ARRAY)[0]

    expected_ratio = 1.0 + ARRAY.gamma_pdc * 30.0
    assert hot < cool
    assert hot / cool == pytest.approx(expected_ratio, rel=1e-9)


def test_dc_power_is_never_negative() -> None:
    """A very hot cell in near-darkness must not produce negative power."""
    result = dc_power(np.array([0.0, 1.0]), np.array([90.0, 90.0]), ARRAY)
    assert np.all(result >= 0.0)


def test_inverter_clips_at_its_ac_limit() -> None:
    """Output saturates at the AC limit no matter how much DC arrives."""
    inverter = InverterSpec(name="default", ac_limit_w=10000.0)
    result = ac_power(np.array([5000.0, 10000.0, 20000.0, 1e9]), inverter)
    assert np.all(result <= inverter.ac_limit_w + 1e-6)
    assert result[-1] == pytest.approx(inverter.ac_limit_w, rel=1e-6)


def test_inverter_efficiency_is_worse_at_low_load() -> None:
    """The PVWatts curve captures the low-load rolloff a hard min() cannot.

    This is why a plain clip is not good enough: an array spends a lot of its
    generating hours well below the inverter's rating.
    """
    inverter = InverterSpec(name="default", ac_limit_w=10000.0)
    dc = np.array([500.0, 5000.0])
    result = ac_power(dc, inverter)
    efficiency = result / dc
    assert efficiency[0] < efficiency[1]
    assert efficiency[1] < 1.0


def test_constant_model_is_a_flat_efficiency_and_a_clip() -> None:
    """The simple model is available for users who only know one number."""
    inverter = InverterSpec(name="default", ac_limit_w=10000.0, eta_nom=0.95, model="constant")
    result = ac_power(np.array([1000.0, 20000.0]), inverter)
    assert result[0] == pytest.approx(950.0)
    assert result[1] == pytest.approx(10000.0)


def test_dc_input_limit_is_applied_before_conversion() -> None:
    """An MPPT input limit truncates DC before the inverter ever sees it."""
    inverter = InverterSpec(name="default", ac_limit_w=1e9, dc_limit_w=6000.0)
    result = ac_power(np.array([6000.0, 12000.0]), inverter)
    assert result[0] == pytest.approx(result[1])
