"""The weather correction, through the integration.

The safety properties matter more than the accuracy ones here. A correction that
survives a change of what it means, or that is lost on restart, does damage that
is hard to notice and harder to attribute.
"""

from typing import Any

import numpy as np
import pytest

from custom_components.soular.core.weather_bias import BINS, WeatherBias
from custom_components.soular.learning.store import BASIS, decode, encode


def recorded(delivered: float = 0.8, days: int = 10) -> WeatherBias:
    """Build a record where the sky delivered ``delivered`` of the forecast."""
    bias = WeatherBias()
    for day in range(days):
        clearness = np.linspace(0.1, 1.0, 50)
        bias.add(1000 + day, clearness, np.full(50, 5000.0), np.full(50, 5000.0 * delivered), 1.0)
    return bias


class TestPersistence:
    """Round-tripping the record through storage."""

    def test_state_survives_a_round_trip(self) -> None:
        """What comes back corrects as what went in."""
        original = recorded()
        restored = decode(encode(original))
        np.testing.assert_allclose(restored.bin_ratios(1100), original.bin_ratios(1100), rtol=1e-6)
        assert restored.day_count == original.day_count

    def test_the_old_learner_is_discarded(self) -> None:
        """The per-array regression it replaced means nothing here, so it is dropped, not misread."""
        assert decode({"feature_basis_id": "v1", "arrays": {"east": {"theta": [0.0] * 8}}}).day_count == 0
        assert decode({"basis": "something-else", "weather": encode(recorded())["weather"]}).day_count == 0
        assert encode(recorded())["basis"] == BASIS

    def test_corrupt_state_is_skipped_not_fatal(self) -> None:
        """A truncated file costs the record, not the integration."""
        payload: dict[str, Any] = {"basis": BASIS, "weather": {"days": [[1000, [1.0], [2.0]], [1001, "x", None]]}}
        assert decode(payload).day_count == 0

    def test_nothing_stored_yields_nothing(self) -> None:
        """A first run starts with no record and no complaint."""
        assert decode(None).day_count == 0
        np.testing.assert_allclose(decode({}).bin_ratios(1), np.ones(BINS))


async def test_a_fresh_install_applies_no_correction(hass: Any, configured: Any) -> None:
    """With no measured-power sensor configured, nothing is learned or applied.

    The reference site's arrays have no power sensor in the fixture, so this also
    checks that the absence is handled as a missing feature rather than an error.
    """
    coordinator = configured.runtime_data.coordinator
    assert coordinator.weather_bias.day_count == 0
    assert coordinator.weather_correction_now == pytest.approx(1.0)
    assert coordinator.data.ac_power_w.max() > 0.0


async def test_learning_is_saved_when_home_assistant_stops(hass: Any, configured: Any, hass_storage: Any) -> None:
    """Stopping Home Assistant writes the learned state.

    Entries are not unloaded on shutdown, so an unload hook alone loses
    everything learned since the last write on every restart.
    """
    from datetime import timedelta  # noqa: PLC0415

    from homeassistant.const import EVENT_HOMEASSISTANT_STOP  # noqa: PLC0415
    from homeassistant.util import dt as dt_util  # noqa: PLC0415
    from pytest_homeassistant_custom_component.common import async_fire_time_changed  # noqa: PLC0415

    key = f"soular.learning.{configured.entry_id}"
    # Let any save scheduled at setup happen, so the one at stop is what counts.
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(minutes=10))
    await hass.async_block_till_done()
    configured.runtime_data.coordinator.weather_bias.days[1234] = ([1.0] * BINS, [42.0] * BINS)
    hass.bus.async_fire(EVENT_HOMEASSISTANT_STOP)
    await hass.async_block_till_done()
    # The write itself happens at the final-write stage, or here immediately.
    await hass.async_stop(force=True)
    assert hass_storage[key]["data"]["weather"]["days"][-1][2][0] == 42.0


async def test_learning_is_saved_while_running(hass: Any, configured: Any, hass_storage: Any) -> None:
    """The state is written periodically, from the event loop, not only once after setup.

    The thread check matters as much as the write: a save handler without
    ``@callback`` is run in an executor thread, where touching the store is
    unsafe, and whether the write then lands in time is a race the test would
    sometimes win.
    """
    from datetime import timedelta  # noqa: PLC0415
    import threading  # noqa: PLC0415
    from unittest.mock import patch  # noqa: PLC0415

    from homeassistant.helpers.storage import Store  # noqa: PLC0415
    from homeassistant.util import dt as dt_util  # noqa: PLC0415
    from pytest_homeassistant_custom_component.common import async_fire_time_changed  # noqa: PLC0415

    from custom_components.soular.learning.store import SAVE_INTERVAL  # noqa: PLC0415

    key = f"soular.learning.{configured.entry_id}"
    bias = configured.runtime_data.coordinator.weather_bias
    threads: set[int] = set()
    delay_save = Store.async_delay_save

    def spy(self: Any, *args: Any, **kwargs: Any) -> None:
        threads.add(threading.get_ident())
        delay_save(self, *args, **kwargs)

    with patch.object(Store, "async_delay_save", spy):
        for samples in (7, 9):
            bias.days[1234] = ([1.0] * BINS, [float(samples)] * BINS)
            async_fire_time_changed(hass, dt_util.utcnow() + SAVE_INTERVAL + timedelta(seconds=1))
            await hass.async_block_till_done()
            # The store writes on its own zero-delay timer, which is another tick.
            async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=1))
            await hass.async_block_till_done()
            assert hass_storage[key]["data"]["weather"]["days"][-1][2][0] == samples

    assert threads == {threading.get_ident()}, "saves ran off the event loop thread"


async def test_the_correction_scales_what_is_served(hass: Any, configured: Any) -> None:
    """A record of the forecast overshooting lowers every array's forecast by the learned ratio."""
    from homeassistant.util import dt as dt_util  # noqa: PLC0415

    coordinator = configured.runtime_data.coordinator
    before = coordinator.data.array("east").ac_power_w.copy()
    today = dt_util.now().date().toordinal()
    one_per_bin = np.array([0.1, 0.3, 0.5, 0.7, 0.8, 1.0])
    for day in range(1, 31):
        coordinator.weather_bias.add(today - day, one_per_bin, np.full(BINS, 50_000.0), np.full(BINS, 40_000.0), 1.0)
    expected = coordinator.weather_bias.bin_ratios(today)
    np.testing.assert_allclose(expected, expected[0])
    assert expected[0] < 0.95

    await coordinator.async_refresh()
    await hass.async_block_till_done()
    after = coordinator.data.array("east").ac_power_w
    lit = before > 100.0
    np.testing.assert_allclose(after[lit] / before[lit], expected[0], rtol=1e-6)
    state = hass.states.get("sensor.morisset_park_east_estimated_energy_production_tomorrow")
    assert state is not None
