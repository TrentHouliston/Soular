"""A coarse guard on how long a refresh takes.

The forecast runs in the executor, so this is not protecting the event loop --
that is already handled. What it protects against is the quantile path quietly
becoming the expensive part: fifteen trajectories through the power model is
fifteen times the arithmetic, and it would be easy to add a per-trajectory
solar-position call and not notice until a Raspberry Pi did.

The bound is deliberately an order of magnitude above the measured cost. A tight
timing assertion in CI is a flaky test; this one only fires on a regression
large enough to be structural.
"""

import time

from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

# Measured around 40 ms for a four-array site on a laptop, quantiles included.
BUDGET_SECONDS = 1.0


async def test_a_refresh_stays_cheap(hass: HomeAssistant, configured: MockConfigEntry) -> None:
    """Recomputing the whole forecast, quantiles and all, on four arrays."""
    coordinator = configured.runtime_data.coordinator
    assert coordinator is not None

    start = time.perf_counter()
    await coordinator.async_refresh()
    elapsed = time.perf_counter() - start

    assert coordinator.last_update_success
    assert elapsed < BUDGET_SECONDS, f"a refresh took {elapsed * 1000:.0f} ms"


async def test_quantiles_do_not_dominate(hass: HomeAssistant, configured: MockConfigEntry) -> None:
    """The band should cost a fraction of the forecast, not a multiple of it.

    Trajectories are scaled from the central result rather than re-run through
    the model precisely so this holds. If it stops holding, that shortcut has
    been undone somewhere.
    """
    coordinator = configured.runtime_data.coordinator
    assert coordinator is not None
    assert coordinator.uncertainty_available

    with_band = time.perf_counter()
    await coordinator.async_refresh()
    with_band = time.perf_counter() - with_band

    coordinator._ensemble = None
    without_band = time.perf_counter()
    await coordinator.async_refresh()
    without_band = time.perf_counter() - without_band

    assert not coordinator.uncertainty_available
    assert with_band < without_band * 4.0 + 0.05, (
        f"quantiles took the refresh from {without_band * 1000:.0f} ms to {with_band * 1000:.0f} ms"
    )
