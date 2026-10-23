"""A coarse guard on how long a refresh takes.

The forecast runs in the executor, so this is not protecting the event loop --
that is already handled. What it protects against is a structural slowdown --
an extra solar-position call per array, say -- that nobody notices until a
Raspberry Pi does.

The bound is deliberately an order of magnitude above the measured cost. A tight
timing assertion in CI is a flaky test; this one only fires on a regression
large enough to be structural.
"""

import time

from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

# Measured well under 40 ms for a four-array site on a laptop.
BUDGET_SECONDS = 1.0


async def test_a_refresh_stays_cheap(hass: HomeAssistant, configured: MockConfigEntry) -> None:
    """Recomputing the whole forecast on four arrays."""
    coordinator = configured.runtime_data.coordinator
    assert coordinator is not None

    start = time.perf_counter()
    await coordinator.async_refresh()
    elapsed = time.perf_counter() - start

    assert coordinator.last_update_success
    assert elapsed < BUDGET_SECONDS, f"a refresh took {elapsed * 1000:.0f} ms"
