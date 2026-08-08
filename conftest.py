"""Test configuration and fixtures.

The repo root goes on ``sys.path`` so tests can import ``custom_components.soular``
the same way Home Assistant does at runtime.
"""

from logging import config as logging_config_module
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))

# Enable custom component for testing
pytest_plugins = ["pytest_homeassistant_custom_component"]


@pytest.fixture(autouse=True, scope="session")
def configure_logging() -> None:
    """Quiet Home Assistant's own DEBUG chatter so test output stays readable."""
    logging_config_module.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"brief": {"format": "%(levelname)s: %(name)s: %(message)s"}},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "brief",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                "homeassistant.core": {"level": "WARNING", "handlers": ["console"], "propagate": False},
                "custom_components.soular": {"level": "INFO", "handlers": ["console"], "propagate": False},
            },
            "root": {"level": "WARNING", "handlers": ["console"]},
        }
    )


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations: None) -> bool:
    """Enable loading custom integrations in all tests."""
    return enable_custom_integrations is None
