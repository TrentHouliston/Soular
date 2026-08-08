"""Config flow for Soular.

Home Assistant discovers the flow class by importing ``config_flow`` from the
integration root, so this module stays a thin re-export and the real flows live in
``flows/`` where they can be split per step without a circular import.
"""

from .flows.hub import HubConfigFlow

SoularConfigFlow = HubConfigFlow

__all__ = ["SoularConfigFlow"]
