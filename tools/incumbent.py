"""A faithful emulation of ``rany2/ha-open-meteo-solar-forecast``.

Written from the upstream source so the comparison is against what that
integration actually does, not against a sketch of it. Everything here is
deliberately *not* shared with soular's own model: the point is to reproduce a
different set of choices, including the ones this project considers mistakes.

What it reproduces, and why each matters:

* **It does not transpose.** It asks Open-Meteo for ``global_tilted_irradiance``
  with a tilt and azimuth and consumes the answer, so the plane-of-array model is
  Open-Meteo's -- isotropic sky, albedo fixed at 0.2. This emulation consumes the
  same numbers from the archive rather than recomputing them, which is the single
  most important thing to get right.
* **A Ross thermal coefficient with no wind term**, hardcoded to "not so well
  cooled". It cannot tell a still 35-degree afternoon from a breezy one.
* **Hard horizon shading**, with an optional "partial shading" mode that
  substitutes *horizontal* diffuse for plane-of-array irradiance when blocked --
  no sky-view scaling, no ground term.
* **Clipping applied to fifteen-minute means**, which understates the energy lost
  inside each interval.
* **The fifteen-minute labelling offset.** Power is accumulated under
  ``time - 15min`` while its irradiance is taken at ``time`` and its temperature
  at ``time - 15min``. Irradiance and temperature come from opposite ends of the
  same interval and the result is filed at the start, so the instantaneous power
  series leads reality by about a quarter hour.

Not reproduced: the damping factors, which default to zero and which this site
does not set, and the efficiency factor, which is a constant and is therefore
absorbed by the harness's per-variant fitted gain either way.
"""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

# Upstream constants, verbatim from its constants.py.
G_STC = 1000.0
TEMP_STC_CELL = 25.0
ALPHA_TEMP = -0.004
ROSS_NOT_SO_WELL_COOLED = 0.0342

# Upstream's own default. A constant, so the harness's fitted gain absorbs it;
# carried anyway so the emulation matches the source line for line.
DEFAULT_EFFICIENCY = 1.0

# The integration works on a quarter-hour grid.
STEP_SECONDS = 900

# A hard horizon is lit or it is not.
HORIZON_LIT = 0.5


@dataclass(frozen=True, slots=True)
class IncumbentArray:
    """One array as the incumbent integration is configured for it."""

    name: str
    dc_watts: float
    # Compass degrees from north. Converted to Open-Meteo's south-based
    # convention at fetch time, exactly as the upstream coordinator does.
    azimuth_deg: float
    tilt_deg: float


class TiltedArchive:
    """Archived plane-of-array irradiance, as the incumbent would have received it."""

    def __init__(self, path: Path) -> None:
        """Load the tilted-irradiance archive into memory."""
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.tables: dict[tuple[str, int, str], dict[np.datetime64, float]] = {}
        for array_name, stamp, lead, variable, value in connection.execute(
            "SELECT array_name, valid_utc, lead_day, variable, value FROM gti"
        ):
            self.tables.setdefault((array_name, lead, variable), {})[np.datetime64(stamp[:19], "s")] = value
        self.leads = sorted({lead for _, lead, _ in self.tables})
        connection.close()

    def available(self) -> bool:
        """Report whether any tilted irradiance was fetched."""
        return bool(self.tables)

    def series(self, array_name: str, variable: str, lead: int, times: np.ndarray) -> np.ndarray:
        """Interpolate one hourly variable onto ``times``.

        Linear interpolation, which is what Open-Meteo itself does to produce the
        fifteen-minute series the integration requests for any model without a
        native sub-hourly resolution -- which is every model covering this site.
        """
        table = self.tables.get((array_name, min(lead, max(self.leads)) if self.leads else lead, variable))
        if not table:
            return np.full(times.size, np.nan)
        stamps = np.array(sorted(table), dtype="datetime64[s]")
        values = np.array([table[stamp] for stamp in stamps], dtype=np.float64)
        return np.asarray(
            np.interp(
                times.astype("datetime64[s]").astype(np.int64).astype(np.float64),
                stamps.astype(np.int64).astype(np.float64),
                values,
                left=np.nan,
                right=np.nan,
            ),
            dtype=np.float64,
        )


def forecast_site(
    arrays: list[IncumbentArray],
    archive: TiltedArchive,
    times: np.ndarray,
    leads: np.ndarray,
    blocked: dict[str, np.ndarray],
    ac_limit_w: float,
    *,
    partial_shading: bool = True,
    efficiency: float = DEFAULT_EFFICIENCY,
    label_offset: bool = True,
) -> np.ndarray:
    """Run the incumbent's model for a whole site, returning AC watts.

    ``times`` must be a regular quarter-hour grid, because the integration's
    clipping and labelling both operate on one.
    """
    total = np.zeros(times.size, dtype=np.float64)

    for array in arrays:
        gti = np.zeros(times.size)
        diffuse = np.zeros(times.size)
        direct = np.zeros(times.size)
        temperature = np.full(times.size, np.nan)

        # One interpolation per distinct lead, so each block comes from a single run.
        for lead in np.unique(leads).astype(int):
            block = leads == lead
            gti[block] = archive.series(array.name, "global_tilted_irradiance", int(lead), times[block])
            diffuse[block] = archive.series(array.name, "diffuse_radiation", int(lead), times[block])
            direct[block] = archive.series(array.name, "direct_radiation", int(lead), times[block])
            temperature[block] = archive.series(array.name, "temperature_2m", int(lead), times[block])

        gti = np.nan_to_num(gti, nan=0.0)
        diffuse = np.nan_to_num(diffuse, nan=0.0)
        direct = np.nan_to_num(direct, nan=0.0)
        temperature = np.nan_to_num(temperature, nan=20.0)

        # Upstream: when blocked, substitute *horizontal* diffuse for the tilted
        # total, optionally scaled by the diffuse fraction. No sky-view factor,
        # no ground-reflected term.
        shade = blocked.get(array.name, np.zeros(times.size, dtype=bool))
        if partial_shading:
            denominator = diffuse + direct
            fraction = np.where(denominator > 0.0, np.clip(diffuse / np.maximum(denominator, 1e-9), 0.0, None), 1.0)
        else:
            fraction = np.ones(times.size)
        irradiance = np.where(shade, diffuse * fraction, gti)

        cell = temperature + irradiance * ROSS_NOT_SO_WELL_COOLED
        power = array.dc_watts * (irradiance / G_STC) * (1.0 + ALPHA_TEMP * (cell - TEMP_STC_CELL)) * efficiency
        total += np.clip(np.round(power), 0.0, None)

    if label_offset:
        # Upstream accumulates power computed from irradiance at ``time`` under
        # the key ``time - 15min``, so the series it serves leads reality by a
        # quarter hour. Reproduced because it is what the integration actually
        # publishes; ``label_offset=False`` exists to measure how much of the gap
        # is this rather than the model.
        total = np.concatenate([total[1:], total[-1:]])

    # Upstream clips the quarter-hour series, so clipping never sees the peaks
    # inside an interval.
    return np.clip(total, 0.0, ac_limit_w)
