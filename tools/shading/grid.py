"""The graded transmittance field: measured per sky cell, not a single edge.

Built in three layers, each validated against the replay:

1. A smooth prior. Median beam transmittance as a function of how far the sun
   is above this array's horizon, forced monotonic and saturating -- 0 below the
   horizon, the measured canopy fall-off through it, 1 well above.
2. The empirical field. Median transmittance per 2 x 2 degree cell, blended with
   the prior by sample count, so sparse or lightly shaded cells fall back to the
   curve instead of chasing noise.
3. Normalisation. Open sky is divided to exactly 1, leaving the residual
   efficiency (soiling, spectral) for the forecaster's learned gain where a
   constant belongs.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from tools.shading.horizon import Horizon

# Output grid: 1 degree everywhere, as the integration's loader expects.
AZIMUTH_AXIS = np.arange(0.0, 361.0, 1.0)
ELEVATION_AXIS = np.arange(0.0, 91.0, 1.0)

# Empirical cells, and how much evidence a cell needs before it outweighs the prior.
CELL_AZ = 2.0
CELL_EL = 2.0
CELL_EL_MAX = 62.0
MIN_CELL_SAMPLES = 8
BLEND_K = 25.0

# Transition curve: elevation above the horizon, in degrees.
TRANSITION_LOW = -14.0
TRANSITION_HIGH = 40.0
TRANSITION_STEP = 2.0
TRANSITION_MIN_SAMPLES = 30
SATURATE_AT = 0.97

# Cells this far above the horizon define open sky for normalisation.
OPEN_SKY_MARGIN_DEG = 20.0


def _transmittance(samples: pd.DataFrame, eta: float) -> np.ndarray:
    """Measured beam transmittance: what the beam delivered over what it could."""
    return (samples["p"].to_numpy() - eta * samples["diffuse"].to_numpy()) / np.maximum(
        eta * samples["beam"].to_numpy(), 1e-6
    )


def _open_sky_eta(samples: pd.DataFrame) -> float:
    """Measure the array's true gain in beam-dominated sky.

    Pinning this, rather than a hard derate, keeps measured transmittance from
    being pushed past 1 and clipped, which flattened real shape in an earlier map.
    """
    total = (samples["beam"] + samples["diffuse"]).to_numpy()
    clean = samples["beam"].to_numpy() > 0.6 * np.maximum(total, 1e-6)
    return float(np.quantile(samples["p"].to_numpy()[clean] / np.maximum(total[clean], 1e-6), 0.90))


def transition(samples: pd.DataFrame, horizon: Horizon) -> tuple[np.ndarray, np.ndarray]:
    """Median transmittance against elevation above the horizon."""
    delta = samples["el"].to_numpy() - horizon.at(samples["az"].to_numpy())
    t = _transmittance(samples, horizon.eta)
    edges = np.arange(TRANSITION_LOW, TRANSITION_HIGH + TRANSITION_STEP, TRANSITION_STEP)
    centres = edges[:-1] + TRANSITION_STEP / 2
    curve = np.full(centres.size, np.nan)
    for i in range(centres.size):
        inside = (delta >= edges[i]) & (delta < edges[i + 1]) & np.isfinite(t)
        if inside.sum() >= TRANSITION_MIN_SAMPLES:
            curve[i] = np.median(np.clip(t[inside], 0.0, 1.2))
    known = np.isfinite(curve)
    curve = np.maximum.accumulate(np.clip(np.interp(centres, centres[known], curve[known]), 0.0, 1.0))
    if (curve >= SATURATE_AT).any():
        curve[np.argmax(curve >= SATURATE_AT) :] = 1.0
    return centres, curve


def build(samples: pd.DataFrame, horizon: Horizon) -> np.ndarray:
    """Build the graded field for one array, ``[azimuth, elevation]`` on the output axes."""
    centres, curve = transition(samples, horizon)
    cell_az = np.arange(0.0, 360.0, CELL_AZ)
    cell_el = np.arange(0.0, CELL_EL_MAX, CELL_EL)
    above = cell_el[None, :] - horizon.at(cell_az)[:, None]
    prior = np.interp(above, centres, curve, left=0.0, right=1.0)

    eta = _open_sky_eta(samples)
    az = samples["az"].to_numpy()
    el = samples["el"].to_numpy()
    t = _transmittance(samples, eta)
    ok = np.isfinite(t)
    az, el, t = az[ok], el[ok], np.clip(t[ok], 0.0, 1.15)

    field = prior.copy()
    ai = np.clip((az // CELL_AZ).astype(int), 0, cell_az.size - 1)
    ei = np.clip((el // CELL_EL).astype(int), 0, cell_el.size - 1)
    for a in range(cell_az.size):
        in_column = ai == a
        for e in range(cell_el.size):
            values = t[in_column & (ei == e)]
            if values.size >= MIN_CELL_SAMPLES:
                weight = values.size / (values.size + BLEND_K)
                field[a, e] = weight * np.median(values) + (1.0 - weight) * prior[a, e]

    # Azimuth wraps around the compass; elevation does not.
    smooth = gaussian_filter(
        np.clip(field, 0.0, 1.15),
        sigma=(1.2, 1.0),
        mode=("wrap", "nearest"),  # pyright: ignore[reportArgumentType] - scipy accepts one mode per axis
    )
    # Divide out the open-sky efficiency, measured only well above this array's
    # own horizon so it removes an efficiency and never a real obstruction.
    open_sky = above > OPEN_SKY_MARGIN_DEG
    efficiency = float(np.median(smooth[open_sky])) if open_sky.any() else 1.0
    smooth = np.clip(smooth / max(efficiency, 0.5), 0.0, 1.0)

    interpolate = RegularGridInterpolator(
        (np.append(cell_az, 360.0), cell_el),
        np.vstack([smooth, smooth[:1]]),
    )
    grid_az, grid_el = np.meshgrid(AZIMUTH_AXIS, np.clip(ELEVATION_AXIS, cell_el[0], cell_el[-1]), indexing="ij")
    return np.clip(interpolate((grid_az % 360.0, grid_el)), 0.0, 1.0)
