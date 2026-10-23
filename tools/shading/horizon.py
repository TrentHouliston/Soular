"""Fit each array's horizon by gradient descent, judged on predicting its output.

Forward model, per clear sample of one array::

    T    = sigmoid((elevation - H(azimuth)) / width)    # soft shadow edge
    P    = eta * (diffuse + T * beam)

``H`` is a horizon at azimuth knots, circularly interpolated; ``width`` is how
soft the shadow edge is. The step "sun above or below the horizon" becomes a
sigmoid, so the whole thing is smooth in ``H`` and an optimiser can find it.

The horizon is not what ships. It is the reference line the graded map is
measured relative to (:mod:`tools.shading.grid`).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

KNOT_STEP_DEG = 3.0
# Taller than this implies a modelling error, not a tree.
MAX_HORIZON_DEG = 60.0
# Differences below this are "flat"; above it, an edge.
HUBER_EPS_DEG = 0.75
# Total variation lets the horizon sit flat at zero in open sky and step up at a
# tree, which is what a real horizon does. An L2 curvature penalty would pull it
# toward a sinusoid instead.
TOTAL_VARIATION = 0.03
# Prefer open sky wherever the data does not hold the horizon up. The pull fades
# with the evidence in each direction: at this many samples a knot carries half
# of it, so a well-observed obstruction is fitted rather than flattened, however
# narrow a share of the whole year's samples it is.
FLOOR_PULL = 0.04
FLOOR_PULL_HALF_SAMPLES = 50.0
INITIAL_HORIZON_DEG = 5.0
# A physical DC derate band for the pinned efficiency.
ETA_MIN = 0.80
ETA_MAX = 0.90


@dataclass(frozen=True, slots=True)
class Horizon:
    """A horizon line, plus the parameters fitted alongside it."""

    azimuth: np.ndarray
    elevation: np.ndarray
    eta: float
    width: float
    converged: bool

    def at(self, azimuth: np.ndarray) -> np.ndarray:
        """Horizon elevation at the given azimuths."""
        return np.interp(np.asarray(azimuth) % 360.0, self.azimuth, self.elevation, period=360.0)

    def to_text(self, step: float = 2.0) -> str:
        """Two-column azimuth/elevation text, as Home Assistant horizon files use."""
        az = np.arange(0.0, 360.0 + step / 2, step)
        return "".join(f"{a:.1f}\t{e:.2f}\n" for a, e in zip(az, self.at(az), strict=True))


def _pinned_eta(beam: np.ndarray, diffuse: np.ndarray, p: np.ndarray) -> float:
    """Efficiency from the brightest, least-shaded samples.

    Pinned before the fit, because otherwise the efficiency and the horizon are
    degenerate: the optimiser flattens the horizon and dims the efficiency
    instead, which erased a heavily shaded array entirely.
    """
    bright = beam > np.quantile(beam, 0.60)
    ratio = np.sort(p[bright] / np.maximum(diffuse[bright] + beam[bright], 1e-6))
    top = ratio[int(0.88 * len(ratio)) :]
    return float(np.clip(np.median(top), ETA_MIN, ETA_MAX))


def initial_horizon(samples: pd.DataFrame, eta: float) -> np.ndarray:
    """Read a starting horizon straight off the data, one value per knot.

    Per azimuth, the lowest sun elevation at which the measured beam gets through
    more than half the time. The fit is not convex, and starting it flat lets a
    heavily shaded array settle into a local minimum; this puts it near the right
    answer before it starts.
    """
    n_knots = round(360.0 / KNOT_STEP_DEG)
    az = samples["az"].to_numpy() % 360.0
    el = samples["el"].to_numpy()
    t = (samples["p"].to_numpy() - eta * samples["diffuse"].to_numpy()) / np.maximum(
        eta * samples["beam"].to_numpy(), 1e-6
    )
    knot = np.round(az / KNOT_STEP_DEG).astype(int) % n_knots
    start = np.zeros(n_knots)
    for k in range(n_knots):
        here = knot == k
        if here.sum() < 20:  # noqa: PLR2004 - too few samples to say anything
            start[k] = np.nan
            continue
        bins = np.floor(el[here] / 2.0).astype(int)
        medians = {b: np.median(t[here][bins == b]) for b in np.unique(bins) if (bins == b).sum() >= 4}  # noqa: PLR2004
        lit = [b for b, m in sorted(medians.items()) if m > 0.5]  # noqa: PLR2004 - "more often lit than not"
        start[k] = 2.0 * lit[0] if lit else np.nan
    known = np.isfinite(start)
    if not known.any():
        return np.full(n_knots, INITIAL_HORIZON_DEG)
    knots = np.arange(n_knots) * KNOT_STEP_DEG
    return np.interp(knots, knots[known], start[known], period=360.0)


def fit(samples: pd.DataFrame, seed: Horizon | None = None) -> Horizon:
    """Fit horizon knots and edge width to one array's clear samples."""
    n_knots = round(360.0 / KNOT_STEP_DEG)
    knot_az = np.arange(n_knots) * KNOT_STEP_DEG
    az = samples["az"].to_numpy()
    el = samples["el"].to_numpy()
    beam = samples["beam"].to_numpy()
    diffuse = samples["diffuse"].to_numpy()
    p = samples["p"].to_numpy()
    n = len(p)

    position = (az % 360.0) / KNOT_STEP_DEG
    k0 = np.floor(position).astype(int) % n_knots
    k1 = (k0 + 1) % n_knots
    frac = position - np.floor(position)

    eta = _pinned_eta(beam, diffuse, p)
    evidence = np.bincount(k0, weights=1 - frac, minlength=n_knots) + np.bincount(k1, weights=frac, minlength=n_knots)
    pull = FLOOR_PULL / (1.0 + evidence / FLOOR_PULL_HALF_SAMPLES)
    difference = np.roll(np.eye(n_knots), -1, 0) - np.eye(n_knots)
    eps = HUBER_EPS_DEG

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        h, width = x[:n_knots], x[n_knots]
        h_at = h[k0] * (1 - frac) + h[k1] * frac
        t = 1.0 / (1.0 + np.exp(-(el - h_at) / width))
        resid = eta * (diffuse + t * beam) - p
        dh = difference @ h
        loss = (
            np.mean(resid**2)
            + TOTAL_VARIATION * np.mean(np.sqrt(dh**2 + eps**2) - eps)
            + np.mean(pull * (np.sqrt(h**2 + eps**2) - eps))
        )
        dt = t * (1 - t)
        dresid = 2 * resid / n
        g_at = dresid * eta * beam * dt * (-1.0 / width)
        grad_h = np.zeros(n_knots)
        np.add.at(grad_h, k0, g_at * (1 - frac))
        np.add.at(grad_h, k1, g_at * frac)
        grad_h += TOTAL_VARIATION / n_knots * (difference.T @ (dh / np.sqrt(dh**2 + eps**2)))
        grad_h += pull / n_knots * (h / np.sqrt(h**2 + eps**2))
        grad_w = np.sum(dresid * eta * beam * dt * (-(el - h_at) / width**2))
        return float(loss), np.concatenate([grad_h, [grad_w]])

    start = seed.at(knot_az) if seed is not None else initial_horizon(samples, eta)
    # Width bounded to a few degrees: it should reflect the array's angular
    # spread and partial-shading softening, not act as a free dimmer.
    bounds = [(0.0, MAX_HORIZON_DEG)] * n_knots + [(1.0, 10.0)]
    result = minimize(
        objective,
        np.concatenate([start, [3.0]]),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-11},
    )
    return Horizon(
        azimuth=knot_az,
        elevation=result.x[:n_knots],
        eta=eta,
        width=float(result.x[n_knots]),
        converged=bool(result.success),
    )


def read_horizon(text: str) -> Horizon:
    """Parse a two-column horizon file, to seed a refit from a previous one."""
    rows = np.array([[float(v) for v in line.split()[:2]] for line in text.splitlines() if line.strip()])
    order = np.argsort(rows[:, 0] % 360.0)
    return Horizon(azimuth=rows[order, 0] % 360.0, elevation=rows[order, 1], eta=1.0, width=3.0, converged=True)
