"""Turn ensemble spread into forecast quantiles.

A battery optimiser cannot act on a point estimate. It needs to know whether
tomorrow is 40 kWh give or take two, or 40 kWh give or take twenty, because those
call for different decisions today. Ensemble members are the only source of that
information available, and converting them into something usable takes three
steps, the last of which is where the obvious approach silently produces wrong
numbers.

**One: members become clear-sky indices.** Same reason as everywhere else -- the
index is smooth and bounded, and differencing against a common clear-sky removes
the part of the members' disagreement that is nobody's forecast error.

**Two: extract the spread as ratios, and interpolate the ratios.** Ensembles here
are three-hourly, the forecast grid is five-minutely. Interpolating individual
members manufactures implausibly smooth trajectories; interpolating the *shape*
of the distribution does not. The median is then replaced with the blended
deterministic forecast, so the nowcast and the correction still drive the central
estimate and the ensemble only supplies the width.

**Three: energy quantiles need trajectories, not pointwise quantiles.** This is
the one that matters. ``P90`` of a day's energy is not the integral of the
pointwise ``P90`` of power. Summing pointwise P90s assumes every hour is at its
90th percentile simultaneously -- perfect rank correlation. Taking the root sum of
squares assumes the hours are independent. Both are wrong, in opposite directions,
by roughly a factor of two on a full day's band width. Ensemble copula coupling
fixes it by keeping each member's *rank trajectory* and reconstructing whole
plausible days, which are then pushed through the full power model.

That last error fails quietly: it produces a plausible-looking number that is
confidently too narrow or too wide, and nothing anywhere complains.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np

from custom_components.soular.core.clearsky import K_MAX, K_MIN
from custom_components.soular.core.series import energy_between
from custom_components.soular.core.types import FloatArray, TimeArray

# What the integration publishes. P50 is the central estimate; the outer pair is
# what a battery plan hedges against.
DEFAULT_QUANTILES: Final = (0.1, 0.5, 0.9)

# Below this the median is too close to zero for a ratio to be meaningful and the
# spread is carried additively instead.
MIN_RATIO_MEDIAN: Final = 0.15

# Members are thinned to this many trajectories before being pushed through the
# power model. Fifty-one adds nothing a systematic rank sample of fifteen misses,
# and the model runs fifteen times instead.
DEFAULT_TRAJECTORIES: Final = 15


@dataclass(frozen=True, slots=True)
class EnsembleSpread:
    """Quantile ratios of the clear-sky index, on the ensemble's own time grid."""

    times: TimeArray
    quantiles: tuple[float, ...]
    # Shape (quantile, time). Multiplicative against the median where it is
    # meaningful, additive where the median is near zero.
    ratios: FloatArray
    offsets: FloatArray
    # Rank of each member at each time, normalised to (0, 1), shape (member, time).
    ranks: FloatArray


def spread_from_members(
    times: TimeArray,
    member_k: FloatArray,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> EnsembleSpread:
    """Summarise members into quantile ratios and rank trajectories.

    ``member_k`` is shape (member, time), in clear-sky index.
    """
    if member_k.ndim != 2:  # noqa: PLR2004
        msg = f"member_k must be (member, time); got shape {member_k.shape}"
        raise ValueError(msg)
    if member_k.shape[1] != times.size:
        msg = f"member_k has {member_k.shape[1]} times but times has {times.size}"
        raise ValueError(msg)

    levels = np.quantile(member_k, quantiles, axis=0)
    median = np.quantile(member_k, 0.5, axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = np.where(median > MIN_RATIO_MEDIAN, levels / np.maximum(median, 1e-9), 1.0)
    offsets = np.where(median > MIN_RATIO_MEDIAN, 0.0, levels - median)

    # Rank within the ensemble at each time, as a probability level. Kept per
    # member so a member that is gloomy all morning stays gloomy all morning.
    order = np.argsort(np.argsort(member_k, axis=0), axis=0)
    ranks = (order + 0.5) / member_k.shape[0]

    return EnsembleSpread(
        times=times,
        quantiles=tuple(quantiles),
        ratios=np.asarray(ratios, dtype=np.float64),
        offsets=np.asarray(offsets, dtype=np.float64),
        ranks=np.asarray(ranks, dtype=np.float64),
    )


def _interpolate(times: TimeArray, values: FloatArray, grid: TimeArray) -> FloatArray:
    """Interpolate rows of ``values`` from ``times`` onto ``grid``."""
    source = times.astype("datetime64[s]").astype(np.int64).astype(np.float64)
    target = grid.astype("datetime64[s]").astype(np.int64).astype(np.float64)
    return np.vstack([np.interp(target, source, row) for row in np.atleast_2d(values)])


def quantile_indices(
    spread: EnsembleSpread,
    grid: TimeArray,
    median_k: FloatArray,
    observed_share: FloatArray | None = None,
) -> FloatArray:
    """Apply the ensemble's shape to a deterministic median, on the fine grid.

    Where an observation has anchored the near term, the spread is tapered toward
    the median in proportion to how much the observation contributed: a forecast
    pinned to a satellite reading twenty minutes old really is more certain than
    the raw ensemble spread implies, and leaving it untapered would publish a
    confidence band that is honest about the model and wrong about reality.
    """
    ratios = _interpolate(spread.times, spread.ratios, grid)
    offsets = _interpolate(spread.times, spread.offsets, grid)

    if observed_share is not None:
        keep = np.clip(1.0 - np.asarray(observed_share, dtype=np.float64), 0.0, 1.0)
        ratios = 1.0 + (ratios - 1.0) * keep
        offsets = offsets * keep

    result = np.clip(ratios * median_k[None, :] + offsets, K_MIN, K_MAX)
    return np.asarray(result, dtype=np.float64)


def trajectories(
    spread: EnsembleSpread,
    grid: TimeArray,
    median_k: FloatArray,
    count: int = DEFAULT_TRAJECTORIES,
    observed_share: FloatArray | None = None,
) -> FloatArray:
    """Reconstruct whole plausible days by ensemble copula coupling.

    Each member keeps its rank trajectory, and that rank is looked up in the
    calibrated quantile function at each instant. The result is a set of
    coherent days -- a gloomy morning followed by a gloomy afternoon, rather than
    an average of everything -- which is what makes integrating them into energy
    quantiles meaningful.

    Ranks are nearest-neighbour in time, never interpolated: a rank is a
    categorical position within the ensemble, and averaging two of them produces
    a member that does not exist.
    """
    levels = np.linspace(0.05, 0.95, 19)
    member_count = spread.ranks.shape[0]
    chosen = np.linspace(0, member_count - 1, min(count, member_count)).round().astype(int)

    source = spread.times.astype("datetime64[s]").astype(np.int64).astype(np.float64)
    target = grid.astype("datetime64[s]").astype(np.int64).astype(np.float64)
    nearest = np.abs(target[:, None] - source[None, :]).argmin(axis=1)

    # Quantile function on the fine grid, at a fixed ladder of probability levels.
    ladder = spread_ladder(spread, grid, median_k, levels, observed_share)

    out = np.empty((chosen.size, grid.size), dtype=np.float64)
    for row, member in enumerate(chosen):
        member_ranks = spread.ranks[member][nearest]
        # Which ladder rung each rank corresponds to, per instant.
        rung = np.clip(np.searchsorted(levels, member_ranks) - 1, 0, levels.size - 1)
        out[row] = ladder[rung, np.arange(grid.size)]
    return out


def spread_ladder(
    spread: EnsembleSpread,
    grid: TimeArray,
    median_k: FloatArray,
    levels: FloatArray,
    observed_share: FloatArray | None = None,
) -> FloatArray:
    """Quantile function of the clear-sky index at a ladder of probability levels."""
    ratios = _interpolate(spread.times, spread.ratios, grid)
    offsets = _interpolate(spread.times, spread.offsets, grid)

    # Interpolate across the quantile axis to reach levels the spread does not
    # carry directly.
    known = np.array(spread.quantiles, dtype=np.float64)
    dense_ratios = np.vstack([np.interp(levels, known, ratios[:, i]) for i in range(grid.size)]).T
    dense_offsets = np.vstack([np.interp(levels, known, offsets[:, i]) for i in range(grid.size)]).T

    if observed_share is not None:
        keep = np.clip(1.0 - np.asarray(observed_share, dtype=np.float64), 0.0, 1.0)
        dense_ratios = 1.0 + (dense_ratios - 1.0) * keep
        dense_offsets = dense_offsets * keep

    return np.asarray(np.clip(dense_ratios * median_k[None, :] + dense_offsets, K_MIN, K_MAX), dtype=np.float64)


def energy_quantiles(
    times: TimeArray,
    power_trajectories: FloatArray,
    start: np.datetime64,
    end: np.datetime64,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> dict[float, float]:
    """Integrate each trajectory over a window, then take quantiles of the totals.

    This ordering is the entire point. Integrating first and taking quantiles
    second gives the distribution of a day's energy; taking quantiles first and
    integrating second gives a number with no probabilistic meaning at all.

    Integration goes through the same routine the median energy sensor uses, so
    the band is measured against its own centre. A separate integrator here --
    even a correct one -- would let the two disagree by the difference between
    two rounding conventions on a mixed-resolution grid, which is exactly the
    kind of discrepancy that reads as a real spread.
    """
    energies = np.array([energy_between(times, row, start, end) for row in power_trajectories])
    return {level: float(np.quantile(energies, level)) for level in quantiles}
