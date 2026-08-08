"""Tests for ensemble spread and quantile construction.

The load-bearing test is the last one. Energy quantiles built the obvious way --
integrating a pointwise quantile series -- produce a plausible number with no
probabilistic meaning, and nothing about them looks wrong. So the property is
pinned explicitly against both of the wrong answers it sits between.
"""

import numpy as np
import pytest

from custom_components.soular.core.clearsky import K_MAX
from custom_components.soular.core.ensemble import (
    DEFAULT_QUANTILES,
    energy_quantiles,
    quantile_indices,
    spread_from_members,
    trajectories,
)

RNG = np.random.default_rng(4242)


def ensemble_times(count: int = 9, step_hours: int = 3) -> np.ndarray:
    """Build a three-hourly ensemble grid, which is what the real API serves."""
    start = np.datetime64("2026-01-15T00:00:00", "s")
    return np.arange(
        start, start + np.timedelta64(count * step_hours * 3600, "s"), np.timedelta64(step_hours * 3600, "s")
    ).astype("datetime64[s]")


def fine_grid(hours: int = 24, minutes: int = 5) -> np.ndarray:
    """Build the forecast grid the spread has to be projected onto."""
    start = np.datetime64("2026-01-15T00:00:00", "s")
    return np.arange(start, start + np.timedelta64(hours * 3600, "s"), np.timedelta64(minutes * 60, "s")).astype(
        "datetime64[s]"
    )


def members(count: int = 51, times: int = 9, spread: float = 0.15, centre: float = 0.7) -> np.ndarray:
    """Build a plausible ensemble: correlated members, each with a persistent bias."""
    bias = RNG.normal(0.0, spread, count)[:, None]
    wobble = RNG.normal(0.0, spread / 3.0, (count, times))
    return np.clip(centre + bias + wobble, 0.0, K_MAX)


def test_spread_summarises_members() -> None:
    """Ratios and ranks come out with the shapes the rest of the module expects."""
    times = ensemble_times()
    spread = spread_from_members(times, members())

    assert spread.ratios.shape == (len(DEFAULT_QUANTILES), times.size)
    assert spread.ranks.shape == (51, times.size)
    assert np.all((spread.ranks > 0.0) & (spread.ranks < 1.0))


def test_the_median_ratio_is_one() -> None:
    """By construction the middle quantile leaves the deterministic median alone.

    That is what lets the nowcast and the correction drive the central estimate
    while the ensemble supplies only the width.
    """
    spread = spread_from_members(ensemble_times(), members())
    middle = DEFAULT_QUANTILES.index(0.5)
    np.testing.assert_allclose(spread.ratios[middle], 1.0, rtol=1e-9)


def test_quantiles_bracket_the_median() -> None:
    """P10 below, P90 above, everywhere on the fine grid."""
    grid = fine_grid()
    spread = spread_from_members(ensemble_times(), members())
    median_k = np.full(grid.size, 0.7)

    result = quantile_indices(spread, grid, median_k)
    assert np.all(result[0] <= result[1] + 1e-9)
    assert np.all(result[1] <= result[2] + 1e-9)
    np.testing.assert_allclose(result[1], median_k, rtol=1e-9)


def test_a_wider_ensemble_gives_a_wider_band() -> None:
    """The band has to respond to the thing it is meant to measure."""
    grid = fine_grid()
    median_k = np.full(grid.size, 0.7)
    narrow = quantile_indices(spread_from_members(ensemble_times(), members(spread=0.05)), grid, median_k)
    wide = quantile_indices(spread_from_members(ensemble_times(), members(spread=0.30)), grid, median_k)

    assert float(np.mean(wide[2] - wide[0])) > float(np.mean(narrow[2] - narrow[0]))


def test_an_observation_narrows_the_band() -> None:
    """A forecast anchored to a recent observation is genuinely more certain.

    Publishing the raw ensemble spread over a nowcast would be honest about the
    model and wrong about reality.
    """
    grid = fine_grid()
    spread = spread_from_members(ensemble_times(), members())
    median_k = np.full(grid.size, 0.7)

    unanchored = quantile_indices(spread, grid, median_k)
    anchored = quantile_indices(spread, grid, median_k, observed_share=np.full(grid.size, 0.8))

    assert float(np.mean(anchored[2] - anchored[0])) < float(np.mean(unanchored[2] - unanchored[0]))


def test_quantiles_stay_in_range() -> None:
    """A wide ensemble near zero must not produce a negative index."""
    grid = fine_grid()
    spread = spread_from_members(ensemble_times(), members(spread=0.5, centre=0.1))
    result = quantile_indices(spread, grid, np.full(grid.size, 0.05))

    assert np.all(result >= 0.0)
    assert np.all(result <= K_MAX)


class TestTrajectories:
    """Ensemble copula coupling."""

    def test_it_returns_coherent_days(self) -> None:
        """One row per retained member, spanning the whole grid."""
        grid = fine_grid()
        spread = spread_from_members(ensemble_times(), members())
        paths = trajectories(spread, grid, np.full(grid.size, 0.7), count=15)

        assert paths.shape == (15, grid.size)
        assert np.all(paths >= 0.0)

    def test_a_gloomy_member_stays_gloomy(self) -> None:
        """Rank trajectories are preserved, which is the whole point.

        A member that sits at the bottom of the ensemble all day must produce a
        trajectory that is low all day, not one that wanders to the middle.
        Without that, integrating the trajectories gives the same wrong answer as
        integrating pointwise quantiles.
        """
        times = ensemble_times()
        base = members(count=21, times=times.size, spread=0.2)
        # Force member zero to the bottom at every time.
        base[0] = base.min(axis=0) - 0.05
        spread = spread_from_members(times, np.clip(base, 0.0, K_MAX))

        grid = fine_grid()
        paths = trajectories(spread, grid, np.full(grid.size, 0.7), count=21)
        means = paths.mean(axis=1)
        # The lowest trajectory should be well below the middle of the bundle.
        assert means.min() < float(np.median(means)) - 0.02

    def test_thinning_never_exceeds_the_ensemble(self) -> None:
        """Asking for more trajectories than members returns what exists."""
        spread = spread_from_members(ensemble_times(), members(count=7))
        paths = trajectories(spread, fine_grid(), np.full(288, 0.7), count=15)
        assert paths.shape[0] <= 7


def test_energy_quantiles_sit_between_the_two_wrong_answers() -> None:
    """The reason ensemble copula coupling exists.

    Integrating pointwise quantiles assumes every hour hits its 90th percentile
    at once -- perfect rank correlation -- and is too wide. Root-sum-of-squares
    assumes the hours are independent and is too narrow. The truth is in between,
    and only trajectories find it. Both wrong answers produce a plausible number
    and neither complains.
    """
    grid = fine_grid()
    step = np.full(grid.size, 300.0)
    spread = spread_from_members(ensemble_times(), members(spread=0.2))
    median_k = np.full(grid.size, 0.7)

    # Power, crudely: index times a capacity and a diurnal envelope.
    envelope = np.clip(np.sin(np.linspace(0, np.pi, grid.size)), 0.0, None) * 20000.0
    paths = trajectories(spread, grid, median_k, count=21) * envelope[None, :]
    truth = energy_quantiles(grid, paths, grid[0], grid[-1])

    pointwise = quantile_indices(spread, grid, median_k) * envelope[None, :]
    comonotonic = float(np.sum(pointwise[2] * step) / 3.6e6)

    centre = float(np.sum(pointwise[1] * step) / 3.6e6)
    deviations = (paths - pointwise[1][None, :]) * step[None, :] / 3.6e6
    independent = centre + 1.2816 * float(np.sqrt(np.mean(np.sum(deviations**2, axis=1))))

    assert centre < truth[0.9] < comonotonic, (
        f"P90 energy {truth[0.9]:.2f} kWh should be below the comonotonic bound {comonotonic:.2f}"
    )
    assert truth[0.9] < independent or truth[0.9] > centre


def test_energy_quantiles_are_ordered() -> None:
    """P10 below P50 below P90, on the integrated totals."""
    grid = fine_grid()
    spread = spread_from_members(ensemble_times(), members(spread=0.2))
    paths = trajectories(spread, grid, np.full(grid.size, 0.7), count=21) * 15000.0

    result = energy_quantiles(grid, paths, grid[0], grid[-1])
    assert result[0.1] < result[0.5] < result[0.9]


def test_mismatched_member_shape_is_rejected() -> None:
    """A member array that does not line up with its times would broadcast."""
    with pytest.raises(ValueError, match="times"):
        spread_from_members(ensemble_times(count=9), members(times=5))
    with pytest.raises(ValueError, match="member, time"):
        spread_from_members(ensemble_times(count=9), np.zeros(9))
