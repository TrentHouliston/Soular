"""The weather correction: what it learns from, and how much it takes to move it."""

import numpy as np
import pytest

from custom_components.soular.core.weather_bias import (
    BINS,
    MAX_SCALE,
    MAX_TICK_SECONDS,
    MIN_SCALE,
    PRIOR_WH,
    WINDOW_DAYS,
    WeatherBias,
    bin_of,
    decode,
    encode,
    ratios,
)

DAY = 740000  # an arbitrary date ordinal
CLOUDY = 0.5
CLEAR = 1.0


def test_nothing_learned_changes_nothing() -> None:
    """A fresh install forecasts exactly what the physics says."""
    np.testing.assert_allclose(WeatherBias().scale(np.array([0.1, 0.5, 1.0]), DAY), 1.0)


def test_the_prior_holds_a_short_record_back() -> None:
    """One day of a cloudy forecast overshooting by half moves its bin only part of the way."""
    bias = WeatherBias()
    bias.add(DAY, np.array([CLOUDY]), np.array([20_000.0]), np.array([10_000.0]), 1.0)
    (scale,) = bias.scale(np.array([CLOUDY]), DAY + 1)
    assert scale == pytest.approx((10_000.0 + PRIOR_WH) / (20_000.0 + PRIOR_WH))
    assert 0.5 < scale < 1.0


def test_a_long_record_reaches_the_truth() -> None:
    """Many times the prior's worth of evidence lands close to the real ratio."""
    bias = WeatherBias()
    for day in range(WINDOW_DAYS - 1):
        bias.add(DAY + day, np.array([CLOUDY]), np.array([100_000.0]), np.array([70_000.0]), 1.0)
    (scale,) = bias.scale(np.array([CLOUDY]), DAY + WINDOW_DAYS - 1)
    assert scale == pytest.approx(0.7, abs=0.02)


def test_each_bin_learns_only_its_own_sky() -> None:
    """Cloudy forecasts overshooting says nothing about clear ones."""
    bias = WeatherBias()
    for day in range(30):
        bias.add(DAY + day, np.array([CLOUDY]), np.array([100_000.0]), np.array([60_000.0]), 1.0)
    cloudy, clear = bias.scale(np.array([CLOUDY, CLEAR]), DAY + 30)
    assert cloudy < 0.8
    assert clear == pytest.approx(1.0)


def test_only_the_window_before_today_counts() -> None:
    """Today's outcome and anything older than the window are left out."""
    bias = WeatherBias()
    bias.add(DAY, np.array([CLOUDY]), np.array([1e7]), np.array([0.0]), 1.0)
    bias.add(DAY + WINDOW_DAYS + 5, np.array([CLOUDY]), np.array([1e7]), np.array([0.0]), 1.0)
    (scale,) = bias.scale(np.array([CLOUDY]), DAY + WINDOW_DAYS + 5)
    assert scale == pytest.approx(1.0)


def test_the_correction_is_clamped() -> None:
    """A broken sensor reading zero for weeks cannot zero the forecast."""
    assert ratios(np.full(BINS, 1e9), np.zeros(BINS)).min() == MIN_SCALE
    assert ratios(np.full(BINS, 1.0), np.full(BINS, 1e9)).max() == MAX_SCALE


def test_ticks_accumulate_and_gaps_add_nothing() -> None:
    """Live readings count for the time since the last one, unless that was an outage."""
    bias = WeatherBias()
    bias.tick(DAY, 0, CLEAR, 1000.0, 900.0)
    bias.tick(DAY, 300, CLEAR, 1000.0, 900.0)
    bias.tick(DAY, 300 + MAX_TICK_SECONDS + 1, CLEAR, 1000.0, 900.0)
    predicted, actual = bias.days[DAY]
    assert sum(predicted) == pytest.approx(1000.0 * 300 / 3600)
    assert sum(actual) == pytest.approx(900.0 * 300 / 3600)


def test_an_unusable_reading_counts_on_neither_side() -> None:
    """A curtailed instant is not the forecast overshooting."""
    bias = WeatherBias()
    bias.tick(DAY, 0, CLEAR, 1000.0, None)
    bias.tick(DAY, 300, CLEAR, 1000.0, None)
    assert bias.days == {}


def test_old_days_are_pruned() -> None:
    """Storage stays bounded to the window."""
    bias = WeatherBias()
    bias.add(DAY, np.array([CLEAR]), np.array([1.0]), np.array([1.0]), 1.0)
    bias.tick(DAY + WINDOW_DAYS + 1, 0, CLEAR, 1.0, 1.0)
    bias.tick(DAY + WINDOW_DAYS + 1, 300, CLEAR, 1.0, 1.0)
    assert DAY not in bias.days


def test_clearness_bins_are_ordered() -> None:
    """Cloudier forecasts fall in lower bins, and a missing value reads as clear."""
    bins = bin_of(np.array([0.0, 0.3, 0.5, 0.7, 0.8, 1.0, np.nan]))
    assert list(bins) == sorted(bins)
    assert bins[-1] == BINS - 1


def test_round_trip_and_garbage() -> None:
    """Encoding round-trips, and an unreadable payload starts afresh."""
    bias = WeatherBias(last_tick=123)
    bias.add(DAY, np.array([CLOUDY, CLEAR]), np.array([5000.0, 8000.0]), np.array([4000.0, 7900.0]), 1.0)
    restored = decode(encode(bias))
    np.testing.assert_allclose(restored.bin_ratios(DAY + 1), bias.bin_ratios(DAY + 1))
    assert restored.last_tick == 123
    assert decode({"days": [[DAY, [1.0], [1.0]]]}).day_count == 0
    assert decode("nonsense").day_count == 0


def test_a_prior_is_where_an_empty_record_starts() -> None:
    """With no live record, the correction is the year's fit."""
    prior = (1.1, 1.0, 0.9, 0.88, 0.87, 0.95)
    bias = WeatherBias(prior=prior)
    np.testing.assert_allclose(bias.bin_ratios(DAY), prior)


def test_the_live_record_moves_away_from_the_prior() -> None:
    """A season that differs from the year pulls the correction its way, held back the same as from 1."""
    bias = WeatherBias(prior=(1.0,) * BINS)
    for day in range(WINDOW_DAYS - 1):
        bias.add(DAY + day, np.array([CLEAR]), np.array([100_000.0]), np.array([80_000.0]), 1.0)
    (scale,) = bias.scale(np.array([CLEAR]), DAY + WINDOW_DAYS - 1)
    assert scale == pytest.approx(0.8, abs=0.02)


def test_the_prior_is_not_stored() -> None:
    """It is configuration from the map, so a changed map is never overridden by an old copy."""
    assert "prior" not in encode(WeatherBias(prior=(0.9,) * BINS))
