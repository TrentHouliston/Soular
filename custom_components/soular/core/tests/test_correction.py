"""Tests for the online bias correction.

Two families. The first is that it learns the thing it exists to learn -- a
constant efficiency error, then a drifting one. The second is that it cannot do
harm: it starts at exactly 1.0, it never leaves its clamp, and a small basis
cannot memorise the sky and undo the shading map.
"""

import math

import numpy as np
import pytest

from custom_components.soular.core.correction import (
    FEATURE_COUNT,
    MAX_CORRECTION,
    MIN_CORRECTION,
    MIN_EFFECTIVE_SAMPLES,
    CorrectionState,
    build_features,
    observation_target,
    sample_weight,
)

RNG = np.random.default_rng(20260808)


def daylight_features(count: int, *, varied: bool = True) -> np.ndarray:
    """Build a run of plausible daylight samples.

    Varied by default, because identical rows are perfectly collinear and no
    estimator can decompose one number into eight coefficients. Real daylight
    sweeps elevation across the day, clearness across the weather and the annual
    harmonics across the year, which is what makes the intercept identifiable.
    """
    if not varied:
        return build_features(
            elevation_deg=np.full(count, 45.0),
            clearness=np.full(count, 0.9),
            day_of_year=np.full(count, 15.0),
            transmittance=np.ones(count),
            temperature_c=np.full(count, 25.0),
        )
    return build_features(
        elevation_deg=RNG.uniform(15.0, 70.0, count),
        clearness=RNG.uniform(0.3, 1.1, count),
        day_of_year=RNG.uniform(0.0, 365.0, count),
        transmittance=np.ones(count),
        temperature_c=RNG.uniform(8.0, 38.0, count),
    )


def train(state: CorrectionState, features: np.ndarray, ratio: float, *, noise: float = 0.0) -> None:
    """Feed samples whose actual output is ``ratio`` times the model's."""
    for row in features:
        target = math.log(ratio) + (float(RNG.normal(0.0, noise)) if noise else 0.0)
        state.update(row, target, weight=1.0)


def test_a_fresh_estimator_applies_no_correction() -> None:
    """Before any data the forecast is pure physics.

    Not approximately 1.0 -- exactly. A correction that starts anywhere else is
    asserting something it has not learned.
    """
    state = CorrectionState()
    result = state.correction(daylight_features(5))

    np.testing.assert_array_equal(result, np.ones(5))
    assert state.effective_samples == 0.0


def test_it_learns_a_constant_efficiency_error() -> None:
    """The thing it exists for: the model over-predicts by a steady fraction.

    Every fitted gain the backtest produced sat between 0.84 and 0.93, so this is
    the case that matters most.
    """
    state = CorrectionState()
    train(state, daylight_features(4000), ratio=0.87)

    assert state.efficiency() == pytest.approx(0.87, rel=0.03)
    np.testing.assert_allclose(state.correction(daylight_features(200)), 0.87, rtol=0.05)


def test_it_learns_through_noise() -> None:
    """A real signal survives sample-to-sample scatter."""
    state = CorrectionState()
    train(state, daylight_features(8000), ratio=0.88, noise=0.25)

    assert state.efficiency() == pytest.approx(0.88, rel=0.06)


def test_it_ramps_in_rather_than_jumping() -> None:
    """A handful of samples must not move the forecast far.

    Otherwise a freshly installed integration lurches on its first cloudy
    afternoon, which looks exactly like a bug to the person watching it.
    """
    state = CorrectionState()
    train(state, daylight_features(50), ratio=0.5)

    early = state.correction(daylight_features(1, varied=False))[0]
    assert early > 0.95, f"50 samples moved the correction to {early:.3f}"

    train(state, daylight_features(3000), ratio=0.5)
    assert state.effective_samples > MIN_EFFECTIVE_SAMPLES
    assert state.efficiency() < 0.85


def test_it_tracks_a_step_change() -> None:
    """Soiling, a prune, a cleaned panel: the estimate has to follow.

    Exponential forgetting is what makes this possible; a plain accumulator would
    still be averaging in last season's data a year later.
    """
    state = CorrectionState()
    train(state, daylight_features(8000), ratio=0.95)
    assert state.efficiency() == pytest.approx(0.95, rel=0.03)

    train(state, daylight_features(12000), ratio=0.80)
    assert state.efficiency() == pytest.approx(0.80, rel=0.06)


def test_the_correction_is_clamped() -> None:
    """A catastrophic mismatch is a fault, not a licence to rewrite the forecast."""
    for ratio in (0.05, 5.0):
        state = CorrectionState()
        train(state, daylight_features(8000), ratio=ratio)
        values = state.correction(daylight_features(500))
        assert np.all(values >= MIN_CORRECTION - 1e-9)
        assert np.all(values <= MAX_CORRECTION + 1e-9)


def test_pure_noise_teaches_it_nothing() -> None:
    """With no signal to find, the correction stays at one.

    The ridge prior is what does this: with nothing pulling the coefficients away
    from zero, they stay there, and the reported efficiency stays at unity.
    """
    state = CorrectionState()
    # Pure noise about zero: nothing to learn, so nothing should be confident.
    for row in daylight_features(500):
        state.update(row, float(RNG.normal(0.0, 0.5)), weight=1.0)

    assert not state.confident().all()
    assert state.efficiency() == pytest.approx(1.0, abs=0.08)


def test_the_covariance_stays_symmetric_and_positive() -> None:
    """Rank-one updates drift; an asymmetric covariance eventually flips the gain."""
    state = CorrectionState()
    train(state, daylight_features(20000), ratio=0.9, noise=0.3)

    np.testing.assert_allclose(state.covariance, state.covariance.T, rtol=0, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(state.covariance) > 0.0)


def test_a_small_basis_cannot_memorise_the_sky() -> None:
    """Eight terms cannot reproduce an arbitrary function of sun position.

    This is the guard against the corrector learning the shading map's inverse
    and undoing it. Fed a sharply direction-dependent error, it must fail to fit
    most of it.
    """
    state = CorrectionState()
    azimuths = np.linspace(0.0, 360.0, 4000)
    features = build_features(
        elevation_deg=np.full(azimuths.size, 40.0),
        clearness=np.full(azimuths.size, 0.9),
        day_of_year=np.full(azimuths.size, 15.0),
        transmittance=np.ones(azimuths.size),
        temperature_c=np.full(azimuths.size, 25.0),
    )
    # A shadow: a deep, narrow notch at one azimuth.
    truth = np.where(np.abs(azimuths - 90.0) < 15.0, math.log(0.3), 0.0)
    for row, target in zip(features, truth, strict=True):
        state.update(row, float(target), weight=1.0)

    fitted = state.correction(features).ravel()
    residual = np.abs(np.log(fitted) - truth)
    # It should not have captured the notch: the deep samples stay badly fitted.
    notch = np.abs(azimuths - 90.0) < 15.0
    assert float(np.mean(residual[notch])) > 0.5


def test_weighting_favours_high_output_samples() -> None:
    """A near-dark sample must not carry the same weight as a peak one."""
    assert sample_weight(7000.0, 7920.0) > sample_weight(80.0, 7920.0)
    assert sample_weight(80.0, 7920.0) == pytest.approx(80.0 / 7920.0)
    assert sample_weight(100.0, 0.0) == 0.0


def test_zero_weight_samples_are_ignored() -> None:
    """A zero-weight sample changes nothing at all."""
    state = CorrectionState()
    before = state.theta.copy()
    state.update(daylight_features(1)[0], math.log(0.5), weight=0.0)
    np.testing.assert_array_equal(state.theta, before)
    assert state.samples == 0


class TestObservationTarget:
    """Turning a pair of powers into a learning target."""

    def test_a_clean_pair_gives_a_log_ratio(self) -> None:
        """Half the expected output is a log ratio of log(0.5)."""
        assert observation_target(2500.0, 5000.0, floor_w=500.0) == pytest.approx(math.log(0.5))

    def test_low_predicted_output_is_refused(self) -> None:
        """Near sunrise the ratio measures everything except efficiency."""
        assert observation_target(50.0, 100.0, floor_w=500.0) is None

    def test_zero_output_is_refused(self) -> None:
        """log(0) is not a target; a dead string is a fault, not a lesson."""
        assert observation_target(0.0, 5000.0, floor_w=500.0) is None

    def test_non_finite_input_is_refused(self) -> None:
        """A gap in the record must not poison the estimate."""
        assert observation_target(float("nan"), 5000.0, floor_w=500.0) is None


def test_features_have_the_declared_width() -> None:
    """The basis identifier is only meaningful if the width matches it."""
    assert daylight_features(3).shape == (3, FEATURE_COUNT)
    assert daylight_features(3, varied=False).shape == (3, FEATURE_COUNT)


def test_features_are_centred_near_zero_in_typical_daylight() -> None:
    """The ridge prior means what it says only if the design is centred.

    An uncentred basis makes the prior asymmetric: it would pull some
    coefficients toward zero much harder than others for no stated reason.
    """
    features = build_features(
        elevation_deg=np.array([30.0, 45.0, 60.0]),
        clearness=np.array([0.6, 0.7, 0.9]),
        day_of_year=np.array([15.0, 180.0, 300.0]),
        transmittance=np.array([1.0, 0.95, 1.0]),
        temperature_c=np.array([20.0, 25.0, 32.0]),
    )
    # Column zero is the intercept and is meant to be one; the rest are centred.
    np.testing.assert_array_equal(features[:, 0], np.ones(3))
    assert np.all(np.abs(features[:, 1:]) <= 1.0)
