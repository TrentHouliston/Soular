"""The online correction, through the integration.

The safety properties matter more than the accuracy ones here. A calibration that
learns from curtailed samples, or that survives a change of feature basis, or
that moves before it has evidence, does damage that is hard to notice and harder
to attribute.
"""

import math
from typing import Any

import numpy as np
import pytest

from custom_components.soular.core.correction import FEATURE_BASIS_ID, FEATURE_COUNT, CorrectionState, build_features
from custom_components.soular.core.mask import CURTAILMENT_SOC_PCT, MaskInputs, usable
from custom_components.soular.learning.store import decode, encode


def trained_state(ratio: float = 0.85, count: int = 4000) -> CorrectionState:
    """Build an estimator that has seen a steady efficiency error."""
    rng = np.random.default_rng(7)
    state = CorrectionState()
    features = build_features(
        elevation_deg=rng.uniform(15.0, 70.0, count),
        clearness=rng.uniform(0.3, 1.1, count),
        day_of_year=rng.uniform(0.0, 365.0, count),
        transmittance=np.ones(count),
        temperature_c=rng.uniform(8.0, 38.0, count),
    )
    for row in features:
        state.update(row, math.log(ratio), weight=1.0)
    return state


class TestPersistence:
    """Round-tripping the estimator through storage."""

    def test_state_survives_a_round_trip(self) -> None:
        """What comes back predicts what went in."""
        original = trained_state()
        restored = decode(encode({"east": original}))["east"]

        np.testing.assert_allclose(restored.theta, original.theta, rtol=1e-9)
        assert restored.samples == original.samples
        assert restored.efficiency() == pytest.approx(original.efficiency(), rel=1e-9)

    def test_the_covariance_comes_back_positive_definite(self) -> None:
        """Storing the Cholesky factor is what guarantees this.

        A covariance reassembled element-wise from JSON can come back very
        slightly asymmetric, and once it stops being positive definite the
        recursive gain changes sign and the estimate diverges quietly.
        """
        restored = decode(encode({"east": trained_state()}))["east"]
        eigenvalues = np.linalg.eigvalsh(restored.covariance)

        assert np.all(eigenvalues > 0.0)
        np.testing.assert_allclose(restored.covariance, restored.covariance.T, atol=1e-12)

    def test_a_different_basis_is_discarded(self) -> None:
        """Coefficients are meaningless against a basis they were not fitted to.

        Reinterpreting them would apply a confident-looking correction derived
        from features that no longer exist.
        """
        payload = encode({"east": trained_state()})
        payload["feature_basis_id"] = "something-else"
        assert decode(payload) == {}

    def test_corrupt_state_is_skipped_not_fatal(self) -> None:
        """A truncated file costs the learning, not the integration."""
        payload: dict[str, Any] = {
            "feature_basis_id": FEATURE_BASIS_ID,
            "arrays": {
                "east": {"theta": [0.1, 0.2], "cholesky": [], "samples": 5},
                "west": encode({"west": trained_state()})["arrays"]["west"],
            },
        }
        restored = decode(payload)
        assert "east" not in restored
        assert "west" in restored

    def test_nothing_stored_yields_nothing(self) -> None:
        """A first run starts with no state and no complaint."""
        assert decode(None) == {}
        assert decode({}) == {}


class TestCurtailmentMasking:
    """Which measured samples are allowed to teach."""

    def base(self, **overrides: Any) -> MaskInputs:
        """Build a single healthy mid-day sample."""
        defaults: dict[str, Any] = {
            "predicted_w": np.array([5000.0]),
            "actual_w": np.array([4300.0]),
            "elevation_deg": np.array([45.0]),
            "capacity_w": 7920.0,
        }
        defaults.update(overrides)
        return MaskInputs(**defaults)

    def test_a_healthy_sample_is_kept(self) -> None:
        """The ordinary case survives every rule."""
        assert bool(usable(self.base())[0])

    def test_a_full_battery_is_excluded(self) -> None:
        """Curtailment is lost production, not lost efficiency.

        This is the rule that matters most on a battery site: the array is
        throttled at the sunniest part of the day, which is exactly when samples
        carry the most weight.
        """
        assert not bool(usable(self.base(soc_pct=np.array([CURTAILMENT_SOC_PCT + 1])))[0])
        assert bool(usable(self.base(soc_pct=np.array([50.0])))[0])

    def test_the_inverter_plateau_is_excluded(self) -> None:
        """Output held at the limit is not output the array chose to make."""
        assert not bool(usable(self.base(actual_w=np.array([9900.0]), ac_limit_w=10000.0))[0])

    def test_a_dead_string_is_excluded(self) -> None:
        """Near-zero output under a bright sky is a fault, not a lesson."""
        assert not bool(usable(self.base(actual_w=np.array([10.0])))[0])

    def test_low_sun_is_excluded(self) -> None:
        """Near sunrise the ratio measures everything except efficiency."""
        assert not bool(usable(self.base(elevation_deg=np.array([3.0])))[0])

    def test_a_flat_top_is_excluded(self) -> None:
        """Pinned output while the model says it should be climbing is curtailment.

        This is what it looks like with no battery sensor to confirm it, so it
        has to be detectable from the two series alone.
        """
        flat = usable(
            MaskInputs(
                predicted_w=np.array([5000.0, 5600.0, 6300.0]),
                actual_w=np.array([4000.0, 4001.0, 4000.0]),
                elevation_deg=np.full(3, 45.0),
                capacity_w=7920.0,
            )
        )
        assert not flat.any()

    def test_a_genuinely_steady_hour_is_kept(self) -> None:
        """Steady output under a steady sun is real production, not a plateau."""
        steady = usable(
            MaskInputs(
                predicted_w=np.array([5000.0, 5010.0, 5005.0]),
                actual_w=np.array([4300.0, 4301.0, 4300.0]),
                elevation_deg=np.full(3, 45.0),
                capacity_w=7920.0,
            )
        )
        assert steady.all()

    def test_a_washed_out_day_is_dropped_whole(self) -> None:
        """A day far below expectation was an outage, not a lesson in efficiency."""
        result = usable(
            MaskInputs(
                predicted_w=np.full(6, 5000.0),
                actual_w=np.full(6, 500.0),
                elevation_deg=np.full(6, 45.0),
                capacity_w=7920.0,
                day_index=np.zeros(6),
            )
        )
        assert not result.any()


async def test_a_fresh_install_applies_no_correction(hass: Any, configured: Any) -> None:
    """With no measured-power sensor configured, nothing is learned or applied.

    The reference site's arrays have no power sensor in the fixture, so this also
    checks that the absence is handled as a missing feature rather than an error.
    """
    coordinator = configured.runtime_data.coordinator
    assert coordinator.learner.states == {}
    assert coordinator.data.ac_power_w.max() > 0.0


def test_the_feature_width_matches_the_declared_basis() -> None:
    """The basis identifier only means something if the width matches it."""
    features = build_features(
        elevation_deg=np.array([45.0]),
        clearness=np.array([0.9]),
        day_of_year=np.array([15.0]),
        transmittance=np.array([1.0]),
        temperature_c=np.array([25.0]),
    )
    assert features.shape == (1, FEATURE_COUNT)
