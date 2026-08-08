"""Learn what the physics keeps getting wrong, from the array's own output.

Every fitted efficiency the backtest produced sits between 0.84 and 0.93: the
forward model over-predicts by 7-16% before anything is calibrated, consistently,
across shading revisions. That is a large, stable, entirely learnable error, and
nothing in the physics is going to find it -- soiling, module tolerance, wiring
and connection losses are not derivable from a datasheet.

The estimator is a recursive weighted ridge regression with exponential
forgetting: equivalent to refitting an exponentially-weighted ridge at every
sample, but O(d^2) per sample with no history to replay and a state small enough
to persist as JSON.

Four choices that are load-bearing.

**The target is a log ratio.** ``y = log(actual / predicted)`` makes the
correction multiplicative and symmetric, and turns the safety clamp into a box
constraint on a linear quantity. The obvious alternative, a plain ratio, is
heteroscedastic and unbounded: at eight watts predicted it is pure noise, and it
would enter a least-squares fit with the same weight as a seven-kilowatt sample.

**Samples are weighted by predicted output.** Approximately inverse-variance
under multiplicative noise, and it makes the fit care about the samples that
carry the energy rather than the ones that merely occur.

**The basis is deliberately tiny.** Eight terms cannot memorise the sky, which
matters because a flexible corrector fitted against a shaded site will happily
learn the shading map's inverse and undo it. The ``transmittance`` term gives the
shading map's own error an explicit, inspectable home: if it grows large the map
needs re-deriving, and that is worth surfacing rather than absorbing silently.

Note that the intercept is *not* the per-array efficiency on its own. Elevation,
clearness and the annual harmonics are correlated over any window shorter than a
year, so the estimator spreads one effect across several coefficients. What the
efficiency means here is the correction evaluated at stated reference
conditions -- see :meth:`CorrectionState.efficiency`.

**It starts at exactly 1.0 and earns its way out.** ``theta`` initialised to zero
with a finite prior covariance *is* the ridge prior, so a fresh install applies
no correction at all and moves only as the data insists.
"""

from dataclasses import dataclass, field
import math
from typing import Final

import numpy as np
from numpy.typing import NDArray

from custom_components.soular.core.types import FloatArray

# Bumped whenever the basis changes. Stored state under a different identifier is
# discarded rather than reinterpreted: coefficients are meaningless against a
# basis they were not fitted to.
FEATURE_BASIS_ID: Final = "v1"

FEATURE_NAMES: Final = (
    "efficiency",
    "elevation",
    "clearness",
    "clearness_squared",
    "annual_cos",
    "annual_sin",
    "transmittance",
    "ambient_temperature",
)
FEATURE_COUNT: Final = len(FEATURE_NAMES)

# Ridge prior. Small enough that the correction starts at 1.0 and stays there
# until the evidence is real, loose enough to move within a few days of sun.
PRIOR_VARIANCE: Final = 0.01

# Effective window of roughly thirty days of five-minute daylight samples.
FORGETTING: Final = 1.0 - 1.0 / 4300.0

# Below this many effective samples the correction is ramped in, so a fresh
# install is pure physics and slides into the learned term over its first week.
MIN_EFFECTIVE_SAMPLES: Final = 500.0

# The correction is a refinement, not a licence to rewrite the forecast. Anything
# outside this band is a modelling failure that a multiplier should not paper over.
MIN_CORRECTION: Final = 0.75
MAX_CORRECTION: Final = 1.30

# Centring constants, chosen so the features are near zero in typical daylight and
# the prior therefore means what it says.
ELEVATION_CENTRE: Final = 0.5
CLEARNESS_CENTRE: Final = 0.7
TEMPERATURE_CENTRE: Final = 25.0
TEMPERATURE_SCALE: Final = 25.0
DAYS_PER_YEAR: Final = 365.25


def build_features(
    elevation_deg: FloatArray,
    clearness: FloatArray,
    day_of_year: FloatArray,
    transmittance: FloatArray,
    temperature_c: FloatArray,
) -> FloatArray:
    """Assemble the design matrix, one row per sample.

    ``clearness`` is the clear-sky index and ``transmittance`` the shading map's
    value at that sun position, so the two terms that most often go wrong each
    get somewhere explicit to land.
    """
    elevation = np.clip(np.sin(np.radians(elevation_deg)), 0.0, 1.0) - ELEVATION_CENTRE
    clear = np.asarray(clearness, dtype=np.float64) - CLEARNESS_CENTRE
    angle = 2.0 * np.pi * np.asarray(day_of_year, dtype=np.float64) / DAYS_PER_YEAR
    return np.column_stack(
        [
            np.ones_like(elevation),
            elevation,
            clear,
            clear**2,
            np.cos(angle),
            np.sin(angle),
            np.asarray(transmittance, dtype=np.float64) - 1.0,
            (np.asarray(temperature_c, dtype=np.float64) - TEMPERATURE_CENTRE) / TEMPERATURE_SCALE,
        ]
    )


# Clear, unshaded, mid-elevation, standard temperature, annual harmonics at their
# year-round mean. The conditions the reported efficiency is quoted at.
REFERENCE_FEATURES: Final = np.array(
    [
        [
            1.0,
            math.sin(math.radians(45.0)) - ELEVATION_CENTRE,
            1.0 - CLEARNESS_CENTRE,
            (1.0 - CLEARNESS_CENTRE) ** 2,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    ]
)


@dataclass(slots=True)
class CorrectionState:
    """Everything the estimator remembers, for one array."""

    theta: FloatArray = field(default_factory=lambda: np.zeros(FEATURE_COUNT))
    covariance: FloatArray = field(default_factory=lambda: np.eye(FEATURE_COUNT) * PRIOR_VARIANCE)
    samples: int = 0
    forgetting: float = FORGETTING

    @property
    def effective_samples(self) -> float:
        """Samples the forgetting factor still remembers.

        Saturates at ``1 / (1 - lambda)``, which is what makes the ramp-in
        terminate rather than creeping upward forever.
        """
        if self.samples == 0:
            return 0.0
        return float((1.0 - self.forgetting**self.samples) / (1.0 - self.forgetting))

    def update(self, features: FloatArray, target: float, weight: float) -> None:
        """Fold one observation into the estimate.

        Weighted recursive least squares. The weight enters as ``lambda / w`` in
        the gain denominator, which is the standard equivalence to scaling the
        sample's contribution to the loss.
        """
        if weight <= 0.0 or not math.isfinite(target):
            return

        x = np.asarray(features, dtype=np.float64).reshape(FEATURE_COUNT)
        covariance_x = self.covariance @ x
        denominator = self.forgetting / weight + float(x @ covariance_x)
        if denominator <= 0.0:  # pragma: no cover - defensive
            return

        gain = covariance_x / denominator
        self.theta = self.theta + gain * (target - float(x @ self.theta))
        self.covariance = (self.covariance - np.outer(gain, covariance_x)) / self.forgetting
        # Symmetrise. Floating-point drift over a hundred thousand rank-one
        # updates is real, and an asymmetric covariance eventually stops being
        # positive definite, at which point the gain changes sign.
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.samples += 1

    def uncertainty(self) -> FloatArray:
        """Posterior standard deviation of each coefficient."""
        return np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))

    def confident(self) -> NDArray[np.bool_]:
        """Which coefficients are larger than their own uncertainty.

        A diagnostic, not a gate. Zeroing individual coefficients would break the
        linearity of the fit: when features are correlated -- and elevation,
        clearness and the annual harmonics are all correlated over any window
        shorter than a year -- the estimator spreads one effect across several
        coefficients, and dropping some of them destroys the sum rather than
        making it safer. Restraint comes from the ridge prior, the ramp and the
        clamp, which act on the answer rather than on its parts.
        """
        return np.abs(self.theta) > self.uncertainty()

    def correction(self, features: FloatArray) -> FloatArray:
        """Multiplicative correction for each sample, clamped and ramped in."""
        design = np.atleast_2d(np.asarray(features, dtype=np.float64))
        ramp = float(np.clip(self.effective_samples / MIN_EFFECTIVE_SAMPLES, 0.0, 1.0))
        log_correction = np.clip(
            ramp * (design @ self.theta),
            math.log(MIN_CORRECTION),
            math.log(MAX_CORRECTION),
        )
        return np.asarray(np.exp(log_correction), dtype=np.float64)

    def efficiency(self) -> float:
        """Learned correction under reference conditions, as a multiplier.

        Reference means clear, unshaded, mid-elevation, at standard temperature,
        with the annual harmonics at their year-round mean of zero. Deliberately
        not ``theta[0]``: the intercept alone is only meaningful if every other
        feature is centred at zero for the sample in question, and the annual
        terms are near +/-1 for months at a time. Reading the intercept would
        report a number the estimator never applies.
        """
        return float(self.correction(REFERENCE_FEATURES)[0])


def observation_target(actual_w: float, predicted_w: float, *, floor_w: float) -> float | None:
    """Log ratio of measured to modelled output, or nothing if unusable.

    Both sides need to be meaningfully above zero. Near sunrise the ratio is
    dominated by everything except the thing being estimated.
    """
    if not (math.isfinite(actual_w) and math.isfinite(predicted_w)):
        return None
    if predicted_w < floor_w or actual_w <= 0.0:
        return None
    return math.log(actual_w / predicted_w)


def sample_weight(predicted_w: float, capacity_w: float) -> float:
    """Weight a sample by how much of the array's capacity it represents."""
    if capacity_w <= 0.0:
        return 0.0
    return float(np.clip(predicted_w / capacity_w, 0.0, 1.0))
