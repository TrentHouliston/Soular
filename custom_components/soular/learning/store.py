"""Persist what the correction has learned.

Stored as the Cholesky factor of the covariance rather than the covariance
itself. Two reasons: it halves the payload, and it guarantees the matrix comes
back positive definite. A covariance round-tripped through JSON and reassembled
element-wise can come back very slightly asymmetric, and once it stops being
positive definite the recursive update's gain changes sign and the estimate
diverges quietly.

``feature_basis_id`` is the real versioning lever. Coefficients are meaningless
against a basis they were not fitted to, so a mismatch discards the state and
starts again rather than reinterpreting numbers that no longer mean anything.
"""

from typing import Any, TypedDict

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
import numpy as np

from custom_components.soular.const import DOMAIN
from custom_components.soular.core.correction import FEATURE_BASIS_ID, FEATURE_COUNT, CorrectionState

STORAGE_VERSION = 1
STORAGE_MINOR_VERSION = 1

# Saving on every sample would write to disk every five minutes for no benefit;
# losing a few minutes of learning to a hard restart costs nothing measurable.
SAVE_DELAY_SECONDS = 300


class ArrayStateDict(TypedDict):
    """Serialised estimator state for one array."""

    theta: list[float]
    cholesky: list[float]
    samples: int
    forgetting: float


class LearningStateDict(TypedDict):
    """Everything persisted for one config entry."""

    feature_basis_id: str
    arrays: dict[str, ArrayStateDict]


def encode(states: dict[str, CorrectionState]) -> LearningStateDict:
    """Serialise estimator state."""
    arrays: dict[str, ArrayStateDict] = {}
    for name, state in states.items():
        # Nudge onto the positive-definite cone before factorising: a hundred
        # thousand rank-one updates leave the smallest eigenvalue near zero.
        symmetric = 0.5 * (state.covariance + state.covariance.T)
        symmetric = symmetric + np.eye(FEATURE_COUNT) * 1e-12
        try:
            factor = np.linalg.cholesky(symmetric)
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            continue
        arrays[name] = ArrayStateDict(
            theta=[float(v) for v in state.theta],
            cholesky=[float(v) for v in factor.reshape(-1)],
            samples=int(state.samples),
            forgetting=float(state.forgetting),
        )
    return LearningStateDict(feature_basis_id=FEATURE_BASIS_ID, arrays=arrays)


def decode(payload: Any) -> dict[str, CorrectionState]:
    """Restore estimator state, discarding anything fitted to another basis."""
    if not isinstance(payload, dict) or payload.get("feature_basis_id") != FEATURE_BASIS_ID:
        return {}

    states: dict[str, CorrectionState] = {}
    for name, entry in (payload.get("arrays") or {}).items():
        try:
            theta = np.array(entry["theta"], dtype=np.float64)
            factor = np.array(entry["cholesky"], dtype=np.float64).reshape(FEATURE_COUNT, FEATURE_COUNT)
        except (KeyError, TypeError, ValueError):
            continue
        if theta.shape != (FEATURE_COUNT,):
            continue
        states[name] = CorrectionState(
            theta=theta,
            covariance=factor @ factor.T,
            samples=int(entry.get("samples", 0)),
            forgetting=float(entry.get("forgetting", CorrectionState().forgetting)),
        )
    return states


def build_store(hass: HomeAssistant, entry_id: str) -> Store[LearningStateDict]:
    """Create the store for one config entry."""
    return Store[LearningStateDict](
        hass,
        STORAGE_VERSION,
        f"{DOMAIN}.learning.{entry_id}",
        minor_version=STORAGE_MINOR_VERSION,
    )
