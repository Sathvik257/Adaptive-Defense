"""Descriptive score shift checks; never an attack verdict or retraining trigger."""
import numpy as np


def score_shift(scores, count, window=64):
    """Compare the first window with the latest disjoint observed window.

    KS distance is descriptive: no p-value, independence assumption or
    calibrated alert threshold is implied for correlated replay traffic.
    """
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= len(scores):
        raise ValueError("count must identify an observed prefix")
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("window must be at least two")
    observed = np.asarray(scores[:count], dtype=float)
    if observed.ndim != 1 or not np.isfinite(observed).all():
        raise ValueError("observed scores must be finite scalars")
    result = {"observed": count, "window": window,
              "required": 2 * window, "status": "warming-up",
              "ks_distance": None, "reference_median": None, "recent_median": None,
              "note": "Score distribution change is not proof of an attack or concept drift. No model update is performed."}
    if count < 2 * window:
        return result
    reference, recent = observed[:window], observed[-window:]
    support = np.sort(np.concatenate((reference, recent)))
    distance = np.max(np.abs(
        np.searchsorted(np.sort(reference), support, side="right") / window -
        np.searchsorted(np.sort(recent), support, side="right") / window))
    return dict(result, status="ready", ks_distance=float(distance),
                reference_median=float(np.median(reference)), recent_median=float(np.median(recent)))
