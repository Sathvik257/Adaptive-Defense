"""Label-free score-distribution monitoring; warnings are not attack labels."""
import numpy as np


def score_shift(reference, current):
    """Two-sample empirical CDF distance (KS statistic, without a p-value)."""
    left, right = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    if any(x.ndim != 1 or not len(x) or not np.isfinite(x).all() for x in (left, right)):
        raise ValueError('Need nonempty finite score vectors.')
    points = np.unique(np.r_[left, right])
    return float(np.max(np.abs(np.searchsorted(np.sort(left), points, side='right')/len(left)
                              - np.searchsorted(np.sort(right), points, side='right')/len(right))))


def monitor_scores(reference, stream, window=512, warning_distance=.2):
    """Non-overlapping completed windows; final partial window stays unreported.

    The warning cutoff is an operational heuristic, not a calibrated hypothesis
    test. Warnings can reflect attacks, benign drift, or changed class mixture.
    No model updates or automatic recalibration are performed.
    """
    values = np.asarray(stream, dtype=float)
    score_shift(reference, values)
    if not isinstance(window, int) or window < 2 or not 0 < warning_distance <= 1:
        raise ValueError('Invalid monitoring window or warning distance.')
    rows = []
    for end in range(window, len(values)+1, window):
        distance = score_shift(reference, values[end-window:end])
        rows.append(dict(start=end-window, end_exclusive=end, distance=distance,
                         warning=distance > warning_distance))
    return rows
