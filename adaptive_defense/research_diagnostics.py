"""Diagnostic score metrics and reproducible replay-order stress tests."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def score_diagnostics(labels, scores):
    y, s = np.asarray(labels), np.asarray(scores, dtype=float)
    if y.ndim != 1 or y.shape != s.shape or not len(y) or not np.isin(y,[0,1]).all() or not np.isfinite(s).all():
        raise ValueError('Need aligned binary labels and finite scores.')
    if len(np.unique(y)) != 2:
        raise ValueError('Both classes are required for score diagnostics.')
    return {'roc_auc':float(roc_auc_score(y,s)),
            'average_precision':float(average_precision_score(y,s)),
            'attack_prevalence':float(y.mean()),
            'normal_score_median':float(np.median(s[y==0])),
            'attack_score_median':float(np.median(s[y==1]))}


def replay_order(size, seed):
    if not isinstance(size, int) or size < 1:
        raise ValueError('size must be a positive integer.')
    return np.random.default_rng(seed).permutation(size)


def require_equal(actual, expected, message):
    """Integrity checks stay active under python -O."""
    if actual != expected:
        raise ValueError(message)
