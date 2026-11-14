import numpy as np
import pytest
from adaptive_defense.research_diagnostics import score_diagnostics, replay_order, require_equal


def test_separation_and_inversion():
    perfect=score_diagnostics([0,0,1,1],[0,.1,.8,1])
    inverted=score_diagnostics([0,0,1,1],[1,.8,.1,0])
    assert perfect['roc_auc'] == perfect['average_precision'] == 1
    assert inverted['roc_auc'] == 0
    assert perfect['attack_prevalence'] == .5
    with pytest.raises(ValueError):
        score_diagnostics([0,0],[1,2])


def test_order_preserves_rows_and_is_reproducible():
    a=replay_order(100,42)
    assert np.array_equal(np.sort(a),np.arange(100))
    assert np.array_equal(a,replay_order(100,42))
    assert not np.array_equal(a,replay_order(100,43))


def test_integrity_check_raises():
    require_equal(['a'],['a'],'mismatch')
    with pytest.raises(ValueError,match='mismatch'):
        require_equal(['a'],['b'],'mismatch')
