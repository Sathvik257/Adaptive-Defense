import numpy as np
import pytest
from adaptive_defense.adt import History, ADT, replay, windows
from adaptive_defense.adt_experiment import static_threshold


def test_history_equations_and_strict_boundary():
    h = History(2,.9)
    np.testing.assert_array_equal(h.state(),np.zeros(6))
    assert h.observe(.2,0,1) == (1,.9)
    prediction,reward = h.observe(.8,0,0)
    assert prediction == 1
    assert reward == pytest.approx(0)
    np.testing.assert_allclose(h.state(),[.5,.18,.5,0,.5,0])
    assert h.observe(1,1,0)[0] == 0
    assert h.observe(0,0,1)[0] == 0


def test_current_label_cannot_change_current_decision():
    agent = ADT(4)
    _, a = replay(agent,[.1,.2,.3],[0,0,0])
    _, b = replay(agent,[.1,.2,.3],[0,1,1])
    assert a[:2] != b[:2]  # Reward changes after the second decision.
    for i in range(2):
        for key in ('threshold','mean','variance','tp_fraction','tn_fraction','fp_fraction','fn_fraction'):
            assert a[i][key] == b[i][key]


def test_no_feedback_and_episode_updates():
    agent = ADT(5)
    before = {k:v.clone() for k,v in agent.online.state_dict().items()}
    _, trace = replay(agent,[.2,.4,.6,.8],[0,1,0,1],training=True,action_interval=2)
    assert trace[0]['threshold'] == trace[1]['threshold']
    assert trace[2]['threshold'] == trace[3]['threshold']
    assert all((before[k]==v).all() for k,v in agent.online.state_dict().items())
    assert agent.memory[-1][-1] is True
    assert np.isfinite(agent.update(4))
    _, trace = replay(agent,[.2,.4,.6])
    assert all(row['reward'] is None and row['tp_fraction']==row['tn_fraction']==row['fp_fraction']==row['fn_fraction']==0 for row in trace)
    with pytest.raises(ValueError):
        replay(agent,[.2],action_interval=2)
    with pytest.raises(ValueError):
        replay(agent,[1.1])


def test_window_label_and_order():
    x,y = windows(np.arange(8).reshape(4,2),[0,1,0,0],2)
    np.testing.assert_array_equal(x,[[0,1,2,3],[2,3,4,5],[4,5,6,7]])
    np.testing.assert_array_equal(y,[1,1,0])


def test_validation_static_threshold_ties():
    scores = np.array([.1,.2,.2,.9])
    threshold = static_threshold(scores,np.array([0,1,1,1]))
    np.testing.assert_array_equal(scores>threshold,[False,True,True,True])
