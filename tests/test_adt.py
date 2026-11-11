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


def test_reference_architecture_and_legacy_shapes():
    import torch
    from torch import nn
    from adaptive_defense.adt_architecture import ReferenceAutoencoder, PAPER_SETTINGS
    ae = ReferenceAutoencoder(612,100)
    assert [(m.in_features,m.out_features) for m in ae.modules() if isinstance(m,nn.Linear)] == [(612,306),(306,153),(153,100),(100,153),(153,306),(306,612)]
    output = ae(torch.zeros(2,612))
    assert output.shape == (2,612) and ((output>=0)&(output<=1)).all()
    assert [(m.in_features,m.out_features) for m in ADT().online if isinstance(m,nn.Linear)] == [(6,32),(32,32),(32,16),(16,2)]
    assert ADT(hidden=(64,64)).online[-1].in_features == 64
    assert PAPER_SETTINGS['SWaT'] == dict(episodes=20000,tau=12,k=2,l=10,alpha=.9,beta=.1)


def test_paper_metric_edge_cases_and_robustness():
    from adaptive_defense.adt_evaluation import paper_metrics, robustness
    assert paper_metrics([0,0],[0,0]) == dict(precision=1.,recall=1.,f1=1.)
    assert paper_metrics([0,1],[1,0]) == dict(precision=0.,recall=0.,f1=0.)
    blocks, comparison = robustness(np.zeros(20),{'static':np.zeros(20),'adt':np.zeros(20)},'static')
    assert len(blocks) == 20
    assert (comparison.wilcoxon_two_sided_p == 1).all()


def test_terminal_bellman_target_does_not_bootstrap():
    import torch
    agent = ADT(7,learning_rate=0)
    for parameter in agent.online.parameters():
        parameter.data.zero_()
    for parameter in agent.target.parameters():
        parameter.data.fill_(10)
    state = np.zeros(6,dtype=np.float32)
    agent.memory.append((state,0,2.,state,True))
    assert agent.update(1) == pytest.approx(4.)
