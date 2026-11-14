import numpy as np
import pytest
from adaptive_defense.calibrated_adt import calibrated_threshold, gated_replay

class Alert:
    def action(self, state):
        return 0


def test_rank_ties_and_small_sample():
    assert calibrated_threshold([1, 2, 3, 4], .4) == 3
    assert np.isinf(calibrated_threshold([1, 2], .01))
    p, _ = gated_replay(Alert(), [3, 4], 3, budget=0)
    assert p.tolist() == [0, 1]
    with pytest.raises(ValueError):
        calibrated_threshold([np.nan])


def test_budget_and_causal_delay():
    called = []
    states = []
    class Capture:
        def action(self, state):
            states.append(state.copy())
            return 0
    def feedback(i):
        called.append(i)
        return 0
    _, trace = gated_replay(Capture(), np.ones(10), .5, feedback=feedback, budget=.5, delay=2, k=5)
    assert called == [1, 3, 5, 7]
    assert all(sum(r['queried'] for r in trace[:n]) <= int(n*.5) for n in range(1, 11))
    assert states[2][4] == 0
    assert states[3][4] == pytest.approx(.2)


def test_gate_dominates_policy_and_no_feedback():
    class Quiet:
        def action(self, state):
            return 1
    scores = np.array([.1, .7, 1.2, 2.])
    for agent in (Alert(), Quiet()):
        p, _ = gated_replay(agent, scores, .8, budget=0)
        assert np.all(p <= (scores > .8))
    with pytest.raises(ValueError):
        gated_replay(Alert(), scores, .8, delay=0)

def test_reviews_survive_delay_longer_than_score_window():
    states = []
    class Capture:
        def action(self, state):
            states.append(state.copy())
            return 0
    gated_replay(Capture(), np.ones(8), .5, feedback=lambda i:0, budget=1, delay=4, k=2)
    assert states[3][4] == 0
    assert states[4][4] == pytest.approx(.5)
    assert states[5][4] == pytest.approx(1.)

def test_boundary_review_uses_only_past_scores_and_delays_from_request():
    calls = []
    _, trace = gated_replay(Alert(), [.49, .9, .8, .7, .6, .2], .5,
                           feedback=lambda i: calls.append(i) or 0,
                           budget=.5, delay=2, review_strategy='boundary')
    assert trace[1]['review_index'] == 0
    assert trace[3]['review_index'] == 3
    assert calls == [0, 3]
    assert trace[2]['delivered'] == 0
    assert trace[3]['delivered'] == 1


def test_monitor_shift_and_complete_windows():
    from adaptive_defense.score_monitor import score_shift, monitor_scores
    assert score_shift([0, 1], [0, 1]) == 0
    assert score_shift([0, 0], [1, 1]) == 1
    rows = monitor_scores([0, 0], [0, 0, 1, 1, 2], window=2)
    assert [r['warning'] for r in rows] == [False, True]
    assert rows[-1]['end_exclusive'] == 4
    with pytest.raises(ValueError):
        score_shift([], [1])
