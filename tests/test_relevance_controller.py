import numpy as np
import pytest
from adaptive_defense.relevance_controller import controller_replay,relevance_weights

class Query:
    def action(self,state,epsilon=0):return 1


def test_delay_budget_and_no_future_feedback():
    called=[]
    pred,trace=controller_replay(Query(),np.full((20,2),.9),[.8,.9,.95],
        feedback=lambda i:called.append(i) or 0,budget=.2,delay=3)
    assert called==[4,9,14]
    for n in range(1,21):assert sum(r['queried'] for r in trace[:n])<=int(n*.2)
    assert trace[6]['delivered']==0 and trace[7]['delivered']==1
    assert all(pred==1)


def test_stale_and_dissimilar_reviews_have_less_weight():
    rows=[dict(index=9,pair=np.array([.5,.5])),dict(index=0,pair=np.array([.5,.5])),dict(index=9,pair=np.array([0,0]))]
    w=relevance_weights(np.array([.5,.5]),rows,10)
    assert w[0]>w[1] and w[0]>w[2]


def test_test_replay_does_not_train():
    class Capture(Query):
        memory=[]
    agent=Capture()
    controller_replay(agent,np.full((4,2),.2),[.8,.9,.95])
    assert agent.memory==[]
    with pytest.raises(ValueError):
        controller_replay(agent,np.full((4,2),.2),[.8,.9,.95],delay=0)
