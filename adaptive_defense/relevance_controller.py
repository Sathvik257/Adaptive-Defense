"""Experimental relevance-weighted feedback controller; separate from paper ADT."""
from collections import deque
import copy
import numpy as np
import torch
from torch import nn
from .adt import ADT


class RelevanceAgent(ADT):
    def __init__(self,seed=42):
        super().__init__(seed=seed)
        self.online=nn.Sequential(nn.Linear(12,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,6))
        self.target=copy.deepcopy(self.online)
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=.001)

    def action(self,state,epsilon=0):
        if self.rng.random()<epsilon:return int(self.rng.integers(6))
        with torch.no_grad():return int(self.online(torch.as_tensor(state,dtype=torch.float32)).argmax())


def relevance_weights(current, reviewed, now, age_scale=128.,distance_scale=.25):
    """Weights depend only on observed score pairs and elapsed replay rows."""
    return np.asarray([np.exp(-(now-r['index'])/age_scale-np.linalg.norm(current-r['pair'])/distance_scale)
                       for r in reviewed])


def controller_replay(agent,pairs,thresholds,*,feedback=None,training_labels=None,budget=.05,delay=32,
                      weighted=True,review_mode='learned',epsilon=0):
    pairs=np.asarray(pairs,dtype=float); thresholds=np.asarray(thresholds,dtype=float)
    if pairs.ndim!=2 or pairs.shape[1]!=2 or not len(pairs) or not np.isfinite(pairs).all() or ((pairs<0)|(pairs>1)).any():
        raise ValueError('Need finite rank pairs in [0,1].')
    if thresholds.shape!=(3,) or not np.isfinite(thresholds).all() or not 0<=budget<=1 or not isinstance(delay,int) or delay<1:
        raise ValueError('Invalid thresholds, budget or delay.')
    if review_mode not in ('learned','periodic'):raise ValueError('Unknown review mode.')
    if training_labels is not None:
        training_labels=np.asarray(training_labels)
        if training_labels.shape!=(len(pairs),) or not np.isin(training_labels,[0,1]).all():raise ValueError('Invalid training labels.')
    reviewed=deque(maxlen=32);pending=deque();history=deque(maxlen=32)
    queried=delivered=0;states=[];actions=[];rewards=[];predictions=[];trace=[]
    for t,pair in enumerate(pairs):
        while pending and pending[0][0]<=t:
            _,record=pending.popleft()
            label=feedback(record['index'])
            if label not in (0,1):raise ValueError('Feedback must be binary.')
            record['label']=int(label);reviewed.append(record);delivered+=1
        weights=relevance_weights(pair,reviewed,t) if weighted else np.ones(len(reviewed))
        counts=np.zeros(4)
        for w,r in zip(weights,reviewed):counts[{(1,1):0,(0,0):1,(1,0):2,(0,1):3}[(r['prediction'],r['label'])]]+=w/32
        credit=max(0,int(np.floor((t+1)*budget+1e-12))-queried)
        age=min((t-reviewed[-1]['index'])/128,1) if reviewed else 1
        state=np.r_[pair,abs(pair[0]-pair[1]),np.mean(history,axis=0) if history else [0,0],
                    min(credit,1),min(len(pending)/32,1),age,counts].astype(np.float32)
        action=agent.action(state,epsilon)
        if action not in range(6):raise ValueError('Invalid controller action.')
        pred=int(pair.max()>thresholds[action//2])
        query=feedback is not None and credit>0 and (review_mode=='periodic' or action%2==1)
        if query:
            queried+=1;pending.append((t+delay,dict(index=t,pair=pair.copy(),prediction=pred)))
        history.append(pair.copy());predictions.append(pred);states.append(state);actions.append(action)
        if training_labels is not None:
            # Offline simulator reward only. Unrequested labels never enter policy state.
            label=int(training_labels[t])
            rewards.append(({(1,1):1.,(0,0):.1,(1,0):-5.,(0,1):-1.}[(pred,label)])-.02*query)
        trace.append(dict(index=t,action=action,queried=bool(query),delivered=delivered,
                          effective_reviews=float(weights.sum()),prediction=pred))
    if training_labels is not None:
        for t in range(len(pairs)):
            agent.memory.append((states[t],actions[t],rewards[t],states[t+1] if t+1<len(pairs) else np.zeros(12,dtype=np.float32),t==len(pairs)-1))
    return np.asarray(predictions),trace
