"""ADT equations 4-8 and episode-end DQN updates (Yang et al., 2023).

This implementation requires explicit post-decision labels for paper-mode feedback.
It is separate from the legacy eight-state, three-action controller.
"""
from collections import deque
import copy
import numpy as np
import torch
from torch import nn


def windows(features, labels, tau=1):
    """Stride-one windows, labelled anomalous when any constituent is anomalous.

    Call separately on chronological partitions; never concatenate independent series.
    """
    x, y = np.asarray(features), np.asarray(labels)
    if x.ndim != 2 or y.shape != (len(x),) or not 1 <= tau <= len(x):
        raise ValueError("Require a feature matrix, aligned labels and 1 <= tau <= rows.")
    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Features must be finite and labels binary.")
    return (np.stack([x[i:i+tau].reshape(-1) for i in range(len(x)-tau+1)]),
            np.array([y[i:i+tau].max() for i in range(len(x)-tau+1)], dtype=int))


class History:
    """Only completed decisions enter the next state; no current/future label access."""
    def __init__(self, k=2, alpha=.9):
        if k < 1 or not 0 <= alpha <= 1:
            raise ValueError("k must be positive and alpha in [0, 1].")
        self.k, self.alpha = k, alpha
        self.items = deque(maxlen=k)

    def state(self):
        if not self.items:
            return np.zeros(6, dtype=np.float32)
        scores = np.array([item[0] for item in self.items])
        # Missing startup history is not invented: fractions retain the k denominator.
        return np.array([scores.mean(), scores.var(ddof=1) if len(scores)>1 else 0,
                         *self.counts()/self.k], dtype=np.float32)

    def counts(self):
        counts = np.zeros(4, dtype=float)  # TP, TN, FP, FN
        for _, prediction, label in self.items:
            if label is not None:
                counts[{(1,1):0, (0,0):1, (1,0):2, (0,1):3}[(prediction,label)]] += 1
        return counts

    def observe(self, score, action, label=None):
        if not np.isfinite(score) or not 0 <= score <= 1 or action not in (0,1):
            raise ValueError("Require a bounded score and binary threshold action.")
        if label is not None and label not in (0,1):
            raise ValueError("Feedback must be binary or absent.")
        prediction = int(score > action)
        self.items.append((float(score), prediction, label))
        tp, tn, fp, fn = self.counts()
        reward = self.alpha*(tp-fp-fn)+(1-self.alpha)*tn
        return prediction, float(reward)


class ADT:
    def __init__(self, seed=42, gamma=.99, learning_rate=1e-4, capacity=20000, hidden=(32,32,16)):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        widths = (6, *hidden, 2)
        layers = []
        for i, (left, right) in enumerate(zip(widths, widths[1:])):
            layers.append(nn.Linear(left, right))
            if i < len(widths)-2:
                layers.append(nn.ReLU())
        self.online = nn.Sequential(*layers)
        self.target = copy.deepcopy(self.online)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=capacity)
        self.gamma = gamma

    def action(self, state, epsilon=0):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(2))
        with torch.no_grad():
            return int(self.online(torch.as_tensor(state, dtype=torch.float32)).argmax())

    def update(self, batch_size=64):
        if len(self.memory) < batch_size:
            return None
        batch = [self.memory[i] for i in self.rng.choice(len(self.memory), batch_size, replace=False)]
        state, action, reward, next_state, terminal = map(np.asarray, zip(*batch))
        with torch.no_grad():
            target = torch.tensor(reward, dtype=torch.float32) + self.gamma * (
                1-torch.tensor(terminal, dtype=torch.float32)) * self.target(
                    torch.tensor(next_state, dtype=torch.float32)).max(1).values
        predicted = self.online(torch.tensor(state, dtype=torch.float32)).gather(
            1, torch.tensor(action, dtype=torch.long)[:,None]).squeeze(1)
        loss = nn.functional.mse_loss(predicted, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())


def replay(agent, scores, labels=None, *, k=2, alpha=.9, training=False,
           epsilon=0, action_interval=1):
    scores = np.asarray(scores)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)):
        raise ValueError("Provide nonempty bounded anomaly scores.")
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape != scores.shape or not np.isin(labels,[0,1]).all():
            raise ValueError("Feedback labels must align with scores and be binary.")
    if training and labels is None:
        raise ValueError("Training requires feedback.")
    if action_interval < 1 or (not training and action_interval != 1):
        raise ValueError("Inference requires l=1; training requires positive l.")
    history, rows = History(k,alpha), []
    for t, score in enumerate(scores):
        state = history.state()
        if t % action_interval == 0:
            action = agent.action(state,epsilon)
        prediction,reward = history.observe(score,action,None if labels is None else int(labels[t]))
        next_state = history.state()
        if training:
            agent.memory.append((state.copy(),action,reward,next_state.copy(),t==len(scores)-1))
        rows.append({"index":t,"threshold":action,"prediction":prediction,"score":float(score),
                     "reward":reward if labels is not None else None,
                     **dict(zip(["mean","variance","tp_fraction","tn_fraction","fp_fraction","fn_fraction"],state.tolist()))})
    return np.array([r["prediction"] for r in rows]), rows
