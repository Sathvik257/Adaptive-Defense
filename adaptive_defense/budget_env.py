"""Offline cost-sensitive research environment; no network enforcement.

Queries assume a perfect, immediate analyst. This is a simulator assumption,
not a claim about human accuracy or response time. Labels only enter rewards
and explicitly requested feedback; they never enter observations.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BudgetedDefenseEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, features, labels, budget=20, miss_cost=10., alarm_cost=.5, query_cost=.3):
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels)
        if self.features.ndim != 2 or not len(self.features) or not np.isfinite(self.features).all():
            raise ValueError('Features must be a nonempty finite matrix.')
        if self.labels.shape != (len(self.features),) or not np.isin(self.labels, [0, 1]).all():
            raise ValueError('One binary label is required per record.')
        if not isinstance(budget, int) or budget < 0:
            raise ValueError('Budget must be a nonnegative integer.')
        if not np.isfinite([miss_cost, alarm_cost, query_cost]).all() or min(miss_cost, alarm_cost, query_cost) < 0:
            raise ValueError('Costs must be finite and nonnegative.')
        self.budget = budget
        self.costs = (miss_cost, alarm_cost, query_cost)
        self.action_space = spaces.Discrete(3)  # allow, alert, query
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.features.shape[1]+2,), dtype=np.float32)
        self.position = 0
        self.remaining = budget

    def _observation(self):
        if self.position == len(self.features):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return np.concatenate([self.features[self.position], [self.remaining/max(self.budget, 1),
                               self.position/len(self.features)]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position, self.remaining = 0, self.budget
        return self._observation(), {}

    def step(self, action):
        if self.position >= len(self.features):
            raise RuntimeError('Reset before starting another episode.')
        if not self.action_space.contains(action):
            raise ValueError('Choose allow=0, alert=1 or query=2.')
        truth = int(self.labels[self.position])
        miss, alarm, query = self.costs
        info = {'queried': False}
        if action == 2 and self.remaining > 0:
            self.remaining -= 1
            cost = query
            info.update(queried=True, analyst_label=truth)
        else:
            # An exhausted query falls back to alert and incurs query overhead.
            effective = 1 if action == 2 else action
            cost = miss*int(effective == 0 and truth == 1) + alarm*int(effective == 1 and truth == 0)
            if action == 2:
                cost += query
                info['budget_exhausted'] = True
        self.position += 1
        info.update(cost=float(cost), remaining=self.remaining)
        return self._observation(), -float(cost), self.position == len(self.features), False, info
