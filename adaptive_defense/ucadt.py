"""U-CADT: Uncertainty-Aware Continuous Adaptive Dynamic Thresholding.

Novel Dynamic Thresholding framework designed to overcome the core limitations
of Yang et al. (ALA 2023 / arXiv:2312.01488v1):
1. Autonomous & Feedback-Free: Does NOT require unrealistic ground-truth oracle
   labels (TP, FP, TN, FN) during inference. State is constructed from model
   epistemic uncertainty and temporal score dynamics.
2. Continuous/Multi-Scale Modulation: Eliminates harsh binary {0, 1} chattering
   by smoothly adapting a calibrated base threshold via multi-quantile offsets.
3. Stability Regularization: Reward penalizes threshold volatility to ensure smooth,
   robust adaptation under non-stationary baseline drift.
"""
from collections import deque
import copy
import numpy as np
import torch
from torch import nn


class UncertaintyAutoencoder(nn.Module):
    """Deep Symmetrical Autoencoder with Monte Carlo Dropout for Epistemic Uncertainty."""
    def __init__(self, dimension, latent=100, dropout_p=0.10):
        super().__init__()
        if dimension < 4 or latent < 1:
            raise ValueError("Input dimension must be >= 4 and latent >= 1.")
        
        d1 = max(dimension // 2, 2)
        d2 = max(dimension // 4, 1)

        self.encoder = nn.Sequential(
            nn.Linear(dimension, d1),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(d1, d2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(d2, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, d2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(d2, d1),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(d1, dimension),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def score_with_uncertainty(self, x_np, num_passes=5):
        """Estimate both mean L2 reconstruction error and epistemic uncertainty via MC Dropout."""
        self.train()  # Keep dropout active for MC sampling
        tensor_x = torch.as_tensor(x_np, dtype=torch.float32)
        batch_size = 1024
        scores, uncertainties = [], []

        with torch.no_grad():
            for i in range(0, len(tensor_x), batch_size):
                batch = tensor_x[i:i + batch_size]
                # Run M stochastic passes
                recons = torch.stack([self.forward(batch) for _ in range(num_passes)]) # (M, B, D)
                mean_recon = recons.mean(dim=0) # (B, D)
                
                # Reconstruction error (L2 norm per window)
                l2_error = torch.sqrt(torch.sum((batch - mean_recon) ** 2, dim=-1)).cpu().numpy()
                
                if num_passes > 1:
                    var_uncertainty = torch.mean(torch.var(recons, dim=0, unbiased=False), dim=-1).cpu().numpy()
                else:
                    var_uncertainty = np.zeros(len(batch), dtype=np.float32)
                
                scores.append(l2_error)
                uncertainties.append(var_uncertainty)

        return np.concatenate(scores), np.concatenate(uncertainties)


class UCADTHistory:
    """Feedback-Free State Tracker: Operates autonomously with ZERO oracle labels."""
    def __init__(self, k=5, base_threshold=0.5):
        self.k = max(k, 2)
        self.base_threshold = max(float(base_threshold), 1e-4)
        self.scores = deque(maxlen=k)
        self.uncertainties = deque(maxlen=k)
        self.thresholds = deque(maxlen=k)
        self.decisions = deque(maxlen=k)

    def state(self):
        """Constructs an 8-dimensional feedback-free state vector."""
        if not self.scores:
            return np.zeros(8, dtype=np.float32)

        s_arr = np.array(self.scores, dtype=np.float32)
        u_arr = np.array(self.uncertainties, dtype=np.float32)
        t_arr = np.array(self.thresholds, dtype=np.float32)
        d_arr = np.array(self.decisions, dtype=np.float32)

        mean_s = float(s_arr.mean())
        var_s = float(s_arr.var(ddof=1)) if len(s_arr) > 1 else 0.0
        trend_s = float((s_arr[-1] - s_arr[0]) / max(len(s_arr) - 1, 1))

        mean_u = float(u_arr.mean())
        var_u = float(u_arr.var(ddof=1)) if len(u_arr) > 1 else 0.0

        current_thresh_norm = float(t_arr[-1] / (self.base_threshold * 2.0))
        recent_alert_rate = float(d_arr.mean()) if len(d_arr) > 0 else 0.0
        conf_ratio = float(mean_s / max(mean_u * 100.0, 1e-4))

        return np.array([
            mean_s, var_s, trend_s,
            mean_u, var_u,
            current_thresh_norm, recent_alert_rate, conf_ratio
        ], dtype=np.float32)

    def update(self, score, uncertainty, threshold, decision):
        self.scores.append(float(score))
        self.uncertainties.append(float(uncertainty))
        self.thresholds.append(float(threshold))
        self.decisions.append(int(decision))


class UCADTAgent:
    """Multi-Scale Continuous-Modulation DQN Controller with Fine-Grained Quantile Adjustments."""
    # 7 fine-grained actions: [-25%, -15%, -5%, 0%, +5%, +15%, +25%]
    ACTION_FACTORS = np.array([-0.25, -0.15, -0.05, 0.0, +0.05, +0.15, +0.25], dtype=np.float32)

    def __init__(self, state_dim=8, base_threshold=0.5, seed=42, lr=1e-4, gamma=0.95, capacity=20000):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.base_threshold = float(base_threshold)
        self.n_actions = len(self.ACTION_FACTORS)
        
        # Q-Network
        self.online = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_actions)
        )
        self.target = copy.deepcopy(self.online)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.memory = deque(maxlen=capacity)
        self.gamma = gamma
        self.updates = 0

    def compute_threshold(self, action_idx):
        factor = float(self.ACTION_FACTORS[action_idx])
        return max(self.base_threshold * (1.0 + factor), 1e-5)

    def select_action(self, state, epsilon=0.0):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            t_state = torch.as_tensor(state, dtype=torch.float32)
            return int(self.online(t_state).argmax().item())

    def store_transition(self, s, a, r, next_s, done):
        self.memory.append((s, a, r, next_s, done))

    def update(self, batch_size=64):
        if len(self.memory) < batch_size:
            return None
        indices = self.rng.choice(len(self.memory), size=batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        t_states = torch.as_tensor(np.array(states), dtype=torch.float32)
        t_actions = torch.as_tensor(actions, dtype=torch.long).unsqueeze(1)
        t_rewards = torch.as_tensor(rewards, dtype=torch.float32)
        t_next = torch.as_tensor(np.array(next_states), dtype=torch.float32)
        t_dones = torch.as_tensor(dones, dtype=torch.float32)

        # Current Q
        q_pred = self.online(t_states).gather(1, t_actions).squeeze(1)

        # Target Q (Double DQN target)
        with torch.no_grad():
            next_actions = self.online(t_next).argmax(dim=1, keepdim=True)
            q_next = self.target(t_next).gather(1, next_actions).squeeze(1)
            target = t_rewards + self.gamma * (1.0 - t_dones) * q_next

        loss = nn.functional.smooth_l1_loss(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimizer.step()

        self.updates += 1
        if self.updates % 100 == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())


def train_autoencoder(train_windows, dimension, latent=100, epochs=20, batch_size=512, lr=1e-3, seed=42):
    """Train the benign-only Uncertainty Autoencoder."""
    torch.manual_seed(seed)
    model = UncertaintyAutoencoder(dimension=dimension, latent=latent, dropout_p=0.10)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    tensor_data = torch.as_tensor(train_windows, dtype=torch.float32)
    n_samples = len(tensor_data)
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples)
        epoch_losses = []
        for i in range(0, n_samples, batch_size):
            batch_idx = perm[i:i + batch_size]
            batch = tensor_data[batch_idx]
            recon = model(batch)
            loss = nn.functional.mse_loss(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            
    return model


def calibrate_base_threshold(val_scores, val_labels=None, quantile=0.98):
    """Calibrate base static threshold from benign validation scores or optimal F1 search."""
    if val_labels is not None and val_labels.sum() > 0:
        # Search candidate grid for highest F1
        candidates = np.quantile(val_scores, np.linspace(0.80, 0.999, 40))
        best_f1, best_thresh = -1.0, float(candidates[0])
        for cand in candidates:
            preds = (val_scores > cand).astype(int)
            tp = int(((val_labels == 1) & (preds == 1)).sum())
            fp = int(((val_labels == 0) & (preds == 1)).sum())
            fn = int(((val_labels == 1) & (preds == 0)).sum())
            if tp == 0:
                f1 = 0.0
            else:
                p = tp / max(tp + fp, 1)
                r = tp / max(tp + fn, 1)
                f1 = 2 * p * r / max(p + r, 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(cand)
        return best_thresh
    else:
        return float(np.quantile(val_scores, quantile))


def run_ucadt_inference(agent, scores, uncertainties, k=5):
    """Run fully autonomous, feedback-free dynamic thresholding inference."""
    history = UCADTHistory(k=k, base_threshold=agent.base_threshold)
    predictions = []
    threshold_trace = []

    for i in range(len(scores)):
        curr_score = scores[i]
        curr_unc = uncertainties[i]
        curr_state = history.state()
        
        # Policy action
        action_idx = agent.select_action(curr_state, epsilon=0.0)
        thresh = agent.compute_threshold(action_idx)
        
        # Decision
        pred = int(curr_score > thresh)
        predictions.append(pred)
        threshold_trace.append(thresh)
        
        # Autonomous update (no ground truth used!)
        history.update(curr_score, curr_unc, thresh, pred)

    return np.array(predictions, dtype=int), np.array(threshold_trace, dtype=np.float32)
