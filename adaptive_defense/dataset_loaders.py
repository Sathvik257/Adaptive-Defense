"""Dataset loaders and sliding window builders for Yahoo, SWaT, and WADI.

Matches the experimental setup of Yang et al. (ALA 2023 / arXiv:2312.01488v1):
- Normalization into [0, 1] range using training-only statistics.
- Sliding window with length tau and stride=1 (point-adjust labeling).
- Benign-only split for Autoencoder training.
- Small mixed normal/abnormal subset for reinforcement learning threshold training.
"""
from pathlib import Path
import glob
import numpy as np
import pandas as pd


def create_sliding_windows(features, labels=None, tau=10):
    """Create stride-1 sliding windows of length tau.

    Parameters:
        features: (N, m) array of normalized values.
        labels: (N,) binary array (0=normal, 1=anomaly). If None, returns no labels.
        tau: window length.

    Returns:
        windows: (N - tau + 1, tau * m) flattened window array.
        window_labels: (N - tau + 1,) binary labels (1 if ANY point in window is 1).
    """
    features = np.asarray(features, dtype=np.float32)
    n, m = features.shape
    if n < tau:
        raise ValueError(f"Feature length {n} must be >= window length {tau}.")
    
    # Efficient stride-1 sliding window view
    raw_windows = np.lib.stride_tricks.sliding_window_view(features, window_shape=tau, axis=0)
    # Shape is (n - tau + 1, m, tau) -> transpose to (n - tau + 1, tau, m) -> flatten to (n - tau + 1, tau * m)
    windows = raw_windows.transpose(0, 2, 1).reshape(n - tau + 1, tau * m).copy()

    if labels is not None:
        labels = np.asarray(labels, dtype=int)
        label_windows = np.lib.stride_tricks.sliding_window_view(labels, window_shape=tau, axis=0)
        # Point-adjust labeling: anomalous if any point in the sequence is anomalous
        window_labels = (label_windows.max(axis=1) > 0).astype(int)
        return windows, window_labels
    
    return windows, None


def load_yahoo_benchmark(root_dir=None, tau=10, test_ratio=0.3, max_files=None, seed=42):
    """Load Yahoo A1Benchmark (67 univariate time series).

    Splits each series temporally:
    - First (1 - test_ratio) is used for training/validation.
    - Remaining test_ratio is held out for testing.
    """
    if root_dir is None:
        root_dir = Path(__file__).resolve().parents[1] / "data" / "raw" / "Yahoo" / "A1Benchmark"
    root_dir = Path(root_dir)
    files = sorted(glob.glob(str(root_dir / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No Yahoo CSV files found in {root_dir}")
    if max_files:
        files = files[:max_files]

    rng = np.random.default_rng(seed)
    train_windows_list, test_windows_list = [], []
    train_labels_list, test_labels_list = [], []

    for fpath in files:
        df = pd.read_csv(fpath)
        values = df["value"].to_numpy(dtype=np.float32).reshape(-1, 1)
        labels = df["is_anomaly"].to_numpy(dtype=int)

        n = len(values)
        split_idx = int(n * (1.0 - test_ratio))
        if split_idx <= tau:
            continue

        train_vals, test_vals = values[:split_idx], values[split_idx:]
        train_labs, test_labs = labels[:split_idx], labels[split_idx:]

        # Min-max normalization fitted on train portion only
        vmin = train_vals.min()
        vmax = train_vals.max()
        scale = max(float(vmax - vmin), 1e-6)
        train_norm = np.clip((train_vals - vmin) / scale, 0.0, 1.0)
        test_norm = np.clip((test_vals - vmin) / scale, 0.0, 1.0)

        # Create windows
        w_train, l_train = create_sliding_windows(train_norm, train_labs, tau=tau)
        train_windows_list.append(w_train)
        train_labels_list.append(l_train)

        if len(test_norm) >= tau:
            w_test, l_test = create_sliding_windows(test_norm, test_labs, tau=tau)
            test_windows_list.append(w_test)
            test_labels_list.append(l_test)

    all_train_w = np.concatenate(train_windows_list, axis=0)
    all_train_l = np.concatenate(train_labels_list, axis=0)
    all_test_w = np.concatenate(test_windows_list, axis=0)
    all_test_l = np.concatenate(test_labels_list, axis=0)

    # Benign-only training for Autoencoder
    benign_mask = (all_train_l == 0)
    ae_train = all_train_w[benign_mask]

    # Small mixed subset for ADT / U-CADT policy training (~0.35% as per paper)
    # Ensure subset contains both normal and anomalous examples
    target_rl_samples = max(256, int(len(all_train_w) * 0.0035))
    anomaly_indices = np.where(all_train_l == 1)[0]
    normal_indices = np.where(all_train_l == 0)[0]
    
    n_ano = min(len(anomaly_indices), target_rl_samples // 4)
    n_norm = target_rl_samples - n_ano
    chosen_ano = rng.choice(anomaly_indices, size=n_ano, replace=False) if n_ano > 0 else np.array([], dtype=int)
    chosen_norm = rng.choice(normal_indices, size=n_norm, replace=False)
    rl_indices = rng.permutation(np.concatenate([chosen_ano, chosen_norm]))
    
    rl_train_w = all_train_w[rl_indices]
    rl_train_l = all_train_l[rl_indices]

    # Validation set for static threshold selection (10% of test size or from train)
    val_size = min(2000, len(all_test_w) // 5)
    val_indices = rng.choice(len(all_test_w), size=val_size, replace=False)
    val_w = all_test_w[val_indices]
    val_l = all_test_l[val_indices]

    return {
        "dataset": "Yahoo A1Benchmark",
        "m": 1,
        "tau": tau,
        "dimension": tau * 1,
        "ae_train": ae_train,
        "rl_train": (rl_train_w, rl_train_l),
        "val": (val_w, val_l),
        "test": (all_test_w, all_test_l),
        "stats": {
            "ae_train_windows": len(ae_train),
            "rl_train_windows": len(rl_train_w),
            "test_windows": len(all_test_w),
            "test_anomaly_rate": float(all_test_l.mean()),
        }
    }


def load_swat_benchmark(root_dir=None, tau=12, max_test_rows=None, seed=42):
    """Load SWaT (Secure Water Treatment) dataset (51 sensors/actuators).

    - swat_train.csv: 496,800 normal points (used for AE training).
    - swat_test.csv: 449,919 points under attack scenarios.
    """
    if root_dir is None:
        root_dir = Path(__file__).resolve().parents[1] / "data" / "raw" / "SWaT"
    root_dir = Path(root_dir)
    train_path = root_dir / "swat_train.csv"
    test_path = root_dir / "swat_test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"SWaT CSV files not found in {root_dir}")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path, nrows=max_test_rows)

    # Clean column names (strip whitespace)
    df_train.columns = df_train.columns.str.strip()
    df_test.columns = df_test.columns.str.strip()

    # Identify label column in test
    label_col = [c for c in df_test.columns if "attack" in c.lower() or "label" in c.lower()][0]
    test_labels_raw = df_test[label_col].to_numpy()
    # Normalize labels: 'Attack' or 1 -> 1, 'Normal' or 0 -> 0
    if test_labels_raw.dtype == object:
        test_labels = (test_labels_raw != "Normal").astype(int)
    else:
        test_labels = (test_labels_raw > 0).astype(int)

    feature_cols = [c for c in df_train.columns if c != label_col and c != "Timestamp"]
    m = len(feature_cols)

    train_vals = df_train[feature_cols].to_numpy(dtype=np.float32)
    test_vals = df_test[feature_cols].to_numpy(dtype=np.float32)

    # Normalize to [0, 1] using training statistics
    vmin = train_vals.min(axis=0)
    vmax = train_vals.max(axis=0)
    scale = np.maximum(vmax - vmin, 1e-6)

    train_norm = np.clip((train_vals - vmin) / scale, 0.0, 1.0)
    test_norm = np.clip((test_vals - vmin) / scale, 0.0, 1.0)

    # Build windows
    ae_train, _ = create_sliding_windows(train_norm, None, tau=tau)
    test_w, test_l = create_sliding_windows(test_norm, test_labels, tau=tau)

    # Small mixed subset for ADT / U-CADT policy (~0.11% as per paper)
    rng = np.random.default_rng(seed)
    target_rl_samples = max(256, int(len(test_w) * 0.0011))
    anomaly_indices = np.where(test_l == 1)[0]
    normal_indices = np.where(test_l == 0)[0]

    n_ano = min(len(anomaly_indices), target_rl_samples // 4)
    n_norm = target_rl_samples - n_ano
    chosen_ano = rng.choice(anomaly_indices, size=n_ano, replace=False)
    chosen_norm = rng.choice(normal_indices, size=n_norm, replace=False)
    rl_indices = rng.permutation(np.concatenate([chosen_ano, chosen_norm]))

    rl_train_w = test_w[rl_indices]
    rl_train_l = test_l[rl_indices]

    # Validation subset for static threshold selection
    val_size = min(5000, len(test_w) // 10)
    val_indices = rng.choice(len(test_w), size=val_size, replace=False)
    val_w = test_w[val_indices]
    val_l = test_l[val_indices]

    return {
        "dataset": "Secure Water Treatment (SWaT)",
        "m": m,
        "tau": tau,
        "dimension": tau * m,
        "ae_train": ae_train,
        "rl_train": (rl_train_w, rl_train_l),
        "val": (val_w, val_l),
        "test": (test_w, test_l),
        "stats": {
            "ae_train_windows": len(ae_train),
            "rl_train_windows": len(rl_train_w),
            "test_windows": len(test_w),
            "test_anomaly_rate": float(test_l.mean()),
        }
    }

