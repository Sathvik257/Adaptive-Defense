"""Read-only experiment artifact access, independent of the web framework."""
from dataclasses import dataclass
import json
from pathlib import Path
import pandas as pd

REQUIRED_FILES = ("summary.json", "metrics.csv", "predictions.csv", "threshold_trace.csv",
    "autoencoder_history.csv", "controller_history.csv", "pipeline.pkl",
    "autoencoder.pt", "dqn.pt", "dqn_adapted.pt")

@dataclass(frozen=True)
class Experiment:
    path: Path
    summary: dict
    metrics: pd.DataFrame
    predictions: pd.DataFrame

    @property
    def is_demo(self):
        return "SYNTHETIC" in self.summary["dataset"]

def discover_runs(directory):
    return sorted(path.parent for path in Path(directory).glob("*/summary.json"))

def load_experiment(path):
    path = Path(path)
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise ValueError("Incomplete experiment: " + ", ".join(missing))
    try:
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        metrics = pd.read_csv(path / "metrics.csv")
        predictions = pd.read_csv(path / "predictions.csv")
        for key in ("dataset", "audit", "configuration", "feature_columns", "seed", "limitations"):
            if key not in summary:
                raise ValueError(f"Experiment summary is missing {key}.")
        required = {"model", "recall", "precision", "f1", "false_positive_rate", "known_recall", "unseen_recall", "tn", "fp", "fn", "tp"}
        if not required.issubset(metrics.columns) or metrics.empty or metrics.model.duplicated().any():
            raise ValueError("Invalid or duplicated model metrics.")
        if not {"attack_cat", "label", "anomaly_score", *metrics.model}.issubset(predictions.columns):
            raise ValueError("Prediction records do not match the model metrics.")
    except (OSError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Cannot read experiment {path.name}: {exc}") from exc
    return Experiment(path, summary, metrics, predictions)
