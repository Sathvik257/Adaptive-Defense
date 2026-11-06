"""Audit prepared UNSW-NB15 CSVs before custom LOAO experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from .data import validate, feature_columns

FAMILIES = {"normal", "fuzzers", "analysis", "backdoor", "backdoors", "dos", "exploits", "generic", "reconnaissance", "shellcode", "worms"}

def inspect_pair(train_path, test_path):
    frames = []
    reports = []
    for path in (Path(train_path), Path(test_path)):
        raw = pd.read_csv(path)
        raw.columns = raw.columns.str.strip().str.lower()
        if not {"label", "attack_cat", "proto", "service", "state", "dur", "spkts", "dpkts"}.issubset(raw.columns):
            raise ValueError("Expected prepared UNSW-NB15 CSV with named flow features and targets.")
        if raw.empty or raw.attack_cat.isna().any():
            raise ValueError("Empty data or missing attack categories.")
        categories = raw.attack_cat.astype(str).str.strip().str.lower()
        if not set(categories).issubset(FAMILIES):
            raise ValueError("Unexpected UNSW-NB15 attack categories.")
        labels = pd.to_numeric(raw.label, errors="raise")
        if ((labels == 0) != (categories == "normal")).any():
            raise ValueError("Normal category and binary label disagree.")
        frame = validate(raw)
        frame["attack_cat"] = frame.attack_cat.str.lower().replace({"backdoors": "backdoor"})
        columns = feature_columns(frame)
        fingerprints = pd.util.hash_pandas_object(frame[columns], index=False).astype(str)
        frame["_fingerprint"] = fingerprints
        reports.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "rows": len(frame), "features": len(columns),
                        "categories": frame.attack_cat.value_counts().to_dict(),
                        "duplicate_feature_rows": int(fingerprints.duplicated().sum())})
        frames.append(frame)
    if feature_columns(frames[0]) != feature_columns(frames[1]):
        raise ValueError("Training and testing feature schemas differ.")
    combined = pd.concat(frames, ignore_index=True)
    grouped = combined.groupby("_fingerprint")
    conflicting = (grouped.label.nunique() > 1) | (grouped.attack_cat.nunique() > 1)
    overlap = len(set(frames[0]._fingerprint) & set(frames[1]._fingerprint))
    return {"dataset": "UNSW-NB15 candidate files", "source": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
            "files": reports, "cross_file_feature_overlap": overlap,
            "conflicting_feature_groups": int(conflicting.sum()),
            "feature_labels_consistent": not bool(conflicting.any()),
            "protocol": "Existing experiment and benchmark commands create custom deduplicated LOAO splits. Official partition membership is superseded.",
            "provenance": "Local hashes identify inputs. Schema checks do not authenticate their source."}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = inspect_pair(args.train, args.test)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    if not report["feature_labels_consistent"]:
        raise SystemExit("Resolve conflicting feature labels before training.")

if __name__ == "__main__":
    main()
