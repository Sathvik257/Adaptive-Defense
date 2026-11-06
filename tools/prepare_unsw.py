"""Prepare a conflict-filtered UNSW-NB15 input for custom LOAO evaluation."""
import argparse
import json
from pathlib import Path
import pandas as pd
from adaptive_defense.data import validate, feature_columns
from adaptive_defense.unsw import inspect_pair

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    report_path = output.with_suffix(".audit.json")
    if output.exists() or report_path.exists():
        raise ValueError("Choose a new output path.")
    report = inspect_pair(args.train, args.test)
    frame = validate(pd.concat([pd.read_csv(args.train), pd.read_csv(args.test)], ignore_index=True))
    frame["attack_cat"] = frame.attack_cat.str.lower().replace({"backdoors": "backdoor"})
    fingerprints = pd.util.hash_pandas_object(frame[feature_columns(frame)], index=False).astype(str)
    grouped = frame.assign(_fingerprint=fingerprints).groupby("_fingerprint")
    conflicts = (grouped.label.nunique() > 1) | (grouped.attack_cat.nunique() > 1)
    excluded = fingerprints.isin(conflicts[conflicts].index)
    report.update(download_source=args.source, excluded_rows=int(excluded.sum()),
                  retained_rows=int((~excluded).sum()),
                  preparation_policy="Exclude all conflicting feature groups before custom LOAO. Preserve original files.")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[~excluded].to_csv(output, index=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {output} and {report_path}")

if __name__ == "__main__":
    main()
