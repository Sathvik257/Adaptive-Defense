"""Dataset validation, group-safe splits and synthetic demonstration data."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EXCLUDED = {"id", "label", "attack_cat", "srcip", "dstip", "stime", "ltime", "timestamp"}

def validate(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names are not supported.")
    if not {"label", "attack_cat"}.issubset(df.columns):
        raise ValueError("Training CSV requires label and attack_cat columns.")
    df["label"] = pd.to_numeric(df.label, errors="raise")
    if not df.label.isin([0, 1]).all():
        raise ValueError("label must contain only 0 (normal) or 1 (attack).")
    df["attack_cat"] = df.attack_cat.fillna("Normal").astype(str).str.strip()
    if ((df.label == 1) & df.attack_cat.str.lower().isin(["normal", ""])).any():
        raise ValueError("Attack rows require a non-normal attack category.")
    df.loc[df.label == 0, "attack_cat"] = "Normal"
    return df.reset_index(drop=True)

def feature_columns(df):
    return [c for c in df if c not in EXCLUDED and not c.startswith("_")]

def clean_features(df, columns):
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    return df[columns].replace([np.inf, -np.inf], np.nan)

def make_preprocessor(df, columns):
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in columns if c not in numeric]
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                              ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                                  ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ], sparse_threshold=0)

def split_data(df, holdout, seed):
    """Deterministic category-stratified split after feature-level deduplication.

    CSV row order is not treated as time. These are synthetic replay splits.
    Official partition membership is intentionally superseded by this protocol.
    """
    df = validate(df)
    columns = feature_columns(df)
    if not columns:
        raise ValueError("No usable features.")
    df["_fingerprint"] = pd.util.hash_pandas_object(df[columns], index=False).astype(str)
    conflicts = df.groupby("_fingerprint").label.nunique()
    if (conflicts > 1).any():
        raise ValueError("Identical feature records have conflicting labels; resolve them first.")
    if (df.groupby('_fingerprint').attack_cat.nunique() > 1).any():
        raise ValueError('Identical feature records have conflicting attack families; resolve them first.')
    original = len(df)
    df = df.drop_duplicates("_fingerprint").copy()
    holdout = {x.strip().lower() for x in holdout}
    available = set(df.loc[df.label == 1, "attack_cat"].str.lower())
    if not holdout or not holdout.issubset(available):
        raise ValueError(f"Unknown held-out category. Available: {sorted(available)}")
    if not available - holdout:
        raise ValueError("At least one attack category must remain for development.")
    rng = np.random.default_rng(seed)
    pieces = {k: [] for k in ["train", "validation", "adaptation", "test"]}
    for category, group in df.groupby("attack_cat", sort=True):
        if len(group) < 20:
            raise ValueError(f"Category {category} has fewer than 20 unique records.")
        group = group.iloc[rng.permutation(len(group))]
        if category.lower() in holdout:
            cut = max(1, int(len(group) * .4))
            pieces["adaptation"].append(group.iloc[:cut])
            pieces["test"].append(group.iloc[cut:])
        else:
            a, b, c = [int(len(group)*p) for p in (.60, .75, .85)]
            for name, part in zip(pieces, [group.iloc[:a], group.iloc[a:b], group.iloc[b:c], group.iloc[c:]]):
                pieces[name].append(part)
    splits = {k: pd.concat(v).sample(frac=1, random_state=seed).reset_index(drop=True) for k, v in pieces.items()}
    audit = {"input_records": original, "unique_records": len(df), "duplicates_removed": original-len(df),
             "held_out_categories": sorted(holdout), "split_counts": {k: len(v) for k,v in splits.items()},
             "protocol": "custom category-stratified partitions; synthetic replay, not chronological"}
    return splits, columns, audit

def generate_demo(path, seed=42, rows=2400):
    """Artificial distributions; never evidence of real attack detection."""
    rng = np.random.default_rng(seed)
    categories = rng.choice(["Normal", "Exploits", "Fuzzers", "Reconnaissance"], rows, p=[.55,.20,.15,.10])
    shifts = np.array([{"Normal":0, "Exploits":2.0, "Fuzzers":1.3, "Reconnaissance":.7}[c] for c in categories])
    df = pd.DataFrame({"id":np.arange(rows), "dur":rng.lognormal(.3+shifts,.7),
        "spkts":rng.poisson(8+8*shifts), "dpkts":rng.poisson(6+3*shifts),
        "sbytes":rng.lognormal(6+shifts,.8), "dbytes":rng.lognormal(7-shifts*.2,.9),
        "rate":rng.lognormal(2+shifts,.5), "proto":rng.choice(["tcp","udp","icmp"],rows),
        "service":rng.choice(["http","dns","-"],rows), "state":rng.choice(["FIN","CON","INT"],rows),
        "attack_cat":categories, "label":(categories!="Normal").astype(int)})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path,index=False)
    return path
