"""dataset_loader.py

Real cybersecurity dataset ingestion + preprocessing utilities.

Supports:
- CIC-IDS2017 (primary)
- NSL-KDD (optional)

Requirements handled:
- load CSV
- clean missing values
- normalize/scale numeric features
- train/test split (stratified)
- map dataset labels into unified classes used by the project:
  DDoS, Brute Force, Port Scan, Botnet, Web Attack, Infiltration, Benign Traffic

This module is designed to be robust to dataset schema variations.
It provides best-effort loading and raises friendly errors when
required columns/datasets are missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DatasetBundle:
    train_X: np.ndarray
    test_X: np.ndarray
    train_y: np.ndarray
    test_y: np.ndarray
    feature_columns: List[str]
    label_mapping: Dict[str, str]


def _read_csv(csv_path: str | Path):
    import pandas as pd  # local import (optional dependency)

    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {p}")
    return pd.read_csv(p)


def _pick_label_column(df) -> str:
    # CIC-IDS2017: common label column names are 'Label' or 'label'
    # Normalize column names to be tolerant of leading/trailing spaces.
    df_cols_norm = {str(c).strip(): c for c in df.columns}

    for cand in ["Label", "label", "attack", "Attack", "Class", "class", "Target", "target"]:
        key = str(cand).strip()
        if key in df_cols_norm:
            return df_cols_norm[key]

    # Sometimes headers contain weird control chars; try a best-effort contains search
    for c in df.columns:
        cn = str(c).strip().lower()
        if cn in {"label", "attack", "class", "target"}:
            return c

    # NSL-KDD: often has 'label'
    for col in df.columns:
        if str(col).lower() in {"label", "attack", "class"}:
            return col
    raise ValueError(
        "Could not find label column. Looked for: Label/label/attack/Class/class/Target. "
        f"Columns present: {list(df.columns)[:20]}..."
    )


def _clean_missing_values(df, strategy: str = "median"):
    # Strategy best-effort: numeric median; categorical mode; then fill remaining.
    df = df.copy()
    num_cols = [c for c in df.columns if df[c].dtype.kind in "biufc"]
    cat_cols = [c for c in df.columns if c not in num_cols]

    for c in num_cols:
        if strategy == "median":
            v = df[c].median()
            df[c] = df[c].fillna(v)
        else:
            df[c] = df[c].fillna(0)

    for c in cat_cols:
        if df[c].isna().any():
            mode = df[c].mode(dropna=True)
            if len(mode) > 0:
                df[c] = df[c].fillna(mode.iloc[0])
            else:
                df[c] = df[c].fillna("unknown")

    # final safety
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df


def _stratified_split(X: np.ndarray, y: np.ndarray, test_size: float, seed: int):
    from sklearn.model_selection import train_test_split

    train_X, test_X, train_y, test_y = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    return train_X, test_X, train_y, test_y


def _scale_numeric(train_X: np.ndarray, test_X: np.ndarray):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    train_Xs = scaler.fit_transform(train_X)
    test_Xs = scaler.transform(test_X)
    return train_Xs, test_Xs


# -------------------
# Label mapping
# -------------------


def map_cic_ids_2017_label(raw: str) -> str:
    """Map CIC-IDS2017 Label values to project classes.

    CIC-IDS2017 label values typically include:
    - BENIGN
    - DoS attacks: DoS Hulk, DoS Slowhttptest, DoS GoldenEye, DoS... (DDoS)
    - Brute Force: FTP-Patator, SSH-Patator
    - Port Scan: PortScan
    - Botnet: Bot
    - Web Attack: Web attacks (Web Attack) incl. SQL injection, XSS, etc.
    - Infiltration: Infiltration

    For robustness we use keyword heuristics.
    """
    s = str(raw).strip()
    sl = s.lower()

    if sl == "benign" or sl == "benign.":
        return "Benign Traffic"
    if "dos" in sl or "ddos" in sl or "slowhttptest" in sl or "hulk" in sl or "goldeneye" in sl:
        return "DDoS"
    if "ftp-patator" in sl or "ssh-patator" in sl or "brute" in sl:
        return "Brute Force"
    if "portscan" in sl or "port scan" in sl:
        return "Port Scan"
    if "bot" in sl:
        return "Botnet"
    if "web" in sl or "xss" in sl or "sql" in sl or "sql injection" in sl or "injection" in sl:
        return "Web Attack"
    if "infiltration" in sl:
        return "Infiltration"

    # fallback: if already one of expected labels
    for cls in [
        "DDoS",
        "Brute Force",
        "Port Scan",
        "Botnet",
        "Web Attack",
        "Infiltration",
        "Benign Traffic",
    ]:
        if cls.lower() == sl:
            return cls

    # Unknown label: treat as Benign (safer for demo) rather than crash.
    return "Benign Traffic"


def map_nsl_kdd_label(raw: str) -> str:
    """Map NSL-KDD attack categories to project classes using heuristics."""
    s = str(raw).strip()
    sl = s.lower()
    if sl in {"normal.", "normal", "benign"}:
        return "Benign Traffic"
    if "back" in sl or "land" in sl or "neptune" in sl or "teardrop" in sl or "smurf" in sl:
        return "DDoS"
    if "warez" in sl or "guess" in sl or "brute" in sl or "ftp" in sl or "ssh" in sl:
        return "Brute Force"
    if "portsweep" in sl or "ipsweep" in sl or "nmap" in sl or "mscan" in sl or "satan" in sl:
        return "Port Scan"
    if "bot" in sl:
        return "Botnet"
    if "apache2" in sl or "http" in sl or "web" in sl or "xss" in sl or "sql" in sl:
        return "Web Attack"
    if "infiltration" in sl:
        return "Infiltration"
    # default fallback
    return "Benign Traffic"


# -------------------
# CIC loader
# -------------------


def load_cic_ids_2017_csv(
    csv_path: str | Path,
    *,
    test_size: float = 0.2,
    seed: int = 42,
    min_rows_required: int = 2000,
    feature_columns_whitelist: Optional[List[str]] = None,
) -> DatasetBundle:
    import pandas as pd  # type: ignore

    df = _read_csv(csv_path)
    if len(df) < min_rows_required:
        raise ValueError(f"CIC-IDS2017 dataset too small: {len(df)} rows (< {min_rows_required}).")

    label_col = _pick_label_column(df)

    # Map labels to unified classes
    mapped_labels = df[label_col].apply(map_cic_ids_2017_label).astype(str)
    df = df.copy()
    df["_mapped_label"] = mapped_labels

    # Choose features
    if feature_columns_whitelist:
        missing = [c for c in feature_columns_whitelist if c not in df.columns]
        if missing:
            raise ValueError(f"Requested feature columns missing from dataset: {missing[:10]}...")
        feature_cols = list(feature_columns_whitelist)
    else:
        # Best-effort: choose numeric columns excluding obvious ID-like fields
        numeric_cols = [c for c in df.columns if c != label_col and df[c].dtype.kind in "biufc"]
        exclude = {"Fwd Header Length.1", "id"}
        feature_cols = [c for c in numeric_cols if c not in exclude]

    # Drop non-feature columns
    X_df = df[feature_cols]

    # Ensure numeric
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = _clean_missing_values(X_df)

    X = X_df.to_numpy(dtype=np.float32)
    y = df["_mapped_label"].to_numpy(dtype=str)

    # Scale
    X_train, X_test, y_train, y_test = None, None, None, None
    X_scaled, X_scaled2 = None, None
    # Split first then scale to avoid leakage
    X_train, X_test, y_train, y_test = _stratified_split(X, y, test_size=test_size, seed=seed)
    X_train_s, X_test_s = _scale_numeric(X_train, X_test)

    return DatasetBundle(
        train_X=X_train_s,
        test_X=X_test_s,
        train_y=y_train,
        test_y=y_test,
        feature_columns=feature_cols,
        label_mapping={"_mapped_label": "mapped"},
    )


# -------------------
# NSL loader (optional)
# -------------------


def load_nsl_kdd_csv(
    csv_path: str | Path,
    *,
    test_size: float = 0.2,
    seed: int = 42,
    min_rows_required: int = 2000,
    feature_columns_whitelist: Optional[List[str]] = None,
) -> DatasetBundle:
    import pandas as pd  # type: ignore

    df = _read_csv(csv_path)
    if len(df) < min_rows_required:
        raise ValueError(f"NSL-KDD dataset too small: {len(df)} rows (< {min_rows_required}).")

    label_col = _pick_label_column(df)
    df = df.copy()
    df["_mapped_label"] = df[label_col].apply(map_nsl_kdd_label).astype(str)

    if feature_columns_whitelist:
        missing = [c for c in feature_columns_whitelist if c not in df.columns]
        if missing:
            raise ValueError(f"Requested feature columns missing from dataset: {missing[:10]}...")
        feature_cols = list(feature_columns_whitelist)
    else:
        numeric_cols = [c for c in df.columns if c != label_col and df[c].dtype.kind in "biufc"]
        feature_cols = numeric_cols

    # include one-hot for categorical if any exist in feature cols
    X_df = df[feature_cols]
    X_df = _clean_missing_values(X_df)

    # Coerce to numeric best-effort
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = _clean_missing_values(X_df)

    X = X_df.to_numpy(dtype=np.float32)
    y = df["_mapped_label"].to_numpy(dtype=str)

    X_train, X_test, y_train, y_test = _stratified_split(X, y, test_size=test_size, seed=seed)
    X_train_s, X_test_s = _scale_numeric(X_train, X_test)

    return DatasetBundle(
        train_X=X_train_s,
        test_X=X_test_s,
        train_y=y_train,
        test_y=y_test,
        feature_columns=feature_cols,
        label_mapping={"_mapped_label": "mapped"},
    )

