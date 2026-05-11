"""ml_attack_detector.py

Supervised ML attack classifier for real cybersecurity datasets.

Pipeline:
  engineered features -> classifier -> attack label + confidence

Recommended models:
  - RandomForestClassifier (always available if sklearn exists)
  - XGBoost optional (uses xgboost if installed)

This module provides:
  - train_detector(...)
  - evaluate_detector(...)
  - save/load model (joblib best-effort)

The rest of the repo must NOT depend on this module at import-time.
Callers should use lazy imports and fallback behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DetectorMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    class_report: Dict[str, Any]


@dataclass(frozen=True)
class DetectorPrediction:
    attack_type: str
    confidence: float  # 0..1
    predicted_class: str
    probs: Dict[str, float]


class MLDetector:
    def __init__(
        self,
        *,
        model: Any,
        feature_columns: List[str],
        class_order: List[str],
    ) -> None:
        self.model = model
        self.feature_columns = list(feature_columns)
        self.class_order = list(class_order)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (pred_labels, probs_max)."""
        probs = self.model.predict_proba(X)
        idx = probs.argmax(axis=1)
        max_probs = probs.max(axis=1)
        labels = self.model.classes_[idx]
        return labels, max_probs

    def predict_proba_one(self, x_row: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x_row.reshape(1, -1))[0]


def _maybe_use_xgboost(model_name: str):
    if model_name.lower() not in {"xgb", "xgboost", "xgboostclassifier"}:
        return None
    try:
        from xgboost import XGBClassifier  # type: ignore

        return XGBClassifier
    except Exception:
        return None


def train_detector(
    *,
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    feature_columns: List[str],
    class_order: List[str],
    model_type: str = "rf",
    seed: int = 42,
) -> Tuple[MLDetector, DetectorMetrics, np.ndarray]:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

    if model_type.lower() in {"rf", "randomforest"}:
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced",
        )
    else:
        xgb_cls = _maybe_use_xgboost(model_type)
        if xgb_cls is None:
            # fallback to RF
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                n_jobs=-1,
                random_state=seed,
                class_weight="balanced",
            )
        else:
            model = xgb_cls(
                n_estimators=400,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=seed,
                n_jobs=-1,
            )

    model.fit(train_X, train_y)

    pred_y = model.predict(test_X)

    acc = float(accuracy_score(test_y, pred_y))
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_y,
        pred_y,
        average="macro",
        zero_division=0,
    )

    # classification report
    report = {}
    try:
        from sklearn.metrics import classification_report

        report = classification_report(test_y, pred_y, output_dict=True, zero_division=0)
    except Exception:
        report = {}

    cm = confusion_matrix(test_y, pred_y, labels=class_order)

    metrics = DetectorMetrics(
        accuracy=acc,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        class_report=report,
    )

    detector = MLDetector(
        model=model,
        feature_columns=feature_columns,
        class_order=class_order,
    )

    return detector, metrics, cm


def save_detector(detector: MLDetector, model_path: str | Path) -> None:
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import joblib  # type: ignore

        joblib.dump(detector, str(model_path))
    except Exception:
        # best-effort: if joblib not available, do nothing.
        return


def load_detector(model_path: str | Path) -> Optional[MLDetector]:
    model_path = Path(model_path)
    if not model_path.exists():
        return None

    try:
        import joblib  # type: ignore

        obj = joblib.load(str(model_path))
        if isinstance(obj, MLDetector):
            return obj
        # allow backward compatibility if wrapper differs
        return obj
    except Exception:
        return None


def evaluate_and_save_artifacts(
    detector: MLDetector,
    *,
    test_X: np.ndarray,
    test_y: np.ndarray,
    class_order: List[str],
    cm: np.ndarray,
    out_dir: str | Path,
) -> Dict[str, float]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save confusion matrix plot if matplotlib exists
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(class_order)))
        ax.set_yticks(range(len(class_order)))
        ax.set_xticklabels(class_order, rotation=45, ha="right")
        ax.set_yticklabels(class_order)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        # annotate
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black", fontsize=8)

        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png")
        plt.close(fig)
    except Exception:
        pass

    # metrics text
    try:
        pred_y = detector.model.predict(test_X)
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        acc = float(accuracy_score(test_y, pred_y))
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_y, pred_y, average="macro", zero_division=0
        )

        metrics_path = out_dir / "ml_metrics.txt"
        metrics_path.write_text(
            f"accuracy={acc}\nprecision_macro={precision}\nrecall_macro={recall}\nf1_macro={f1}\n",
            encoding="utf-8",
        )
    except Exception:
        acc = 0.0
        precision = recall = f1 = 0.0

    return {"accuracy": acc, "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def predict_from_raw_features(
    *,
    detector: MLDetector,
    raw_features: Dict[str, Any],
    default_float: float = 0.0,
) -> DetectorPrediction:
    """Best-effort mapping from incoming feature dict to detector feature vector.

    The API/dashboard currently provide a different small feature subset.
    We handle missing features by defaulting to 0.
    """

    x = []
    for col in detector.feature_columns:
        v = raw_features.get(col, default_float)
        try:
            x.append(float(v))
        except Exception:
            x.append(float(default_float))

    X = np.array([x], dtype=np.float32)
    probs = detector.model.predict_proba(X)[0]

    class_idx = int(probs.argmax())
    pred_class = str(detector.model.classes_[class_idx])
    confidence = float(probs[class_idx])

    probs_map = {str(detector.model.classes_[i]): float(probs[i]) for i in range(len(probs))}

    return DetectorPrediction(
        attack_type=pred_class,
        confidence=confidence,
        predicted_class=pred_class,
        probs=probs_map,
    )

