"""
ml/evaluate.py
Evaluation helpers for all three models.
"""

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from common.logger import get_logger

log = get_logger(__name__)


def evaluate_classifier(model: Any, X_test: np.ndarray, Y_test: np.ndarray) -> Dict:
    Y_pred = model.predict(X_test)
    return _score_dict(Y_test, Y_pred, model.__class__.__name__)


def evaluate_predictions(Y_test: np.ndarray, Y_pred: List[int]) -> Dict:
    return _score_dict(Y_test, np.array(Y_pred), "IsolationForest")


def _score_dict(Y_true: np.ndarray, Y_pred: np.ndarray, name: str) -> Dict:
    acc = accuracy_score(Y_true, Y_pred)
    f1  = f1_score(Y_true, Y_pred, average="weighted", zero_division=0)
    pre = precision_score(Y_true, Y_pred, average="weighted", zero_division=0)
    rec = recall_score(Y_true, Y_pred, average="weighted", zero_division=0)
    report = classification_report(Y_true, Y_pred, zero_division=0)

    log.info(
        "%s | acc=%.3f  f1=%.3f  precision=%.3f  recall=%.3f",
        name, acc, f1, pre, rec,
    )
    log.debug("\n%s", report)

    return {
        "accuracy":    acc,
        "f1_weighted": f1,
        "precision":   pre,
        "recall":      rec,
        "report":      report,
    }
