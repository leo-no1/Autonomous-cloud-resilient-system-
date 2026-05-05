"""
ml/train_iforest.py
Isolation Forest — unsupervised anomaly detection.
Trained only on NORMAL (label=0) samples.
"""

from typing import List

import numpy as np
from sklearn.ensemble import IsolationForest

from common.logger import get_logger

log = get_logger(__name__)


def train_isolation_forest(
    X_normal: np.ndarray,
    contamination: float = 0.05,
) -> IsolationForest:
    log.info("Training Isolation Forest on %d normal samples ...", len(X_normal))
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
    )
    model.fit(X_normal)
    log.info("IsolationForest training complete.")
    return model


def isoforest_to_state_predictions(
    model: IsolationForest,
    X: np.ndarray,
    cfg: dict,
) -> List[int]:
    """
    Map anomaly scores → 0/1/2 using thresholds from runtime_config.
    decision_function: higher = more normal, lower = more anomalous.
    """
    lo = cfg.get("low_anomaly_threshold",  -0.1)
    hi = cfg.get("high_anomaly_threshold",  0.1)
    scores = model.decision_function(X)
    preds  = []
    for s in scores:
        if s >= lo:
            preds.append(0)
        elif s >= hi:
            preds.append(1)
        else:
            preds.append(2)
    return preds
