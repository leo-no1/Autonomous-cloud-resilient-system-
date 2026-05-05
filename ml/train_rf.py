"""
ml/train_rf.py
Random Forest — main candidate classifier.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from common.logger import get_logger

log = get_logger(__name__)


def train_random_forest(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    n_estimators: int = 100,
) -> RandomForestClassifier:
    log.info("Training Random Forest (n_estimators=%d) ...", n_estimators)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, Y_train)
    log.info("RF training complete.")
    return model
