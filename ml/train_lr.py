"""
ml/train_lr.py
Logistic Regression — baseline classifier.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common.logger import get_logger

log = get_logger(__name__)


def train_logistic_regression(
    X_train: np.ndarray,
    Y_train: np.ndarray,
) -> Pipeline:
    log.info("Training Logistic Regression ...")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=500, random_state=42)),
    ])
    model.fit(X_train, Y_train)
    log.info("LR training complete.")
    return model
