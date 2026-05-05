"""
ml/select_best.py
Selects the best model by F1-weighted score, saves it to disk.
"""

import json
import os
from typing import Any, List, Tuple

import joblib

from common.logger import get_logger
from common.utils import save_json

log = get_logger(__name__)

MODEL_PATH  = "models/best_model.pkl"
REPORT_PATH = "models/model_report.json"


def select_best_model(results: List[dict]) -> Tuple[Any, str]:
    """
    results: list of {"name": str, "model": obj, "metrics": dict}
    Returns (best_model_object, best_model_name).
    """
    best = max(results, key=lambda r: r["metrics"]["f1_weighted"])

    os.makedirs("models", exist_ok=True)
    joblib.dump(best["model"], MODEL_PATH)
    log.info("Best model '%s' saved to %s", best["name"], MODEL_PATH)

    report = {
        r["name"]: {k: v for k, v in r["metrics"].items() if k != "report"}
        for r in results
    }
    report["selected"] = best["name"]
    save_json(REPORT_PATH, report)
    log.info("Model report saved to %s", REPORT_PATH)

    return best["model"], best["name"]


def load_best_model() -> Any:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No saved model at {MODEL_PATH}. Run training first.")
    model = joblib.load(MODEL_PATH)
    log.info("Loaded best model from %s", MODEL_PATH)
    return model
