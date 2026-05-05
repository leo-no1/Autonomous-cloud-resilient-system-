"""
ml/dataset.py
Loads or builds the labeled dataset.

Priority:
  1. data/metrics_logged.jsonl  (controller-generated, rich)
  2. data/metrics_labeled.csv   (CSV fallback)
  3. Synthetic generation        (for first-run / offline demo)
"""

import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from common.logger import get_logger
from common.types import FEATURE_NAMES
from common.utils import load_json

log = get_logger(__name__)

JSONL_PATH = "data/metrics_logged.jsonl"
CSV_PATH   = "data/metrics_labeled.csv"


# ── Rule-based labeler (used by controller for online logging) ────────────────

def rule_based_label(features: List[float], cfg: dict) -> int:
    """
    features = [avg_cpu, max_cpu, avg_lat, max_lat, avg_queue, max_queue, failing_count]
    Returns 0/1/2.
    """
    avg_cpu, max_cpu, avg_lat, max_lat, avg_queue, max_queue, failing_count = features

    if failing_count >= 1 or max_lat > cfg.get("high_lat_threshold", 300.0):
        return 2  # CRITICAL
    if avg_cpu > cfg.get("mid_cpu_threshold", 70.0) or \
       avg_queue > cfg.get("mid_queue_threshold", 5):
        return 1  # WARNING
    return 0  # NORMAL


# ── JSONL loader ──────────────────────────────────────────────────────────────

def _load_jsonl() -> Tuple[np.ndarray, np.ndarray]:
    rows, labels = [], []
    with open(JSONL_PATH) as f:
        for line in f:
            r = json.loads(line)
            rows.append([r[k] for k in FEATURE_NAMES])
            labels.append(int(r["label"]))
    log.info("Loaded %d rows from %s", len(rows), JSONL_PATH)
    return np.array(rows, dtype=float), np.array(labels, dtype=int)


# ── CSV loader ────────────────────────────────────────────────────────────────

def _load_csv() -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(CSV_PATH)
    X  = df[FEATURE_NAMES].values.astype(float)
    Y  = df["label"].values.astype(int)
    log.info("Loaded %d rows from %s", len(Y), CSV_PATH)
    return X, Y


# ── Synthetic generator ───────────────────────────────────────────────────────

def _generate_synthetic(n: int = 600) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic labeled data for offline testing."""
    log.info("Generating %d synthetic samples.", n)
    rng = np.random.default_rng(42)
    rows, labels = [], []

    for _ in range(n):
        label = rng.choice([0, 1, 2], p=[0.6, 0.25, 0.15])
        if label == 0:
            avg_cpu   = rng.uniform(5,  60)
            max_cpu   = avg_cpu + rng.uniform(0, 20)
            avg_lat   = rng.uniform(5,  100)
            max_lat   = avg_lat + rng.uniform(0, 50)
            avg_q     = rng.uniform(0, 3)
            max_q     = avg_q + rng.integers(0, 3)
            failing   = 0
        elif label == 1:
            avg_cpu   = rng.uniform(60, 80)
            max_cpu   = avg_cpu + rng.uniform(5, 15)
            avg_lat   = rng.uniform(80, 200)
            max_lat   = avg_lat + rng.uniform(20, 80)
            avg_q     = rng.uniform(3, 7)
            max_q     = avg_q + rng.integers(1, 4)
            failing   = 0
        else:
            avg_cpu   = rng.uniform(75, 100)
            max_cpu   = min(avg_cpu + rng.uniform(5, 20), 100)
            avg_lat   = rng.uniform(200, 500)
            max_lat   = avg_lat + rng.uniform(50, 200)
            avg_q     = rng.uniform(6, 15)
            max_q     = avg_q + rng.integers(2, 6)
            failing   = rng.integers(1, 4)

        rows.append([avg_cpu, max_cpu, avg_lat, max_lat, avg_q, float(max_q), float(failing)])
        labels.append(label)

    X = np.array(rows, dtype=float)
    Y = np.array(labels, dtype=int)

    # Save as CSV for future runs
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"] = Y
    df.to_csv(CSV_PATH, index=False)
    log.info("Synthetic dataset saved to %s", CSV_PATH)
    return X, Y


# ── Public entry point ────────────────────────────────────────────────────────

def load_or_build_dataset() -> Tuple[np.ndarray, np.ndarray]:
    if os.path.exists(JSONL_PATH):
        return _load_jsonl()
    if os.path.exists(CSV_PATH):
        return _load_csv()
    return _generate_synthetic()
