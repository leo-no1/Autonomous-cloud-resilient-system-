"""
controller/controller.py
Main controller process.

Two timers run inside the same loop:
  LOG_TIMER   (5 s)  — collect heartbeats, update state, log dataset row
  CONTROL_TIMER (30 s) — run ML prediction + take actions

Usage:
    python -m controller.controller
"""

import json
import time
from typing import Dict, List, Optional

from common.logger import get_logger
from common.types import (
    HeartbeatMessage, Node, NodeRole, NodeStatus, SystemState,
)
from common.utils import load_json
from common.timekeeper import Timekeeper
from aws.sqs_client import sqs_poll, sqs_delete
from controller.health_monitor import HealthMonitor
from controller.feature_builder import build_feature_vector
from controller.consensus import elect_leader_if_needed
from controller.actions import action_normal, action_warning, action_critical
from controller.dataset_logger import DatasetLogger
from ml.dataset import rule_based_label
from ml.train_lr import train_logistic_regression
from ml.train_rf import train_random_forest
from ml.train_iforest import train_isolation_forest
from ml.evaluate import evaluate_classifier, evaluate_predictions
from ml.select_best import select_best_model
from ml.dataset import load_or_build_dataset
from rl.q_learning_agent import QLearningAgent

log = get_logger(__name__)


def _load_nodes() -> List[Node]:
    cfg   = load_json("config/nodes.json")
    nodes = [Node(id=n["id"], instance_id=n["instance_id"])
             for n in cfg["nodes"]]
    # Seed heartbeat time so they start HEALTHY
    now = time.time()
    for n in nodes:
        n.last_heartbeat_time = now
    return nodes


def _ingest_heartbeats(
    nodes: List[Node],
    hb_queue_url: str,
) -> None:
    messages = sqs_poll(hb_queue_url, max_msgs=20, wait_time=2)
    node_map: Dict[str, Node] = {n.id: n for n in nodes}
    for item in messages:
        hb = HeartbeatMessage.from_dict(item["body"])
        if hb.node_id in node_map:
            n = node_map[hb.node_id]
            n.last_heartbeat_time = hb.timestamp
            m = hb.metrics
            n.last_metrics.cpu        = m.get("cpu", 0.0)
            n.last_metrics.memory     = m.get("memory", 0.0)
            n.last_metrics.latency_ms = m.get("latency_ms", 0.0)
            n.last_metrics.queue_len  = m.get("queue_len", 0)
        sqs_delete(hb_queue_url, item["receipt_handle"])


def _predict_state(model, model_name: str, features: List[float],
                   cfg: dict) -> int:
    import numpy as np
    X = np.array(features).reshape(1, -1)
    if model_name == "IsolationForest":
        score = model.decision_function(X)[0]
        lo = cfg["low_anomaly_threshold"]
        hi = cfg["high_anomaly_threshold"]
        if score >= lo:
            return SystemState.NORMAL
        elif score >= hi:
            return SystemState.WARNING
        else:
            return SystemState.CRITICAL
    else:
        return int(model.predict(X)[0])


# ─────────────────────────────────────────────────────────────────────────────

def run_controller() -> None:
    aws_cfg = load_json("config/aws_config.json")
    rt_cfg  = load_json("config/runtime_config.json")

    hb_queue  = aws_cfg["sqs_heartbeat_queue_url"]
    cmd_queue = aws_cfg["sqs_command_queue_url"]

    # ── Phase 1: Build / load dataset ────────────────────────────────────────
    log.info("=== Phase 1: Loading/building dataset ===")
    X, Y = load_or_build_dataset()

    # ── Phase 2: Train + evaluate 3 models ───────────────────────────────────
    log.info("=== Phase 2: Training models ===")
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42,
    )

    results = []

    lr_model = train_logistic_regression(X_train, Y_train)
    lr_score = evaluate_classifier(lr_model, X_test, Y_test)
    results.append({"name": "LogisticRegression", "model": lr_model, "metrics": lr_score})

    rf_model = train_random_forest(X_train, Y_train)
    rf_score = evaluate_classifier(rf_model, X_test, Y_test)
    results.append({"name": "RandomForest", "model": rf_model, "metrics": rf_score})

    normal_idx  = [i for i, y in enumerate(Y_train) if y == 0]
    X_normal    = X_train[normal_idx]
    iso_model   = train_isolation_forest(X_normal)
    from ml.train_iforest import isoforest_to_state_predictions
    Y_pred_iso  = isoforest_to_state_predictions(iso_model, X_test, rt_cfg)
    iso_score   = evaluate_predictions(Y_test, Y_pred_iso)
    results.append({"name": "IsolationForest", "model": iso_model, "metrics": iso_score})

    best_model, best_name = select_best_model(results)
    log.info("Best model: %s", best_name)

    # ── Phase 3: Init RL agent ────────────────────────────────────────────────
    log.info("=== Phase 3: Init RL agent ===")
    rl_agent = QLearningAgent()

    # ── Phase 4: Live control loop ────────────────────────────────────────────
    log.info("=== Phase 4: Live control loop ===")
    nodes          = _load_nodes()
    monitor        = HealthMonitor()
    ds_logger      = DatasetLogger()
    current_leader: Optional[str] = None

    current_leader = elect_leader_if_needed(nodes, current_leader, cmd_queue)

    log_timer     = Timekeeper(rt_cfg["log_interval_sec"])
    control_timer = Timekeeper(rt_cfg["control_interval_sec"])

    while True:
        # ── LOG tick (every 5 s) ──────────────────────────────────────────────
        if log_timer.is_due():
            _ingest_heartbeats(nodes, hb_queue)
            leader_failed = monitor.update(nodes, current_leader)
            if leader_failed:
                current_leader = elect_leader_if_needed(
                    nodes, current_leader, cmd_queue,
                )

            features = build_feature_vector(nodes)
            label    = rule_based_label(features, rt_cfg)

            if not ds_logger.is_full:
                ds_logger.log_row(features, label)

            log_timer.reset()

        # ── CONTROL tick (every 30 s) ─────────────────────────────────────────
        if control_timer.is_due():
            features     = build_feature_vector(nodes)
            system_state = _predict_state(best_model, best_name, features, rt_cfg)

            log.info(
                "CONTROL | state=%s | features=%s",
                SystemState(system_state).name,
                [round(f, 2) for f in features],
            )

            if system_state == SystemState.NORMAL:
                action_normal(nodes)
            elif system_state == SystemState.WARNING:
                action_warning(nodes, rl_agent, cmd_queue)
            else:
                current_leader = action_critical(
                    nodes, rl_agent, cmd_queue, current_leader,
                )

            control_timer.reset()

        time.sleep(0.5)


if __name__ == "__main__":
    run_controller()
