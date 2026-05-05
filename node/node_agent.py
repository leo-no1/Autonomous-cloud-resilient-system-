"""
node/node_agent.py
Runs on each EC2 / Docker container.
  - Collects metrics
  - Sends heartbeats via SQS
  - Listens for commands via SQS
  - Executes tasks (unless isolated)

Usage:
    python -m node.node_agent --node-id N1
"""

import argparse
import sys
import time

from common.logger import get_logger
from common.types import (
    HeartbeatMessage, NodeRole, Task, TaskPriority,
)
from common.utils import load_json
from aws.sqs_client import sqs_send
from node.metrics_collector import collect_metrics
from node.task_worker import TaskWorker
from node.command_listener import CommandListener

log = get_logger(__name__)


def run_agent(node_id: str) -> None:
    cfg     = load_json("config/runtime_config.json")
    aws_cfg = load_json("config/aws_config.json")

    hb_queue  = aws_cfg["sqs_heartbeat_queue_url"]
    cmd_queue = aws_cfg["sqs_command_queue_url"]
    interval  = cfg["heartbeat_interval_sec"]

    worker   = TaskWorker()
    listener = CommandListener(node_id, cmd_queue)

    # Mutable node state dict (mutated by command listener)
    state = {
        "role":     NodeRole.FOLLOWER,
        "isolated": False,
        "shutdown": False,
    }

    log.info("NodeAgent %s starting (interval=%ss)", node_id, interval)

    # Seed a few demo tasks
    for i in range(3):
        worker.add_task(Task(
            task_id=f"task-{node_id}-{i}",
            priority=TaskPriority.CRITICAL if i == 0 else TaskPriority.NORMAL,
            cpu_cost=float(i + 1),
        ))

    while not state["shutdown"]:
        # (A) Collect metrics
        metrics = collect_metrics(queue_len=worker.size())

        # (B) Send heartbeat
        hb = HeartbeatMessage(
            node_id=node_id,
            metrics={
                "cpu":        metrics.cpu,
                "memory":     metrics.memory,
                "latency_ms": metrics.latency_ms,
                "queue_len":  metrics.queue_len,
            },
        )
        sqs_send(hb_queue, hb.to_dict())
        log.info(
            "HB sent | cpu=%.1f%% mem=%.1f%% lat=%.1fms q=%d role=%s",
            metrics.cpu, metrics.memory, metrics.latency_ms,
            metrics.queue_len, state["role"].value,
        )

        # (C) Poll + apply commands
        listener.poll_and_apply(state)

        # (D) Execute tasks if not isolated
        if not state["isolated"]:
            worker.execute_one()
        else:
            log.warning("Node %s is ISOLATED — skipping task execution.", node_id)

        time.sleep(interval)

    log.info("NodeAgent %s shutting down.", node_id)
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-id", required=True)
    args = parser.parse_args()
    run_agent(args.node_id)
