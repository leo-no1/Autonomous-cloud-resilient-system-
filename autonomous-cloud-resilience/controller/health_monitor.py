"""
controller/health_monitor.py
Updates node health based on heartbeat timeout.
Triggers re-election when the leader fails.
"""

import time
from typing import List, Optional

from common.logger import get_logger
from common.types import Node, NodeStatus
from common.utils import load_json

log = get_logger(__name__)


class HealthMonitor:
    def __init__(self, timeout_sec: Optional[float] = None):
        cfg = load_json("config/runtime_config.json")
        self.timeout = timeout_sec or cfg["heartbeat_timeout_sec"]

    def update(self, nodes: List[Node], current_leader_id: Optional[str]) -> bool:
        """
        Update each node's status.
        Returns True if the current leader has just failed (re-election needed).
        """
        leader_failed = False
        for node in nodes:
            age = time.time() - node.last_heartbeat_time
            if age > self.timeout:
                if node.status == NodeStatus.HEALTHY:
                    log.warning(
                        "Node %s timed out (%.1fs > %.1fs) → FAILING",
                        node.id, age, self.timeout,
                    )
                node.status = NodeStatus.FAILING
                if node.id == current_leader_id:
                    leader_failed = True
            else:
                node.status = NodeStatus.HEALTHY
        return leader_failed
