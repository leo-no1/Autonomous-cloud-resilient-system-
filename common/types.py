"""
common/types.py
Shared data structures, enums, and type definitions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
import time


# ── Enums ────────────────────────────────────────────────────────────────────

class NodeStatus(str, Enum):
    HEALTHY = "HEALTHY"
    FAILING = "FAILING"


class NodeRole(str, Enum):
    LEADER   = "LEADER"
    FOLLOWER = "FOLLOWER"


class SystemState(int, Enum):
    NORMAL   = 0
    WARNING  = 1
    CRITICAL = 2


class CommandType(str, Enum):
    ISOLATE            = "ISOLATE"
    SET_ROLE_LEADER    = "SET_ROLE_LEADER"
    SET_ROLE_FOLLOWER  = "SET_ROLE_FOLLOWER"
    SHUTDOWN           = "SHUTDOWN"


class TaskPriority(str, Enum):
    CRITICAL = "CRITICAL"
    NORMAL   = "NORMAL"


# ── Metrics ───────────────────────────────────────────────────────────────────

@dataclass
class NodeMetrics:
    cpu: float        = 0.0   # 0-100 %
    memory: float     = 0.0   # 0-100 %
    latency_ms: float = 0.0   # milliseconds
    queue_len: int    = 0


# ── Node ──────────────────────────────────────────────────────────────────────

@dataclass
class Node:
    id: str
    instance_id: str
    status: NodeStatus          = NodeStatus.HEALTHY
    role: NodeRole              = NodeRole.FOLLOWER
    last_heartbeat_time: float  = field(default_factory=time.time)
    last_metrics: NodeMetrics   = field(default_factory=NodeMetrics)


# ── Messages ──────────────────────────────────────────────────────────────────

@dataclass
class HeartbeatMessage:
    msg_type: str       = "HEARTBEAT"
    node_id: str        = ""
    timestamp: float    = field(default_factory=time.time)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "msg_type":  self.msg_type,
            "node_id":   self.node_id,
            "timestamp": self.timestamp,
            "metrics":   self.metrics,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "HeartbeatMessage":
        return HeartbeatMessage(
            msg_type=d.get("msg_type", "HEARTBEAT"),
            node_id=d.get("node_id", ""),
            timestamp=d.get("timestamp", time.time()),
            metrics=d.get("metrics", {}),
        )


@dataclass
class CommandMessage:
    msg_type: str       = "COMMAND"
    target_node_id: str = ""
    command: str        = ""
    timestamp: float    = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "msg_type":       self.msg_type,
            "target_node_id": self.target_node_id,
            "command":        self.command,
            "timestamp":      self.timestamp,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CommandMessage":
        return CommandMessage(
            msg_type=d.get("msg_type", "COMMAND"),
            target_node_id=d.get("target_node_id", ""),
            command=d.get("command", ""),
            timestamp=d.get("timestamp", time.time()),
        )


# ── Task ──────────────────────────────────────────────────────────────────────

@dataclass
class Task:
    task_id: str
    priority: TaskPriority = TaskPriority.NORMAL
    cpu_cost: float        = 1.0
    net_cost: float        = 1.0
    status: str            = "PENDING"

    def __lt__(self, other: "Task") -> bool:
        # CRITICAL < NORMAL so CRITICAL is served first in a min-heap
        order = {TaskPriority.CRITICAL: 0, TaskPriority.NORMAL: 1}
        return order[self.priority] < order[other.priority]


# ── Feature vector (7 features) ───────────────────────────────────────────────

FEATURE_NAMES = [
    "avg_cpu", "max_cpu",
    "avg_latency_ms", "max_latency_ms",
    "avg_queue_len", "max_queue_len",
    "failing_count",
]
