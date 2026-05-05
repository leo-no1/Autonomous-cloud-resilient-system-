"""
controller/task_manager.py
Task assignment and migration between nodes.
"""

import itertools
from typing import List, Optional

from common.logger import get_logger
from common.types import Node, NodeStatus, Task, TaskPriority

log = get_logger(__name__)

# Simple in-memory task store keyed by node_id
_node_tasks: dict[str, List[Task]] = {}
_rr_cycle = None  # round-robin iterator


def assign_task(task: Task, nodes: List[Node], policy: str = "ROUND_ROBIN") -> Optional[str]:
    """
    Assign a task to a node using the given policy.
    Returns the node_id it was assigned to, or None.
    """
    global _rr_cycle
    healthy = [n for n in nodes if n.status == NodeStatus.HEALTHY]
    if not healthy:
        log.error("No healthy nodes to assign task %s", task.task_id)
        return None

    if policy == "ROUND_ROBIN":
        if _rr_cycle is None:
            _rr_cycle = itertools.cycle(healthy)
        target = next(_rr_cycle)

    elif policy == "LEAST_QUEUE":
        target = min(healthy, key=lambda n: n.last_metrics.queue_len)

    elif policy == "PRIORITY":
        # CRITICAL tasks go to least-loaded; NORMAL tasks round-robin
        if task.priority == TaskPriority.CRITICAL:
            target = min(healthy, key=lambda n: n.last_metrics.cpu)
        else:
            if _rr_cycle is None:
                _rr_cycle = itertools.cycle(healthy)
            target = next(_rr_cycle)
    else:
        target = healthy[0]

    _node_tasks.setdefault(target.id, []).append(task)
    log.info("Task %s (pri=%s) assigned to %s via %s",
             task.task_id, task.priority.value, target.id, policy)
    return target.id


def migrate_tasks(failing_nodes: List[Node], healthy_nodes: List[Node]) -> None:
    """Reassign pending tasks from failing nodes to healthy ones."""
    if not healthy_nodes:
        log.error("No healthy nodes to migrate tasks to.")
        return

    for fn in failing_nodes:
        pending = [t for t in _node_tasks.get(fn.id, []) if t.status == "PENDING"]
        for task in pending:
            target = min(healthy_nodes, key=lambda n: n.last_metrics.queue_len)
            _node_tasks.setdefault(target.id, []).append(task)
            log.info("Migrated task %s from %s → %s", task.task_id, fn.id, target.id)
        _node_tasks[fn.id] = [t for t in _node_tasks.get(fn.id, [])
                               if t.status != "PENDING"]


def set_policy(policy: str) -> None:
    global _rr_cycle
    _rr_cycle = None  # reset iterator on policy change
    log.info("Task assignment policy set to: %s", policy)
