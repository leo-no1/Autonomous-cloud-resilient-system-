"""
controller/actions.py
Actions executed for each SYSTEM_STATE (NORMAL / WARNING / CRITICAL).
"""

from typing import List

from common.logger import get_logger
from common.types import Node, NodeStatus, CommandType, SystemState
from common.utils import load_json
from aws.sqs_client import sqs_send
from controller.task_manager import set_policy, migrate_tasks
from controller.consensus import elect_leader_if_needed

log = get_logger(__name__)


def _send_command(node_id: str, command: CommandType, cmd_queue_url: str) -> None:
    sqs_send(cmd_queue_url, {
        "msg_type":       "COMMAND",
        "target_node_id": node_id,
        "command":        command.value,
    })
    log.info("Command %s → node %s", command.value, node_id)


def action_normal(nodes: List[Node]) -> None:
    log.info("STATE=NORMAL — balanced round-robin scheduling.")
    set_policy("ROUND_ROBIN")


def action_warning(nodes: List[Node], scheduler_agent, cmd_queue_url: str) -> None:
    log.warning("STATE=WARNING — switching to priority + least-queue scheduling.")
    set_policy("PRIORITY")
    _throttle_non_critical(nodes, cmd_queue_url)
    if scheduler_agent:
        scheduler_agent.step(state_label="WARNING", nodes=nodes)


def action_critical(
    nodes: List[Node],
    scheduler_agent,
    cmd_queue_url: str,
    current_leader_id,
) -> str:
    log.error("STATE=CRITICAL — isolating failing nodes + failover.")

    failing = [n for n in nodes if n.status == NodeStatus.FAILING]
    healthy = [n for n in nodes if n.status == NodeStatus.HEALTHY]

    # 1. Isolate failing nodes
    for fn in failing:
        _send_command(fn.id, CommandType.ISOLATE, cmd_queue_url)

    # 2. Re-elect leader if needed
    new_leader = elect_leader_if_needed(nodes, current_leader_id, cmd_queue_url)

    # 3. Migrate tasks from failing → healthy
    migrate_tasks(failing, healthy)

    # 4. RL-guided scheduling under stress
    if scheduler_agent:
        scheduler_agent.step(state_label="CRITICAL", nodes=nodes)

    log.info("CRITICAL actions complete. Leader=%s", new_leader)
    return new_leader


def _throttle_non_critical(nodes: List[Node], cmd_queue_url: str) -> None:
    """Placeholder: in a real system this would send THROTTLE commands."""
    log.info("Throttling non-critical network tasks (conceptual).")
