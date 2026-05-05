"""
rl/scheduler_policy.py
Translates RL actions into concrete scheduling directives
and applies them to the task manager.
"""

from common.logger import get_logger
from controller.task_manager import set_policy

log = get_logger(__name__)

# Action index → (policy, description)
ACTION_MAP = {
    0: ("PRIORITY",    "Prioritise CRITICAL tasks — more CPU to critical work"),
    1: ("LEAST_QUEUE", "Increase time-slice — route to least-loaded node"),
    2: ("PRIORITY",    "Defer non-critical — PRIORITY policy, non-critical deferred"),
}


def apply_rl_action(action: int) -> None:
    policy, desc = ACTION_MAP.get(action, ("ROUND_ROBIN", "Default round-robin"))
    log.info("RL scheduling action %d → %s | %s", action, policy, desc)
    set_policy(policy)
