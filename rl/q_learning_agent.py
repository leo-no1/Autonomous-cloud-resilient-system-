"""
rl/q_learning_agent.py
Q-Learning agent for CPU scheduling decisions.

State space  : (system_state_bucket, cpu_bucket, queue_bucket)
Action space :
    0 — Give more CPU to CRITICAL tasks
    1 — Increase time-slice for long tasks
    2 — Pause / defer non-critical tasks
"""

import json
import os
import random
from typing import List, Optional, Tuple

import numpy as np

from common.logger import get_logger
from common.types import Node, NodeStatus

log = get_logger(__name__)

ACTION_NAMES = {
    0: "MORE_CPU_TO_CRITICAL",
    1: "INCREASE_TIME_SLICE",
    2: "DEFER_NON_CRITICAL",
}

N_STATES  = 3 * 5 * 5   # 3 sys states × 5 cpu bins × 5 queue bins
N_ACTIONS = 3


class QLearningAgent:
    def __init__(
        self,
        alpha: float  = 0.1,   # learning rate
        gamma: float  = 0.9,   # discount factor
        epsilon: float = 0.2,  # exploration rate
        q_table_path: str = "models/q_table.npy",
    ):
        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon
        self.q_path  = q_table_path

        if os.path.exists(q_table_path):
            self.Q = np.load(q_table_path)
            log.info("Q-table loaded from %s", q_table_path)
        else:
            self.Q = np.zeros((N_STATES, N_ACTIONS))

        self._prev_state : Optional[int] = None
        self._prev_action: Optional[int] = None
        self._prev_metrics: Optional[dict] = None

    # ── State encoding ────────────────────────────────────────────────────────

    @staticmethod
    def _encode_state(state_label: str, nodes: List[Node]) -> int:
        sys_idx = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}.get(state_label, 0)

        healthy = [n for n in nodes if n.status == NodeStatus.HEALTHY]
        if healthy:
            avg_cpu   = sum(n.last_metrics.cpu      for n in healthy) / len(healthy)
            avg_queue = sum(n.last_metrics.queue_len for n in healthy) / len(healthy)
        else:
            avg_cpu, avg_queue = 100.0, 10.0

        cpu_bin   = min(int(avg_cpu   / 20), 4)   # 0-4
        queue_bin = min(int(avg_queue / 3),  4)   # 0-4

        return sys_idx * 25 + cpu_bin * 5 + queue_bin

    # ── Reward ────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_reward(prev: Optional[dict], curr: dict) -> float:
        if prev is None:
            return 0.0
        delta_lat   = prev["lat"]   - curr["lat"]    # positive = improved
        delta_queue = prev["queue"] - curr["queue"]  # positive = improved
        return float(delta_lat * 0.5 + delta_queue * 0.5)

    @staticmethod
    def _node_summary(nodes: List[Node]) -> dict:
        healthy = [n for n in nodes if n.status == NodeStatus.HEALTHY]
        if not healthy:
            return {"lat": 999.0, "queue": 99}
        return {
            "lat":   sum(n.last_metrics.latency_ms for n in healthy) / len(healthy),
            "queue": sum(n.last_metrics.queue_len  for n in healthy) / len(healthy),
        }

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, state: int) -> int:
        if random.random() < self.epsilon:
            return random.randrange(N_ACTIONS)
        return int(np.argmax(self.Q[state]))

    # ── Q update ──────────────────────────────────────────────────────────────

    def update(self, s: int, a: int, reward: float, s_next: int) -> None:
        best_next = np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha * (
            reward + self.gamma * best_next - self.Q[s, a]
        )

    # ── Public step (called by actions.py) ────────────────────────────────────

    def step(self, state_label: str, nodes: List[Node]) -> int:
        curr_state   = self._encode_state(state_label, nodes)
        curr_metrics = self._node_summary(nodes)
        reward       = self._compute_reward(self._prev_metrics, curr_metrics)

        # Update Q for previous transition
        if self._prev_state is not None and self._prev_action is not None:
            self.update(self._prev_state, self._prev_action, reward, curr_state)

        action = self.select_action(curr_state)
        log.info(
            "RL | state=%s(%d) action=%s reward=%.2f",
            state_label, curr_state,
            ACTION_NAMES[action], reward,
        )

        self._prev_state   = curr_state
        self._prev_action  = action
        self._prev_metrics = curr_metrics
        return action

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.q_path), exist_ok=True)
        np.save(self.q_path, self.Q)
        log.info("Q-table saved to %s", self.q_path)
