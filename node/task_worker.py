"""
node/task_worker.py
Priority queue + task execution logic for a Node Agent.
"""

import heapq
import time
from typing import List

from common.logger import get_logger
from common.types import Task, TaskPriority

log = get_logger(__name__)


class TaskWorker:
    def __init__(self):
        self._heap: List[Task] = []  # min-heap (CRITICAL < NORMAL)

    def add_task(self, task: Task) -> None:
        heapq.heappush(self._heap, task)
        log.debug("Task added: %s (priority=%s)", task.task_id, task.priority)

    def size(self) -> int:
        return len(self._heap)

    def execute_one(self) -> None:
        if not self._heap:
            return
        task = heapq.heappop(self._heap)
        task.status = "RUNNING"
        log.info("Executing task %s (cpu_cost=%.1f)", task.task_id, task.cpu_cost)
        # Simulate work proportional to cpu_cost
        time.sleep(min(task.cpu_cost * 0.01, 0.1))
        task.status = "DONE"
        log.info("Task %s done.", task.task_id)

    def execute_all(self) -> None:
        while self._heap:
            self.execute_one()

    def get_pending_tasks(self) -> List[Task]:
        return [t for t in self._heap if t.status == "PENDING"]

    def clear(self) -> None:
        self._heap.clear()
