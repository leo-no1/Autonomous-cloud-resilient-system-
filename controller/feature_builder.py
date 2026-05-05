"""
controller/feature_builder.py
Builds the 7-feature vector from the current node list.
Features: [avg_cpu, max_cpu, avg_lat, max_lat, avg_queue, max_queue, failing_count]
"""

from typing import List

from common.types import Node, NodeStatus, FEATURE_NAMES
from common.logger import get_logger

log = get_logger(__name__)


def build_feature_vector(nodes: List[Node]) -> List[float]:
    healthy = [n for n in nodes if n.status == NodeStatus.HEALTHY]

    if not healthy:
        failing_count = len(nodes)
        log.warning("No healthy nodes — returning zero feature vector.")
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(failing_count)]

    avg_cpu   = sum(n.last_metrics.cpu        for n in healthy) / len(healthy)
    max_cpu   = max(n.last_metrics.cpu        for n in healthy)
    avg_lat   = sum(n.last_metrics.latency_ms for n in healthy) / len(healthy)
    max_lat   = max(n.last_metrics.latency_ms for n in healthy)
    avg_queue = sum(n.last_metrics.queue_len  for n in healthy) / len(healthy)
    max_queue = float(max(n.last_metrics.queue_len for n in healthy))
    failing_count = float(sum(1 for n in nodes if n.status == NodeStatus.FAILING))

    features = [avg_cpu, max_cpu, avg_lat, max_lat, avg_queue, max_queue, failing_count]
    log.debug("Features: %s", dict(zip(FEATURE_NAMES, features)))
    return features
