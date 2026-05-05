"""
node/metrics_collector.py
Collects local CPU, memory, latency, queue length.
Uses psutil when available; falls back to random simulation for testing.
"""

import random
import time

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from common.types import NodeMetrics


def get_local_cpu() -> float:
    if _HAS_PSUTIL:
        return psutil.cpu_percent(interval=0.1)
    # Simulation: random walk between 10-95
    return round(random.uniform(10.0, 95.0), 2)


def get_local_memory() -> float:
    if _HAS_PSUTIL:
        return psutil.virtual_memory().percent
    return round(random.uniform(20.0, 85.0), 2)


def measure_latency(target: str = "8.8.8.8") -> float:
    """
    Measures round-trip latency in ms.
    In simulation mode returns a random value.
    """
    if _HAS_PSUTIL:
        import subprocess
        try:
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "1", target],
                capture_output=True, text=True, timeout=2,
            )
            for line in result.stdout.splitlines():
                if "time=" in line:
                    return float(line.split("time=")[1].split(" ")[0])
        except Exception:
            pass
    return round(random.uniform(5.0, 350.0), 2)


def collect_metrics(queue_len: int = 0) -> NodeMetrics:
    return NodeMetrics(
        cpu=get_local_cpu(),
        memory=get_local_memory(),
        latency_ms=measure_latency(),
        queue_len=queue_len,
    )
