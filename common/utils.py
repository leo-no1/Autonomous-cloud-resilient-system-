"""
common/utils.py
Shared utility helpers.
"""

import json
import os
import time
from typing import Any, Dict


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def now_ts() -> float:
    return time.time()


def majority(n: int) -> int:
    """Return simple majority threshold."""
    return n // 2 + 1
