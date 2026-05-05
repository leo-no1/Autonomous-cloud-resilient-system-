"""
controller/dataset_logger.py
Appends feature rows to a JSONL file.
Stops automatically after max_log_rows are written.
"""

import json
import os
import time
from typing import List, Optional

from common.logger import get_logger
from common.types import FEATURE_NAMES, SystemState
from common.utils import load_json

log = get_logger(__name__)


class DatasetLogger:
    def __init__(self):
        cfg = load_json("config/runtime_config.json")
        self.path     : str = cfg["log_file_path"]
        self.max_rows : int = cfg["max_log_rows"]
        self._count   : int = self._count_existing()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        log.info("DatasetLogger ready — %d/%d rows written so far.",
                 self._count, self.max_rows)

    def _count_existing(self) -> int:
        if not os.path.exists(self.path):
            return 0
        with open(self.path) as f:
            return sum(1 for _ in f)

    @property
    def is_full(self) -> bool:
        return self._count >= self.max_rows

    def log_row(self, features: List[float], label: int,
                extra: Optional[dict] = None) -> bool:
        """
        Write one row.  Returns False (and stops) when limit is reached.
        """
        if self.is_full:
            return False

        row = {
            "ts":    time.time(),
            "label": label,
            **dict(zip(FEATURE_NAMES, features)),
        }
        if extra:
            row.update(extra)

        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")

        self._count += 1
        if self._count >= self.max_rows:
            log.info("Dataset full (%d rows). Logging disabled.", self.max_rows)
        return True
