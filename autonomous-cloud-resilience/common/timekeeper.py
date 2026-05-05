"""
common/timekeeper.py
Simple helpers for managing recurring timer intervals.
"""

import time


class Timekeeper:
    """Fires True when the interval has elapsed since the last reset."""

    def __init__(self, interval_sec: float):
        self.interval = interval_sec
        self._last    = time.time()

    def is_due(self) -> bool:
        return (time.time() - self._last) >= self.interval

    def reset(self) -> None:
        self._last = time.time()

    def wait_until_due(self) -> None:
        remaining = self.interval - (time.time() - self._last)
        if remaining > 0:
            time.sleep(remaining)
        self.reset()
