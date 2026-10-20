"""A virtual clock: simulated time only advances when told to."""

from __future__ import annotations

__all__ = ["VirtualClock"]


class VirtualClock:
    def __init__(self, dt: float = 1.0, start: float = 0.0) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.dt = dt
        self.now = start
        self._ticks = 0
        self._start = start

    def advance(self) -> float:
        """Move one tick forward; computed from the tick count so float error never accumulates."""
        self._ticks += 1
        self.now = self._start + self._ticks * self.dt
        return self.now
