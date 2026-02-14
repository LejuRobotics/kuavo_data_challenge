from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class LowPassFilter:
    cutoff_hz: Optional[float] = None
    tau: Optional[float] = None
    dt: Optional[float] = None
    init: Optional[float] = None

    def __post_init__(self):
        if (self.cutoff_hz is None) == (self.tau is None):
            raise ValueError("Specify exactly one of cutoff_hz or tau.")
        if self.tau is None:
            self.tau = 1.0 / (2.0 * np.pi * float(self.cutoff_hz))
        self._y = None if self.init is None else float(self.init)

    def alpha(self, dt: float) -> float:
        return float(dt) / (self.tau + float(dt))

    def update(self, x, dt: Optional[float] = None) -> float:
        if dt is None:
            if self.dt is None:
                raise ValueError("dt was not provided at init and not passed to update.")
            dt = self.dt
        a = self.alpha(dt)
        if self._y is None:
            self._y = x
        else:
            self._y = self._y + a * (x - self._y)
        return self._y

    def reset(self, value: Optional[float] = None):
        self._y = None if value is None else float(value)
