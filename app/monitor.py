from __future__ import annotations

import time
from collections import deque

import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    _GPU = pynvml.nvmlDeviceGetHandleByIndex(0)
    _GPU_NAME = pynvml.nvmlDeviceGetName(_GPU)
    if isinstance(_GPU_NAME, bytes):
        _GPU_NAME = _GPU_NAME.decode()
except Exception:
    _GPU = None
    _GPU_NAME = None


class Monitor:
    """Rolling FPS (with detector/tracker split) plus system resource sampling."""

    def __init__(self, window=30):
        self._frame_times = deque(maxlen=window)
        self._det_ms = deque(maxlen=window)
        self._trk_ms = deque(maxlen=window)
        self._last = None
        self.process = psutil.Process()
        self.process.cpu_percent(None)  # prime

    def tick(self, det_ms: float, trk_ms: float):
        now = time.perf_counter()
        if self._last is not None:
            self._frame_times.append(now - self._last)
        self._last = now
        self._det_ms.append(det_ms)
        self._trk_ms.append(trk_ms)

    @staticmethod
    def _avg(d):
        return sum(d) / len(d) if d else 0.0

    def stats(self) -> dict:
        fps = 1.0 / self._avg(self._frame_times) if self._frame_times else 0.0
        s = {
            "fps": round(fps, 1),
            "detect_ms": round(self._avg(self._det_ms), 1),
            "track_ms": round(self._avg(self._trk_ms), 1),
            "cpu_percent": round(psutil.cpu_percent(), 1),
            "ram_mb": round(self.process.memory_info().rss / 1e6, 1),
            "gpu": None,
        }
        if _GPU is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(_GPU)
                mem = pynvml.nvmlDeviceGetMemoryInfo(_GPU)
                s["gpu"] = {
                    "name": _GPU_NAME,
                    "util_percent": util.gpu,
                    "mem_used_mb": round(mem.used / 1e6, 1),
                    "mem_total_mb": round(mem.total / 1e6, 1),
                }
            except Exception:
                pass
        return s
