"""
FPS (Frames Per Second) and latency measurement utilities.

Provides accurate FPS calculation and per-frame latency tracking for real-time
performance monitoring and optimization.
"""

import time
from collections import deque


class FPSCounter:
    """
    FPS counter for measuring real-time performance.

    Uses a rolling window of recent frame times for smooth, accurate FPS estimation.
    """

    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter.

        Args:
            window_size: Number of frames to use for rolling average.
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time.perf_counter()
        self.fps = 0.0

    def update(self) -> float:
        """
        Update FPS counter. Call this once per frame.

        Returns:
            Current FPS estimate.
        """
        current_time = time.perf_counter()
        delta_time = current_time - self.last_time
        self.last_time = current_time

        if delta_time > 0:
            self.frame_times.append(delta_time)

        # Calculate average FPS from rolling window
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            if avg_frame_time > 0:
                self.fps = 1.0 / avg_frame_time
        else:
            self.fps = 0.0

        return self.fps

    def get_fps(self) -> float:
        """Get current FPS value without updating."""
        return self.fps

    def reset(self):
        """Reset FPS counter."""
        self.frame_times.clear()
        self.fps = 0.0
        self.last_time = time.perf_counter()


class LatencyCounter:
    """
    Measure per-frame operation latency.

    Tracks how long each frame takes to process (inference, postprocessing, etc.)
    """

    def __init__(self, window_size: int = 30):
        """
        Initialize latency counter.

        Args:
            window_size: Number of samples for rolling average.
        """
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.start_time = None
        self.avg_latency_ms = 0.0

    def start(self):
        """Mark the start of an operation."""
        self.start_time = time.perf_counter()

    def end(self) -> float:
        """
        Mark the end of an operation and record latency.

        Returns:
            Latency in milliseconds.
        """
        if self.start_time is None:
            return 0.0

        elapsed = time.perf_counter() - self.start_time
        latency_ms = elapsed * 1000.0

        if latency_ms > 0:
            self.latencies.append(latency_ms)

        # Calculate rolling average
        if len(self.latencies) > 0:
            self.avg_latency_ms = sum(self.latencies) / len(self.latencies)

        return latency_ms

    def get_avg_latency_ms(self) -> float:
        """Get average latency in milliseconds."""
        return self.avg_latency_ms

    def reset(self):
        """Reset latency counter."""
        self.latencies.clear()
        self.avg_latency_ms = 0.0
        self.start_time = None
