import time
from .mylogging import create_logger
from .mymetrics import track_stats

#
# Using this to time ops for perf comparisons
#

logger = create_logger("mytimer")

class Timer:
    def __init__(self, operation_name=None, writelog=False):
        self.start_time = None
        self.end_time = None
        self.operation_name = operation_name or 'unnamed'
        self.writelog = writelog

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        # Log the duration
        if self.writelog:
            if duration < 60:
                logger.debug(f"{self.operation_name or 'Operation'} took {duration:.4f} seconds")
            else:
                minutes = int(duration // 60)
                seconds = duration % 60
                logger.debug(f"{self.operation_name or 'Operation'} took {minutes}m:{seconds:.0f}s")

        # Register as metric
        metrics_name = f"timer_{self.operation_name}"
        track_stats(metrics_name, duration)

    def get_duration(self):
        return self.end_time or time.time() - self.start_time