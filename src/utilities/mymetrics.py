
import os
import sys
import json
import atexit
import threading
import numpy as np
from enum import Enum

#
# A very naive but effective metrics and observability lib
#

_metrics = {}
_lock = threading.Lock()

class MetricType(Enum):
    SUM = "sum"
    AVG = "avg"
    LATEST = "latest"
    SERIES = "series"
    STATS = "stats"

class Metric:
    def __init__(self, name, type: MetricType):
        self.series = []
        self.type = type

def track_count(name):
    with _lock:
        metric = _metrics.get(name, Metric(name, MetricType.SUM))
        if len(metric.series) == 0:
            metric.series.append(1)
        else:
            metric.series[0] += 1
        _metrics[name] = metric

def track_sum(name, value):
    with _lock:
        metric = _metrics.get(name, Metric(name, MetricType.SUM))
        if len(metric.series) == 0:
            metric.series.append(value)
        else:
            metric.series[0] += 1
        _metrics[name] = metric

def track_avg(name, value):
    with _lock:
        metric = _metrics.get(name, Metric(name, MetricType.AVG))
        metric.series.append(value)
        _metrics[name] = metric

def track_latest(name, value):
    with _lock:
        metric = _metrics.get(name, Metric(name, MetricType.LATEST))
        metric.series = [value]
        _metrics[name] = metric

def track_series(name, value):
    with _lock:
        metric = _metrics.get(name, Metric(name, MetricType.SERIES))
        metric.series.append(value)
        _metrics[name] = metric

def track_stats(name, value):
    with _lock:
        metric = _metrics.get(name, Metric(name, MetricType.STATS))

        if not hasattr(metric, 'count'):
            metric.count = 0
            metric.sum = 0
            metric.min = float('inf')
            metric.max = float('-inf')

        metric.count += 1
        metric.sum += value
        metric.min = min(metric.min, value)
        metric.max = max(metric.max, value)
        metric.avg = metric.sum / metric.count

        _metrics[name] = metric

def on_exit():
    report = {}

    with _lock:
        snapshot = list(_metrics.items())

    for name, metric in snapshot:
        if metric.type == MetricType.SUM:
            report[name] = {
                "type": "sum",
                "value": sum(metric.series)
            }
        elif metric.type == MetricType.AVG:
            report[name] = {
                "type": "avg",
                "value": sum(metric.series) / len(metric.series)
            }
        elif metric.type == MetricType.LATEST:
            report[name] = {
                "type": "latest",
                "value": metric.series[-1]
            }
        elif metric.type == MetricType.SERIES:
            report[name] = {
                "type": "series",
                "min": min(metric.series),
                "max": max(metric.series),
                "avg": sum(metric.series) / len(metric.series),
                "med": np.median(metric.series),
                "sum": sum(metric.series),
                "values": metric.series
            }
        elif metric.type == MetricType.STATS:
            report[name] = {
                "type": "stats",
                "count": metric.count,
                "sum": metric.sum,
                "min": metric.min,
                "max": metric.max,
                "avg": metric.avg
            }

    mainfile = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    with open(f'{mainfile}.metrics.json', 'w') as f:
        json.dump(report, f, indent=4)

atexit.register(on_exit)