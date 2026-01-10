# src/utils.py
import os
import time
import random
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import psutil

# headless figure saving
import matplotlib
matplotlib.use("Agg")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_system_ram_gb() -> Tuple[float, float, float, float]:
    mem = psutil.virtual_memory()
    total = mem.total / (1024**3)
    used = mem.used / (1024**3)
    free = mem.available / (1024**3)
    percent = mem.percent
    return total, used, free, percent


@dataclass
class ResourceSnapshot:
    t: float
    proc_rss_mb: float
    sys_used_gb: float


def take_snapshot() -> ResourceSnapshot:
    p = psutil.Process(os.getpid())
    rss_mb = p.memory_info().rss / (1024 * 1024)
    _, sys_used, _, _ = get_system_ram_gb()
    return ResourceSnapshot(t=time.time(), proc_rss_mb=rss_mb, sys_used_gb=sys_used)


@dataclass
class ResourceDelta:
    wall_time_s: float
    proc_rss_delta_mb: float
    sys_used_delta_gb: float


def diff_snapshot(start: ResourceSnapshot, end: ResourceSnapshot) -> ResourceDelta:
    return ResourceDelta(
        wall_time_s=end.t - start.t,
        proc_rss_delta_mb=end.proc_rss_mb - start.proc_rss_mb,
        sys_used_delta_gb=end.sys_used_gb - start.sys_used_gb,
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def model_file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)
