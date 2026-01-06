from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


def now_utc_iso() -> str:
    # ISO-ish without importing datetime everywhere
    # (keep it simple + stable)
    import datetime as dt

    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_git_commit(repo_root: str | None = None) -> str | None:
    try:
        cwd = repo_root or os.getcwd()
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def platform_dict() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "cpu_count_logical": os.cpu_count(),
        "machine": platform.machine(),
    }


def percentile(xs: Sequence[float], p: float) -> float:
    if not xs:
        return float("nan")
    arr = np.array(xs, dtype=float)
    return float(np.percentile(arr, p))


def mean_std(xs: Sequence[float]) -> tuple[float, float]:
    if not xs:
        return (float("nan"), float("nan"))
    arr = np.array(xs, dtype=float)
    return (float(arr.mean()), float(arr.std(ddof=0)))


def sha1_file(path: str) -> str | None:
    import hashlib

    if not os.path.exists(path):
        return None
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
