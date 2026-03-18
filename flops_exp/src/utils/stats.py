from typing import Iterable

import numpy as np


def summarize_numeric(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot summarize an empty value list")
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "std": float(arr.std(ddof=0)),
    }

