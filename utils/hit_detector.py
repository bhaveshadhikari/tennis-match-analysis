"""
hit_detector.py
Detects racket-ball hit events using local minima of the distance signal.
A hit is a local minimum below the distance threshold.
Expands each hit frame by a configurable offset to capture surrounding context.
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


def detect_hits(df_dist: pd.DataFrame, distance_threshold: float, order: int = 3) -> pd.DataFrame:
    dip_indices = argrelextrema(df_dist["distance"].values, np.less_equal, order=order)[0]
    local_minima = df_dist.iloc[dip_indices].copy()

    # Keep one row per distance plateau
    local_minima = local_minima.groupby("distance").head(1)
    hits = local_minima[local_minima["distance"] < distance_threshold].copy()

    print(f"[hit_detector] Detected {len(hits)} hit events")
    return hits


def expand_hit_frames(hit_frames, offset: int = 3) -> set:
    expanded = set()
    for f in hit_frames:
        for delta in range(-offset, offset + 1):
            expanded.add(f + delta)
    return expanded
