"""
distance.py
Computes windowed racket-ball distance over all frames.
Returns a DataFrame of (frame, racket_id, ball_id, distance) for frames
where the racket and ball are within the configured threshold.
"""

import numpy as np
import pandas as pd


def compute_racket_ball_distance(df: pd.DataFrame, window: int, threshold: float) -> pd.DataFrame:
    results = []
    frames = sorted(df["frame"].unique())

    for frame in frames:
        window_df = df[
            (df["frame"] >= frame - window) &
            (df["frame"] <= frame + window)
        ]

        balls = window_df[window_df["label"] == "ball"]
        rackets = window_df[window_df["label"] == "racket"]

        if balls.empty or rackets.empty:
            continue

        min_dist = float(threshold)
        closest_racket_id = None
        closest_ball_id = None

        for _, b in balls.iterrows():
            for _, r in rackets.iterrows():
                dist = np.sqrt((r["cx"] - b["cx"]) ** 2 + (r["cy"] - b["cy"]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    closest_racket_id = r["track_id"]
                    closest_ball_id = b["track_id"]

        if closest_racket_id is not None:
            results.append({
                "frame": frame,
                "racket_id": closest_racket_id,
                "ball_id": closest_ball_id,
                "distance": min_dist
            })

    df_dist = pd.DataFrame(results)
    print(f"[distance] Computed distances for {len(df_dist)} frames")
    return df_dist
