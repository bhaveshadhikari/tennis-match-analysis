"""
merger.py
Merges detections.csv and ball_detections.csv into a single combined_detections.csv.
Adds center coordinates (cx, cy) and tags each row with its source model.
"""

import pandas as pd


COLS = ["frame", "track_id", "label", "x1", "y1", "x2", "y2", "confidence"]


def merge_detections(main_csv: str, ball_csv: str, output_csv: str) -> pd.DataFrame:
    df_main = pd.read_csv(main_csv)[COLS]
    df_ball = pd.read_csv(ball_csv)[COLS]

    # Offset ball track IDs to avoid clashes with person/racket IDs
    df_ball["track_id"] = df_ball["track_id"] + 10000

    df_main["source"] = "main"
    df_ball["source"] = "ball"

    df_all = pd.concat([df_main, df_ball], ignore_index=True)
    df_all = df_all.sort_values(["frame", "track_id"]).reset_index(drop=True)

    df_all["cx"] = ((df_all["x1"] + df_all["x2"]) / 2).round(2)
    df_all["cy"] = ((df_all["y1"] + df_all["y2"]) / 2).round(2)

    df_all.to_csv(output_csv, index=False)
    print(f"[merger] Saved → {output_csv}  ({len(df_all)} rows)")
    return df_all
