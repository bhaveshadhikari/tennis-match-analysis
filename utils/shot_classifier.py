"""
shot_classifier.py
Assigns a shot type to each detected hit frame based on
the relative position of the racket to the closest player.

Shot types: serve, forward shot, back shot, neutral
"""

import numpy as np
import pandas as pd


def assign_closest_player(df_all: pd.DataFrame, hits: pd.DataFrame, expanded_frames: set) -> tuple:
    df_context = df_all[
        (df_all["frame"].isin(expanded_frames)) &
        (df_all["label"].isin(["person", "racket"]))
    ].copy()

    results = []

    for frame in df_context["frame"].unique():
        frame_df = df_context[df_context["frame"] == frame]
        hit_row = hits[hits["frame"] == frame]

        if hit_row.empty:
            continue

        racket_id = hit_row.iloc[0]["racket_id"]
        racket = frame_df[(frame_df["label"] == "racket") & (frame_df["track_id"] == racket_id)]
        players = frame_df[frame_df["label"] == "person"]

        if racket.empty or players.empty:
            continue

        rx, ry = racket.iloc[0]["cx"], racket.iloc[0]["cy"]

        for _, p in players.iterrows():
            dist = np.sqrt((rx - p["cx"]) ** 2 + (ry - p["cy"]) ** 2)
            results.append({
                "frame": frame,
                "racket_id": racket_id,
                "ball_id": hit_row.iloc[0]["ball_id"],
                "player_id": p["track_id"],
                "distance": dist
            })

    df_pr = pd.DataFrame(results)

    # Closest player per frame
    frame_player_map = (
        df_pr.groupby("frame")
        .apply(lambda x: x.loc[x["distance"].idxmin(), "player_id"])
        .to_dict()
    )

    return df_pr, frame_player_map


def classify_shots(df_all: pd.DataFrame, hits: pd.DataFrame, frame_player_map: dict,
                   x_thresh: float, y_thresh: float) -> pd.DataFrame:
    results = []

    hit_frames = set(hits["frame"].values)

    for frame, group in df_all[df_all["frame"].isin(frame_player_map)].groupby("frame"):
        player_id = frame_player_map[frame]

        hit_row = hits[hits["frame"] == frame]
        if hit_row.empty:
            continue

        racket_id = hit_row.iloc[0]["racket_id"]

        person = group[(group["label"] == "person") & (group["track_id"] == player_id)]
        racket = group[(group["label"] == "racket") & (group["track_id"] == racket_id)]

        if person.empty or racket.empty:
            continue

        cx_p, cy_p = person.iloc[0]["cx"], person.iloc[0]["cy"]
        cx_r, cy_r = racket.iloc[0]["cx"], racket.iloc[0]["cy"]

        dx = cx_r - cx_p
        dy = cy_r - cy_p

        if abs(dx) < x_thresh and abs(dy) < y_thresh:
            shot = "neutral"
        elif cy_r < cy_p - y_thresh:
            shot = "serve"
        elif dx > x_thresh:
            shot = "forward shot"
        elif dx < -x_thresh:
            shot = "back shot"
        else:
            shot = "neutral"

        temp = group[
            (group["label"] == "ball") |
            ((group["label"] == "person") & (group["track_id"] == player_id))
        ].copy()
        
        temp["result"] = shot
        temp["distance_pr"] = np.sqrt(dx ** 2 + dy ** 2)
        results.append(temp)

    if not results:
        return pd.DataFrame()

    df_out = pd.concat(results, ignore_index=True)
    df_out = df_out[["frame", "track_id", "label", "result", "distance_pr"]]

    print(f"[classifier] Classified {len(df_out)} rows across {df_out['frame'].nunique()} hit frames")
    return df_out
