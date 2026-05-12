"""
main.py
Tennis Racket Movement Analyzer
--------------------------------
Entry point. Configure all paths and parameters below, then run.
"""

import os
import sys

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
INPUT_DIR = "input"
OUTPUT_DIR = "output"
GRAPHS_DIR = os.path.join(OUTPUT_DIR, "graphs")

MAIN_CSV    = os.path.join(INPUT_DIR, "detections.csv")
BALL_CSV    = os.path.join(INPUT_DIR, "ball_detections.csv")
INPUT_VIDEO = os.path.join(INPUT_DIR, "input.mp4")

COMBINED_CSV = os.path.join(INPUT_DIR, "combined_detections.csv")
RESULT_CSV   = os.path.join(OUTPUT_DIR, "result.csv")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "video_analyzed.mp4")

GRAPH_DISTANCE  = os.path.join(GRAPHS_DIR, "distance_over_time.png")
GRAPH_SHOT_FREQ = os.path.join(GRAPHS_DIR, "shot_frequency.png")

# ──────────────────────────────────────────────
# PARAMETERS
# ──────────────────────────────────────────────

# Max pixel distance between racket and ball to count as a hit candidate
DISTANCE_THRESHOLD = 120

# Half-size of the frame window when searching for nearest racket/ball pair
FRAME_WINDOW = 5

# Local minima detection sensitivity (higher = fewer, more prominent hits)
HIT_ORDER = 3

# Frames before/after a hit frame to include for player-racket matching
HIT_EXPAND_OFFSET = 3

# Pixel thresholds for classifying shot direction from player-racket offset
SHOT_X_THRESHOLD = 30
SHOT_Y_THRESHOLD = 30


# ──────────────────────────────────────────────
# RUNNER
# ──────────────────────────────────────────────
def main():
    from utils import (
        merge_detections,
        compute_racket_ball_distance,
        detect_hits,
        expand_hit_frames,
        assign_closest_player,
        classify_shots,
        plot_distance,
        plot_shot_frequency,
        render_video,
    )

    os.makedirs(GRAPHS_DIR, exist_ok=True)

    # Step 1: Merge detection CSVs
    print("\n[1/6] Merging detections...")
    df_all = merge_detections(MAIN_CSV, BALL_CSV, COMBINED_CSV)

    # Step 2: Compute racket-ball distance over time
    print("\n[2/6] Computing racket-ball distances...")
    df_dist = compute_racket_ball_distance(df_all, window=FRAME_WINDOW, threshold=DISTANCE_THRESHOLD)

    if df_dist.empty:
        print("[ERROR] No racket-ball pairs found within threshold. Check your CSVs and threshold.")
        sys.exit(1)

    # Step 3: Detect hit events
    print("\n[3/6] Detecting hit events...")
    hits = detect_hits(df_dist, distance_threshold=DISTANCE_THRESHOLD, order=HIT_ORDER)

    if hits.empty:
        print("[ERROR] No hits detected. Try lowering DISTANCE_THRESHOLD or HIT_ORDER.")
        sys.exit(1)

    expanded_frames = expand_hit_frames(hits["frame"].values, offset=HIT_EXPAND_OFFSET)

    # Step 4: Classify shots
    print("\n[4/6] Classifying shots...")
    df_pr, frame_player_map = assign_closest_player(df_all, hits, expanded_frames)
    df_shots = classify_shots(
        df_all, hits, frame_player_map,
        x_thresh=SHOT_X_THRESHOLD,
        y_thresh=SHOT_Y_THRESHOLD
    )

    if df_shots.empty:
        print("[ERROR] Shot classification returned no results.")
        sys.exit(1)

    df_shots.to_csv(RESULT_CSV, index=False)
    print(f"[4/6] Result CSV saved → {RESULT_CSV}")

    # Step 5: Generate graphs
    print("\n[5/6] Generating graphs...")
    plot_distance(df_dist, hits, GRAPH_DISTANCE)
    plot_shot_frequency(df_shots, GRAPH_SHOT_FREQ)

    # Step 6: Render annotated video
    print("\n[6/6] Rendering output video...")
    render_video(INPUT_VIDEO, OUTPUT_VIDEO, df_all, df_shots)

    print("\nDone.")
    print(f"  Video  → {OUTPUT_VIDEO}")
    print(f"  CSV    → {RESULT_CSV}")
    print(f"  Graphs → {GRAPHS_DIR}/")


if __name__ == "__main__":
    main()
