"""
visualizer.py
Generates all output graphs and saves them to the graphs directory.
- distance_over_time.png: racket-ball distance per frame with hit markers
- shot_frequency.png: bar chart of shot type counts
"""

import matplotlib.pyplot as plt
import pandas as pd
import os


SHOT_COLORS = {
    "forward shot": "green",
    "back shot": "red",
    "serve": "blue",
    "neutral": "gray"
}

ALL_SHOTS = ["forward shot", "back shot", "serve", "neutral"]


def plot_distance(df_dist: pd.DataFrame, hits: pd.DataFrame, output_path: str):
    plt.figure(figsize=(12, 5))
    plt.plot(df_dist["frame"], df_dist["distance"], label="Racket-Ball Distance")

    if not hits.empty:
        plt.scatter(hits["frame"], hits["distance"], color="red", label="Hit Events", zorder=5)

    plt.xlabel("Frame")
    plt.ylabel("Distance (pixels)")
    plt.title("Racket vs Ball Distance Over Time")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[visualizer] Saved → {output_path}")


def plot_shot_frequency(df_shots: pd.DataFrame, output_path: str):
    counts = df_shots["result"].value_counts().reindex(ALL_SHOTS, fill_value=0)

    plt.figure(figsize=(6, 4))
    bars = plt.bar(counts.index, counts.values, color=[SHOT_COLORS[k] for k in counts.index])

    for bar in bars:
        y = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, y, str(int(y)), ha="center", va="bottom")

    plt.title("Shot Type Frequency")
    plt.xlabel("Shot Type")
    plt.ylabel("Count")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[visualizer] Saved → {output_path}")
