"""
video_writer.py
Renders the final annotated output video.
Draws bounding boxes for all detections on every frame.
On hit frames: shows the shot label and updates the running scoreboard.
"""

import cv2
import pandas as pd


COLORS = {
    "person": (0, 255, 0),
    "racket": (255, 0, 0),
    "ball": (0, 0, 255)
}


def render_video(
    input_video: str,
    output_video: str,
    df_det: pd.DataFrame,
    df_shots: pd.DataFrame
):
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    # Build event map and display windows from shot results
    event_map = {}
    event_windows = []

    for _, row in df_shots.iterrows():
        frame = int(row["frame"])
        result = str(row["result"]).lower()
        event_map[frame] = result

        start = frame - int(0.2 * fps)
        end = frame + int(1.8 * fps)
        event_windows.append((start, end, result))

    counter = {"forward shot": 0, "back shot": 0, "serve": 0, "neutral": 0}

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[video_writer] Rendering {total_frames} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw all detections
        frame_df = df_det[df_det["frame"] == frame_idx]
        for _, row in frame_df.iterrows():
            x1, y1, x2, y2 = map(int, [row.x1, row.y1, row.x2, row.y2])
            color = COLORS.get(row.label, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{row.label} {row.track_id}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

        # Update counter on exact hit frame
        if frame_idx in event_map:
            event = event_map[frame_idx]
            if event in counter:
                counter[event] += 1

        # Scoreboard
        cv2.rectangle(frame, (20, 10), (420, 180), (0, 0, 0), -1)
        scoreboard = [
            f"Forward: {counter['forward shot']}",
            f"Backward: {counter['back shot']}",
            f"Serve/Smash: {counter['serve']}",
            f"Neutral: {counter['neutral']}"
        ]
        for i, text in enumerate(scoreboard):
            cv2.putText(
                frame, text, (40, 60 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
            )

        # Event label
        event_y = 60 + len(scoreboard) * 30 + 70
        for start, end, result in event_windows:
            if start <= frame_idx <= end:
                cv2.putText(
                    frame, result.upper(), (40, event_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA
                )
                break

        out.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"[video_writer]   {frame_idx}/{total_frames} frames done")

    cap.release()
    out.release()
    print(f"[video_writer] Saved → {output_video}")
