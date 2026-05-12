# Tennis Racket Movement Analyzer - Bhavesh Adhikari

Analyzes a tennis match video using pre-computed detection CSVs.
Detects hit events, classifies shot types, and produces an annotated video with graphs.

Detection and tracking (YOLO + ByteTrack) run separately in Colab and produce the input CSVs.

---

## Folder Structure

```
full-analysis/
    # contains all the files that is required to perform the analysis in long length video
notebooks/
    player_racket_ball_detection.ipynb

input/
    detections.csv          # person + racket detections from main model
    ball_detections.csv     # ball detections from ball model
    combined_detections.csv # merged output (auto-generated)
    input.mp4               # source video

output/
    video_analyzed.mp4      # annotated output video
    result.csv              # shot classification results
    graphs/
        distance_over_time.png
        shot_frequency.png

utils/
    merger.py
    distance.py
    hit_detector.py
    shot_classifier.py
    visualizer.py
    video_writer.py

main.py
```

---

## Setup

```bash
pip install pandas numpy scipy matplotlib opencv-python ultralytics
```

---

## Usage

1. Place your input files in `input/`.
2. Adjust parameters in `main.py` if needed.
3. Run:

```bash
python main.py
```

Progress prints to the console for each step.

---

## Parameters (main.py)

| Parameter | Default | Description |
|---|---|---|
| `DISTANCE_THRESHOLD` | 120 | Max pixel distance between racket and ball to count as a hit candidate |
| `FRAME_WINDOW` | 5 | Half-size of frame window when searching for nearest racket/ball pair |
| `HIT_ORDER` | 3 | Local minima sensitivity; higher means fewer but more prominent hits |
| `HIT_EXPAND_OFFSET` | 3 | Frames before/after a hit to include for player-racket matching |
| `SHOT_X_THRESHOLD` | 30 | Horizontal pixel offset to distinguish forward/back shot from neutral |
| `SHOT_Y_THRESHOLD` | 30 | Vertical pixel offset to identify a serve |

---

## Utils

### `merger.py`
Reads `detections.csv` (person + racket) and `ball_detections.csv` (ball).
Offsets ball track IDs by 10000 to avoid clashes with person/racket IDs.
Adds center coordinates (`cx`, `cy`) to each row.
Saves the result as `combined_detections.csv`.

### `distance.py`
For each frame, looks within a frame window to find the nearest racket-ball pair.
Returns a DataFrame of `(frame, racket_id, ball_id, distance)` for frames where
the pair is within the distance threshold.

### `hit_detector.py`
Finds local minima of the distance signal using `scipy.signal.argrelextrema`.
Filters minima below the distance threshold to identify actual hit events.
Also expands hit frames by a configurable offset for context window creation.

### `shot_classifier.py`
For each hit frame, finds the closest player to the racket.
Computes the vector from player center to racket center.
Classifies the shot as `serve`, `forward shot`, `back shot`, or `neutral`
based on configurable x/y thresholds.

### `visualizer.py`
Produces two graphs saved to `output/graphs/`:
- `distance_over_time.png`: distance signal with hit events marked in red.
- `shot_frequency.png`: bar chart of how many times each shot type occurred.

### `video_writer.py`
Reads the source video frame by frame.
Draws bounding boxes and track IDs for all detections.
Overlays a running scoreboard (shot counts) in the top-left corner.
Displays the current shot type as a large label for ~2 seconds around each hit frame.
Saves the result to `output/video_analyzed.mp4`.

---

## Output

| File | Description |
|---|---|
| `output/video_analyzed.mp4` | Annotated video with scoreboard and shot labels |
| `output/result.csv` | Per-frame shot classification (frame, player, shot type, distance) |
| `output/graphs/distance_over_time.png` | Racket-ball distance with hit markers |
| `output/graphs/shot_frequency.png` | Shot type distribution bar chart |
| `notebooks\player_racket_ball_detection.ipynb` | Colab Notebook to get detection CV for large videos|

## Output Link
Short Sample :
<video controls src="https://github.com/bhaveshadhikari/tennis-match-analysis/blob/main/output/video_analyzed.mp4" title="input\input.mp4"></video>

Full Length : https://youtu.be/8-rpe6ig7ug

<iframe width="560" height="315" src="https://www.youtube.com/embed/8-rpe6ig7ug?si=hvrEW9Z8mw9l8f4n" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>