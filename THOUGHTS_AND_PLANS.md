# PLAN.md — My Approach to the Padel Shot Classification Assignment

## How I Thought About the Problem

When I first read the assignment, I broke it into two separate questions:

1. **When** did a shot happen?
2. **What kind** of shot was it?

These are different problems. The first is about detecting a specific moment in time. The second is about geometry at that moment.

---

## Step 1 — Object Detection

I used **YOLOv8x** (pretrained on COCO) because it already knows how to detect:
- `person` — I map this to "player"
- `sports ball` — I map this to "ball"
- `tennis racket` — I map this to "racket"

but it doesnt detect the ball movement
so i look for the finetuned ball detection models.
https://huggingface.co/RJTPP/tennis-ball-detection



Padel rackets look different from tennis rackets but close enough for a prototype. I run YOLOv8x with ByteTrack (`model.track(..., persist=True)`) so each player gets a consistent ID across frames.

similarly for ball detection

---

## Step 2 — Finding the Contact Moment

A shot happens when the racket and ball make contact. In video, that corresponds to the frame where the distance between the **ball center** and **racket center** is at a local minimum.

I compute Euclidean distance between the two centers in every frame, then scan for local minima within a sliding window of 5 frames. Any minimum below 150 pixels is treated as a contact event.

Due to poor accuracy of ball and contact moment, and the flicker of the racket, 
i looked for +-5(FRAME WINDOW) frames for each ball moment to identity the nearest racket

---

## Step 3 — Shot Classification

At each contact frame, I look at three things:

**Smash detection:**
If the racket center Y position is above the player's bounding box top (with a small buffer), the player is hitting with a raised arm. That is a smash.

**Forehand vs Backhand:**
I compare the racket center X with the player body center X (midpoint of the player bounding box).
- Racket on the right side of body center → **forehand** (assumes right-handed)
- Racket on the left side → **backhand**

This is a simplification. A proper classifier would use wrist and elbow keypoints from pose estimation. But for a prototype with no labeled training data, this geometric rule gives reasonable results.

---

## Step 4 — Output

Each shot event stores: frame ID, timestamp, shot type, player ID, ball position, racket position.
frame,track_id,label,result,distance_pr
This gets saved to CSV. The annotated video draws bounding boxes every frame and overlays the shot label for 20 frames after each contact event so it is readable in the demo.

---

## What I Would Improve With More Time

1. ** valid shot detection** : catchinf and passing the ball casually is also being counted as a tennis 
1. **Racket tracking:** The model is confused with racket for arm or legs too,..

2. **Pose estimation:** Using MediaPipe to get wrist, elbow, and shoulder keypoints would give much more accurate forehand vs backhand classification. The current rule works but it is brittle for players positioned at angles.

3. **Player-racket-ball association:** right now the path follows -> detect ball -> closest racket -> closest player
but it should also agree with tat the palyer mapped should be the active player. 
which might not always be the case.


4. **Fine-tuned model:** Training YOLOv8 on actual padel footage with labeled balls and rackets would improve detection accuracy significantly, especially for the small fast-moving ball.

---

## Challenges I Faced

- Ball detection is inconsistent. COCO-trained models often miss the ball when it is moving fast or when the frame is crowded.
- The contact threshold distance (120px) needed manual tuning. Too small and many real contacts are missed. Too large and false positives appear.
- YOLO sometimes detects spectators as players. Adding a court boundary mask would help filter those out.

---

## Tools Used

- Python 3.10
- Ultralytics YOLOv8 (detection + tracking)
- OpenCV (video I/O, frame annotation)
- NumPy (distance calculations)
