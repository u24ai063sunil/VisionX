# Known Limitations & Scope Notes

This document is intentional. We'd rather state clearly what does and doesn't work than have a judge discover a gap unannounced.

## 1. Ball detection accuracy
mAP50 for the ball class is 0.396, substantially lower than player/goalkeeper/referee (all 0.92+). This is a well-documented, industry-wide difficulty: the ball is small, fast-moving, and frequently occluded by players. We did not attempt to hide or downplay this — it's stated on our pitch deck and here. Potential future improvements: higher input resolution specifically for ball detection, tiled/sliced inference, or a dedicated small-object detection head.

## 2. Homography / real-world pitch mapping on live footage
The tactical reasoning layer (team shape via convex hull, zone occupancy) is fully built and validated — but only against SoccerTrack v2's **ground-truth** 2D pitch coordinates, not against our own detections on real DFL broadcast footage.

To connect the two, we would need a homography transform: mapping pixel-space detections to real pitch coordinates (meters), which requires either:
- Manual per-camera-angle calibration (4+ known pitch landmark points per clip), or
- An automated pitch-line/keypoint detection model

We attempted manual calibration during the sprint and found it produced numerically unstable results when calibration points were closely clustered in one image region (e.g. just the goal box) — small pixel errors get massively amplified when extrapolating across the full pitch. A wider, well-distributed set of landmarks (e.g. center circle + both penalty boxes) is needed for a stable result, and reliably identifying such landmarks by eye across arbitrary broadcast angles proved to be a bigger task than the remaining sprint time allowed.

We also attempted an automated approach: masking the pitch region by color and running Hough line detection within it to auto-locate pitch markings. This also failed in practice — the pitch color mask picked up large sections of crowd, advertising boards, and off-pitch elements under real broadcast lighting conditions, producing hundreds of spurious "line segments" rather than clean pitch lines. Robust pitch-line detection under varying stadium lighting is a non-trivial CV sub-problem in its own right (entire research papers address just this), not something reliably solvable in a time-boxed sprint slot.

Rather than ship an unstable or misleading pixel-to-pitch mapping, we chose to:
- Fully validate the tactical logic on ground-truth data (proving the *reasoning* is correct)
- Fully validate detection/tracking/team-assignment on real, unseen footage (proving the *perception* generalizes)
- Build an honest, clearly-labeled **pixel-space tactical layer** (team shape via convex hull + frame-relative left/middle/right zone occupancy) directly on our own detections on real footage, explicitly NOT claiming real pitch coordinates
- Render this into a full annotated demo video with a live stats panel, so the tactical signal is visibly demonstrated on unseen broadcast footage even without calibration
- Document the homography bridge as the clear next milestone

## 3. Team assignment consistency
Jersey-color clustering (K-means) achieves 98% (45/46) track-level consistency after majority-vote smoothing across a sampled real DFL clip. The one flagged low-confidence track is attributed to a ByteTrack ID handoff during player occlusion/crossing — i.e. the tracker briefly assigned one track ID to two different physical players — rather than a failure of the color-clustering logic itself.

## 4. Tracking ID churn
ByteTrack assigns new IDs relatively often on broadcast footage with camera motion and player occlusion (observed track IDs climbing quickly, e.g. reaching 100+ within the first ~100 frames of a 750-frame clip). This is a known characteristic of IoU/motion-based trackers on wide-angle broadcast footage and did not meaningfully affect the team-clustering result above, since majority-vote smoothing is robust to short-lived ID fragments.

## 5. Dataset scope
- Detector trained only on the Roboflow 4-class dataset (~2000 images, single source). A larger, more diverse training set (multiple leagues/broadcasts) would likely improve generalization further, particularly for ball detection.
- SoccerTrack v2 ground-truth validation was performed on a single match's data due to file size constraints (2.5GB per half); broader validation across multiple matches was out of scope for this sprint.
