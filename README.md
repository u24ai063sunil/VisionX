# Tactics Cam — Football Tactical Intelligence Pipeline

**ACM RVCE Code Cup 2026 — Computer Vision Track — Problem Statement 1: "Tactics Cam"**
Team: VisionX

A computer vision pipeline that turns raw football broadcast footage into tactical insight — player/ball detection, player tracking, team assignment, and pitch-level tactical analysis (team shape, zone occupancy).

---

## Problem Statement

Football analysts extract tactical insights (formations, zone occupancy, pressing shape) by manually reviewing footage frame by frame — slow, subjective, and impossible to scale across matches. This project automates that process: detect players/ball, track them across frames, assign team identity, and compute tactical metrics automatically.

---

## What This System Does

| Capability | Status | Validated On |
|---|---|---|
| Player / GK / referee / ball detection | ✅ Working | Roboflow validation set + real DFL broadcast footage |
| Multi-object tracking (persistent IDs) | ✅ Working | Real DFL broadcast footage |
| Team assignment (jersey color clustering) | ✅ Working — 98% track consistency | Real DFL broadcast footage |
| Tactical minimap (team shape, convex hulls) | ✅ Working | SoccerTrack v2 ground-truth pitch coordinates |
| Zone occupancy (defensive/mid/attacking thirds) | ✅ Working | SoccerTrack v2 ground-truth pitch coordinates |
| Zone occupancy proxy (frame-relative thirds) | ✅ Working | Real DFL broadcast footage (pixel-space approximation) |
| Homography (pixel → real pitch coordinates on live footage) | ⏳ Not yet implemented | — |

**Honest scope note:** the tactical-reasoning layer (team shape, real zone occupancy) is fully validated against ground-truth pitch coordinates from SoccerTrack v2. Detection, tracking, and team assignment are fully validated on real, unseen DFL broadcast footage. The bridge between the two — mapping detections on live footage into real pitch coordinates via homography — was scoped as a stretch goal and is documented as future work rather than shipped as an unreliable approximation. See `docs/limitations.md` for details.

---

## Architecture

```
                    ┌─────────────────────┐
                    │  Roboflow Dataset    │
                    │  (4-class labels)    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  YOLOv8n Training    │
                    │  (Kaggle T4 GPU)     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Trained Detector    │
                    │  mAP50: 0.815        │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                                 ▼
   ┌─────────────────────┐          ┌─────────────────────┐
   │  Real DFL clips      │          │  SoccerTrack v2 GSR  │
   │  (unseen footage)    │          │  (ground-truth data) │
   └──────────┬───────────┘          └──────────┬──────────┘
              │                                 │
   ┌──────────▼───────────┐          ┌──────────▼──────────┐
   │  Tracking (ByteTrack) │          │  Coordinate parser   │
   │  Team clustering      │          │  Team shape / zones  │
   │  Zone proxy (pixel)   │          │  (real pitch coords)  │
   └──────────────────────┘          └──────────────────────┘
```

---

## Results

**Detection (YOLOv8n, fine-tuned on Roboflow 4-class dataset):**
- Overall mAP50: 0.815 | mAP50-95: 0.540
- Player: mAP50 0.988 | Goalkeeper: mAP50 0.947 | Referee: mAP50 0.929
- Ball: mAP50 0.396 — a known, documented industry-wide difficulty (small, fast, frequently occluded object). Stated openly rather than hidden.

**Tracking + team assignment (real DFL footage, unseen during training):**
- Persistent IDs via ByteTrack across a full 750-frame broadcast clip
- Team assignment via jersey-color K-means clustering: 45/46 tracked players (98%) showed ≥70% team-label consistency after majority-vote smoothing across the clip
- The one low-confidence track is attributed to a tracker ID handoff (occlusion/crossing players), not a clustering failure

**Tactical layer (validated on SoccerTrack v2 ground-truth):**
- Team shape (convex hull) and zone occupancy computed directly from real 2D pitch coordinates
- Demonstrated team-shape evolution across a 4-frame, ~9-second sequence

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── tactics_cam_pipeline.ipynb      # Full Kaggle notebook, cell-by-cell
├── src/
│   ├── detection.py                    # YOLOv8 training + inference helpers
│   ├── tracking.py                     # ByteTrack integration
│   ├── team_clustering.py              # Jersey color K-means + smoothing
│   ├── tactical_analysis.py            # Convex hull, zone occupancy (GSR-based)
│   └── zone_proxy.py                   # Frame-relative zone proxy (real footage)
├── outputs/
│   ├── detection_samples/              # Annotated detection screenshots
│   ├── tactical_minimaps/              # Team shape / zone visualizations
|   |__ tracking_team_assignment/       # team assignment / zone occupancy
│   └── demo_video.mp4                  # End-to-end demo clip
├── docs/
│   ├── architecture.png                # System diagram (see above)
│   └── limitations.md                  # Honest scope notes, known issues
└── models/
    └── best.pt                         # Trained YOLOv8n weights (or link if too large for repo)
```

---

## Setup

```bash
git clone <this-repo-url>
cd tactics-cam
pip install -r requirements.txt
```

Trained model weights: place `best.pt` in `models/`, or download from `<Kaggle dataset / GitHub release link>`.

## Running Inference

```python
from ultralytics import YOLO

model = YOLO('models/best.pt')
results = model.track(source='path/to/clip.mp4', conf=0.35, persist=True, tracker='bytetrack.yaml')
```

See `notebooks/tactics_cam_pipeline.ipynb` for the full pipeline, cell by cell, including team clustering and tactical analysis.

---

## Datasets Used

1. **[Football Detection (4-class)](https://universe.roboflow.com/footballresearch/football-detection-4)** — Roboflow — training data for the detector
2. **[SoccerTrack v2](https://github.com/AtomScott/SoccerTrack-v2)** — ground-truth Game State Reconstruction (GSR) data — used to validate the tactical reasoning layer
3. **DFL Bundesliga Data Shootout clips** — real broadcast footage — used to test detector/tracker/clustering generalization on unseen data

---

## Team

Team VisonX — ACM RVCE Code Cup 2026, Computer Vision Track
Sunil Jaat , U24AI063 , AI 3rd Year
Md Aftab Siddiqui , U24AI058 , AI 3rd Year
Aarju Pawara , U24AI030 , AI 3rd Year
Vaishnav Mahla , U24AI120 , CSE 3rd Year
