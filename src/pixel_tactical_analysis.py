"""
Pixel-space (frame-relative) tactical analysis for real broadcast footage.

IMPORTANT: unlike tactical_analysis.py (which operates on real pitch meters
from SoccerTrack v2 ground truth), everything here operates directly on
detection pixel coordinates. No camera calibration / homography is applied.
Shapes and zones are frame-relative approximations, not true pitch geometry.
See docs/limitations.md for the full rationale.
"""
import numpy as np
from scipy.spatial import ConvexHull


def get_team_positions_pixel(result, model, smoothed_team):
    """
    Get pixel-space (x_center, y_bottom) 'feet position' per team for one
    frame's tracking result.

    smoothed_team: dict of {track_id: {'team': 'A'|'B', 'consistency': float}}
    """
    positions = {'A': [], 'B': []}
    if result.boxes.id is None:
        return positions
    for box, track_id, cls in zip(result.boxes.xyxy, result.boxes.id, result.boxes.cls):
        if model.names[int(cls)] != 'player':
            continue
        tid = int(track_id)
        if tid not in smoothed_team:
            continue
        team = smoothed_team[tid]['team']
        x1, y1, x2, y2 = box.tolist()
        positions[team].append((int((x1 + x2) / 2), int(y2)))
    return positions


def pixel_convex_hull(points):
    """Compute convex hull for a set of (x, y) pixel points. Returns closed polygon or None."""
    if len(points) < 3:
        return None
    coords = np.array(points)
    hull = ConvexHull(coords)
    hull_pts = coords[hull.vertices]
    return np.vstack([hull_pts, hull_pts[0]])


def classify_pixel_zone(x_center, frame_width):
    """Split the camera frame into horizontal thirds. A proxy, not a real pitch zone."""
    third = frame_width / 3
    if x_center < third:
        return 'Left Third'
    elif x_center < 2 * third:
        return 'Middle Third'
    return 'Right Third'


def zone_occupancy_pixel(positions, frame_width):
    """Count players per frame-relative zone per team."""
    counts = {'A': {'Left Third': 0, 'Middle Third': 0, 'Right Third': 0},
              'B': {'Left Third': 0, 'Middle Third': 0, 'Right Third': 0}}
    for team in ['A', 'B']:
        for (x, _) in positions[team]:
            zone = classify_pixel_zone(x, frame_width)
            counts[team][zone] += 1
    return counts
