"""
Frame-relative zone occupancy proxy for real broadcast footage.

IMPORTANT: this is NOT real pitch coordinates. Without a homography calibration
per camera angle, we cannot know real-world attacking direction or true pitch
position. This module reports pixel-space horizontal thirds of the camera frame
as an honest, clearly-labeled approximation only -- see docs/limitations.md.
"""
from collections import defaultdict


def classify_pixel_zone(x_center, frame_width):
    """Split the camera frame into horizontal thirds. A proxy, not a real pitch zone."""
    third = frame_width / 3
    if x_center < third:
        return 'Left Third (frame)'
    elif x_center < 2 * third:
        return 'Middle Third (frame)'
    return 'Right Third (frame)'


def compute_zone_proxy(track_results, model, smoothed_team, frame_width, sample_every=25):
    """
    Compute frame-relative zone occupancy per team across sampled frames.

    smoothed_team: dict of {track_id: {'team': 'A'|'B', 'consistency': float}}
    Returns: dict of {'A': {zone: count}, 'B': {zone: count}}
    """
    zone_counts = defaultdict(lambda: defaultdict(int))

    for frame_idx in range(0, len(track_results), sample_every):
        result = track_results[frame_idx]
        if result.boxes.id is None:
            continue
        for box, track_id, cls in zip(result.boxes.xyxy, result.boxes.id, result.boxes.cls):
            if model.names[int(cls)] != 'player':
                continue
            tid = int(track_id)
            if tid not in smoothed_team:
                continue
            team = smoothed_team[tid]['team']
            x1, _, x2, _ = box.tolist()
            x_center = (x1 + x2) / 2
            zone = classify_pixel_zone(x_center, frame_width)
            zone_counts[team][zone] += 1

    return zone_counts
