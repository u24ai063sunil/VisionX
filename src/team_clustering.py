"""
Team assignment via jersey color clustering, with temporal smoothing to
handle single-frame clustering noise.
"""
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans


def get_dominant_jersey_color(crop):
    """
    Extract the dominant non-grass color from a player's torso region.

    Uses the central 15-45% vertical band (avoids head and grass-adjacent legs)
    and center 60% horizontal band (avoids bbox edge bleed), then filters out
    grass-green pixels before taking the median color.
    """
    if crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if h < 10 or w < 6:
        return None

    y1, y2 = int(h * 0.15), int(h * 0.45)
    x1, x2 = int(w * 0.2), int(w * 0.8)
    torso = crop[y1:y2, x1:x2]
    if torso.size == 0:
        return None

    pixels = torso.reshape(-1, 3).astype(np.float32)  # BGR
    b, g, r = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    is_grass = (g > b + 15) & (g > r + 15)
    non_grass_pixels = pixels[~is_grass]

    if len(non_grass_pixels) < 5:
        return None
    return np.median(non_grass_pixels, axis=0)


def cluster_teams(jersey_colors, n_teams=2, random_state=42):
    """K-means cluster jersey colors into team groups."""
    km = KMeans(n_clusters=n_teams, n_init=10, random_state=random_state)
    labels = km.fit(jersey_colors).labels_
    return labels, km.cluster_centers_


def smooth_team_assignment(track_team_history):
    """
    Resolve per-frame clustering noise via majority vote per track ID.

    track_team_history: dict of {track_id: [(frame_idx, team_label), ...]}
    Returns: dict of {track_id: {'team': majority_label, 'consistency': float}}
    """
    smoothed = {}
    for tid, history in track_team_history.items():
        teams = [t for _, t in history]
        majority_team = max(set(teams), key=teams.count)
        consistency = teams.count(majority_team) / len(teams)
        smoothed[tid] = {'team': majority_team, 'consistency': consistency}
    return smoothed


def build_team_history(track_results, model, video_frames_fn, sample_every=25):
    """
    Run jersey clustering across sampled frames and build a per-track team history.

    video_frames_fn(frame_idx) -> BGR image for that frame index.
    Returns: dict of {track_id: [(frame_idx, 'A'|'B'), ...]}
    """
    track_team_history = defaultdict(list)

    for frame_idx in range(0, len(track_results), sample_every):
        result = track_results[frame_idx]
        if result.boxes.id is None:
            continue

        img = video_frames_fn(frame_idx)
        if img is None:
            continue

        colors, tids = [], []
        for box, track_id, cls in zip(result.boxes.xyxy, result.boxes.id, result.boxes.cls):
            if model.names[int(cls)] != 'player':
                continue
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            color = get_dominant_jersey_color(img[y1:y2, x1:x2])
            if color is not None:
                colors.append(color)
                tids.append(int(track_id))

        if len(colors) < 4:
            continue

        labels, centers = cluster_teams(np.array(colors))
        ref_cluster = np.argmin(centers[:, 0])  # lower avg blue = reference "Team A"

        for tid, label in zip(tids, labels):
            team = 'A' if label == ref_cluster else 'B'
            track_team_history[tid].append((frame_idx, team))

    return track_team_history
