"""
Tactical analysis on real pitch coordinates: team shape (convex hull) and
zone occupancy. Built and validated against SoccerTrack v2 ground-truth GSR data.
"""
import json

import numpy as np
from scipy.spatial import ConvexHull


def load_gsr(gsr_json_path):
    """Load a SoccerTrack v2 GSR annotation file."""
    with open(gsr_json_path, 'r') as f:
        return json.load(f)


def get_frame_players(gsr_data, image_id):
    """Extract player positions + attributes for a single frame from GSR data."""
    frame_anns = [a for a in gsr_data['annotations']
                  if a['image_id'] == image_id and a.get('category_id') in [1, 2, 3, 4]]

    players = []
    for a in frame_anns:
        pitch = a['bbox_pitch']
        if pitch['x_bottom_middle'] is None:
            continue
        players.append({
            'track_id': a['track_id'],
            'role': a['attributes'].get('role'),
            'team': a['attributes'].get('team'),
            'jersey': a['attributes'].get('jersey'),
            'x': pitch['x_bottom_middle'],
            'y': pitch['y_bottom_middle'],
        })
    return players


def team_shape_hull(players, team):
    """Compute the convex hull (team shape) for one team's players in a frame."""
    team_players = [p for p in players if p['team'] == team]
    coords = np.array([[p['x'], p['y']] for p in team_players])
    if len(coords) < 3:
        return None
    hull = ConvexHull(coords)
    hull_pts = coords[hull.vertices]
    return np.vstack([hull_pts, hull_pts[0]])  # closed polygon


def classify_zone(x, pitch_length=105):
    """Classify a pitch x-coordinate into defensive/middle/attacking third."""
    if x < -pitch_length / 6:
        return 'Defensive Third'
    elif x > pitch_length / 6:
        return 'Attacking Third'
    return 'Middle Third'


def zone_occupancy(players, pitch_length=105):
    """
    Count players per zone per team, correcting for attack direction
    (the 'right' team attacks toward -x, so their zones are flipped).
    """
    counts = {'left': {'Defensive Third': 0, 'Middle Third': 0, 'Attacking Third': 0},
              'right': {'Defensive Third': 0, 'Middle Third': 0, 'Attacking Third': 0}}
    flip = {'Defensive Third': 'Attacking Third', 'Attacking Third': 'Defensive Third',
            'Middle Third': 'Middle Third'}

    for p in players:
        zone = classify_zone(p['x'], pitch_length)
        if p['team'] == 'right':
            zone = flip[zone]
        if p['team'] in counts:
            counts[p['team']][zone] += 1
    return counts
