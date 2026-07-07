"""
Render the full annotated demo video: detection boxes + track IDs + team-colored
convex hulls + a live broadcast-style stats panel (team counts, zone occupancy
bar chart, ball-detection indicator, timestamp).

Usage (inside a notebook, with `best_model`, `track_results`, `smoothed_team`,
and `sample_clip` already defined -- see notebooks/tactics_cam_pipeline.ipynb):

    from src.render_demo_video import render_annotated_video
    render_annotated_video(sample_clip, best_model, track_results, smoothed_team,
                            out_path='outputs/demo_video.mp4')
"""
import cv2
import numpy as np

from src.pixel_tactical_analysis import (
    get_team_positions_pixel, pixel_convex_hull, classify_pixel_zone
)

TEAM_COLORS_BGR = {'A': (0, 220, 220), 'B': (255, 255, 255)}  # yellow, white
OTHER_COLOR = (0, 0, 255)  # red - GK / referee
BALL_COLOR = (0, 255, 0)


def _draw_hull(frame, points, color_bgr, alpha=0.15):
    hull_pts = pixel_convex_hull(points)
    if hull_pts is None:
        return frame
    hull_pts_int = hull_pts.astype(np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [hull_pts_int], color_bgr)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.polylines(frame, [hull_pts_int], isClosed=True, color=color_bgr, thickness=2)
    return frame


def _draw_info_panel(frame, positions, ball_detected, frame_idx, fps, frame_w, scale):
    panel_h = int(190 * scale)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame_w, panel_h), (15, 15, 15), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    title_scale, label_scale, small_scale = 1.1 * scale, 0.85 * scale, 0.65 * scale

    cv2.putText(frame, "TACTICS CAM  |  Live Tactical Analysis",
                (int(25 * scale), int(45 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                title_scale, (255, 255, 255), max(2, int(2 * scale)))

    t_sec = frame_idx / fps
    ts_text = f"t = {t_sec:.1f}s"
    (tw, _), _ = cv2.getTextSize(ts_text, cv2.FONT_HERSHEY_SIMPLEX, label_scale, 2)
    cv2.putText(frame, ts_text, (frame_w - tw - int(25 * scale), int(45 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, label_scale, (210, 210, 210), max(1, int(2 * scale)))

    y_legend = int(90 * scale)
    sw = int(26 * scale)
    x = int(25 * scale)

    for team, label in [('A', 'Team A'), ('B', 'Team B')]:
        cv2.rectangle(frame, (x, y_legend), (x + sw, y_legend + sw), TEAM_COLORS_BGR[team], -1)
        cv2.rectangle(frame, (x, y_legend), (x + sw, y_legend + sw), (255, 255, 255), 1)
        x += sw + int(12 * scale)
        text = f"{label}: {len(positions[team])}"
        cv2.putText(frame, text, (x, y_legend + sw - int(4 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255, 255, 255), max(1, int(2 * scale)))
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, label_scale, 2)
        x += tw + int(40 * scale)

    cv2.rectangle(frame, (x, y_legend), (x + sw, y_legend + sw), OTHER_COLOR, -1)
    cv2.rectangle(frame, (x, y_legend), (x + sw, y_legend + sw), (255, 255, 255), 1)
    x += sw + int(12 * scale)
    cv2.putText(frame, "GK / Referee", (x, y_legend + sw - int(4 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255, 255, 255), max(1, int(2 * scale)))

    ball_color = (0, 255, 0) if ball_detected else (110, 110, 110)
    ball_text = "BALL DETECTED" if ball_detected else "ball not visible"
    ball_font_scale = label_scale * (1.15 if ball_detected else 0.9)
    (tw, _), _ = cv2.getTextSize(ball_text, cv2.FONT_HERSHEY_SIMPLEX, ball_font_scale, 2)
    cv2.putText(frame, ball_text, (frame_w - tw - int(25 * scale), y_legend + sw - int(4 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, ball_font_scale, ball_color,
                max(2, int(2 * scale)) if ball_detected else 1)

    zone_labels = ["Left", "Mid", "Right"]
    bar_max_h, bar_w, bar_gap = int(65 * scale), int(30 * scale), int(4 * scale)
    group_gap, chart_x_start = int(50 * scale), int(25 * scale)
    chart_base_y = int(190 * scale) - int(10 * scale)

    zone_counts = {'A': [0, 0, 0], 'B': [0, 0, 0]}
    zone_names = ['Left Third', 'Middle Third', 'Right Third']
    for team in ['A', 'B']:
        for (px, _) in positions[team]:
            zone = classify_pixel_zone(px, frame_w)
            zone_counts[team][zone_names.index(zone)] += 1

    max_count = max(max(zone_counts['A']), max(zone_counts['B']), 1)

    cv2.putText(frame, "Zone Occupancy:",
                (chart_x_start, chart_base_y - bar_max_h - int(12 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, small_scale, (200, 200, 200), 1)

    for i in range(3):
        group_x = chart_x_start + i * group_gap
        for team, offset in [('A', 0), ('B', bar_w + bar_gap)]:
            bx = group_x + offset
            h = int((zone_counts[team][i] / max_count) * bar_max_h)
            cv2.rectangle(frame, (bx, chart_base_y - h), (bx + bar_w, chart_base_y),
                          TEAM_COLORS_BGR[team], -1)
            cv2.rectangle(frame, (bx, chart_base_y - h), (bx + bar_w, chart_base_y),
                          (255, 255, 255), 1)
            if zone_counts[team][i] > 0:
                cv2.putText(frame, str(zone_counts[team][i]),
                            (bx + int(6 * scale), chart_base_y - h - int(6 * scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, small_scale * 0.8, TEAM_COLORS_BGR[team], 1)
        cv2.putText(frame, zone_labels[i], (group_x, chart_base_y + int(20 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, small_scale, (220, 220, 220), 1)

    return frame


def render_annotated_video(source_clip, model, track_results, smoothed_team, out_path):
    """Render the full annotated + stats-panel demo video and write it to out_path."""
    cap = cv2.VideoCapture(source_clip)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    scale = frame_w / 1920
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_w, frame_h))

    cap = cv2.VideoCapture(source_clip)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = track_results[frame_idx] if frame_idx < len(track_results) else None
        positions = {'A': [], 'B': []}
        ball_detected = False

        if result is not None:
            if result.boxes.id is not None:
                positions = get_team_positions_pixel(result, model, smoothed_team)
                for team in ['A', 'B']:
                    frame = _draw_hull(frame, positions[team], TEAM_COLORS_BGR[team])

                for box, track_id, cls in zip(result.boxes.xyxy, result.boxes.id, result.boxes.cls):
                    cls_name = model.names[int(cls)]
                    tid = int(track_id)
                    x1, y1, x2, y2 = [int(v) for v in box.tolist()]

                    if cls_name == 'player' and tid in smoothed_team:
                        team = smoothed_team[tid]['team']
                        color, label = TEAM_COLORS_BGR[team], f"{team} #{tid}"
                    elif cls_name == 'ball':
                        color, label = BALL_COLOR, "ball"
                    else:
                        color, label = OTHER_COLOR, cls_name

                    thickness = max(2, int(2 * scale))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    cv2.putText(frame, label, (x1, max(y1 - int(6 * scale), int(10 * scale))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, color, thickness)

            for cls in result.boxes.cls:
                if model.names[int(cls)] == 'ball':
                    ball_detected = True
                    break

        frame = _draw_info_panel(frame, positions, ball_detected, frame_idx, fps, frame_w, scale)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Saved annotated video to {out_path} ({frame_idx}/{total} frames)")
