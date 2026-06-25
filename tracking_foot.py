import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

import signal
import sys
import cv2
import math
import torch

# Converts SIGTERM (sent by proc.terminate()) into KeyboardInterrupt
# so the finally block always runs and partial stats are saved.
signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
from ultralytics import YOLO

from tracking.config import (
    BALL_COLOR, JERSEY_MAX_HIST, POSSESSION_MAX_DIST,
    PASS_MAX_FLIGHT_FRAMES, REF_UPDATE_INTERVAL, PX_PER_METER,
    YOLO_CONF_TRACK, YOLO_CONF_BALL, YOLO_IMGSZ, SLICE_MIN_CONF, SCORE_SPEED_DIV,
)
from tracking.state import TrackingState
from tracking.reid import batch_assign_players, cleanup_old_tids
from tracking.field import detect_field, on_field
from tracking.team import (
    get_jersey_color, update_referee_detection,
    closest_player_to_ball, player_color,
)
from tracking.passes import update_pass_detection
from tracking.speed import estimate_cam_motion, compute_speed
from tracking.stats import save_stats
from tracking.render import draw_label, slice_detect, iou

# ── Vidéo en argument ou valeur par défaut ────────────────────────────────────
VIDEO_PATH  = sys.argv[1] if len(sys.argv) > 1 else os.path.join("input_videos", "match_extrait.mp4")
_base_name  = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_PATH  = os.path.join("output_videos", _base_name + "_tracking.mp4")
STATS_PATH   = os.path.join("output_videos", _base_name + "_stats.json")
PREVIEW_PATH = os.path.join("output_videos", _base_name + "_preview.jpg")

os.makedirs("output_videos", exist_ok=True)

MODEL_PATH  = "yolov8m.pt"
TRACKER_CFG = "bytetrack.yaml"

# ── GPU ───────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda:0"
    print(f"GPU détecté : {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    print("Aucun GPU — utilisation CPU.")

model      = YOLO(MODEL_PATH)   # tracking principal — ne jamais appeler .predict() dessus
ball_model = YOLO(MODEL_PATH)   # détection balle seule — jamais .track()
model.to(DEVICE)
ball_model.to(DEVICE)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Erreur: impossible d'ouvrir {VIDEO_PATH}")
    sys.exit(1)

fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

print(f"Entrée  : {VIDEO_PATH}")
print(f"Sortie  : {OUTPUT_PATH}")
print(f"Stats   : {STATS_PATH}")
print(f"Vidéo   : {total_frames} frames @ {fps:.1f}fps")

state = TrackingState()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        state.frame_count += 1
        h, w = frame.shape[:2]

        cleanup_old_tids(state)
        if state.frame_count % REF_UPDATE_INTERVAL == 0:
            update_referee_detection(state)

        field_mask               = detect_field(state, frame)
        gray                     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cam_dx, cam_dy, cam_resp = estimate_cam_motion(state, gray, h, w)

        results = model.track(
            source=frame,
            persist=True,
            tracker=TRACKER_CFG,
            conf=YOLO_CONF_TRACK,
            classes=[0, 32],
            verbose=False,
            imgsz=YOLO_IMGSZ,
            device=DEVICE,
        )

        display            = frame.copy()
        tracked_boxes      = []
        ball_found         = False
        players_this_frame = {}
        ball_cx = ball_cy  = None
        ball_speed         = 0.0

        # ── Collecter personnes + balles ──────────────────────────────────────
        person_dets = []   # (tid, cx, cy, xyxy)
        ball_dets   = []   # (tid, cx, cy, xyxy, conf)

        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                tid    = int(box.id[0]) if box.id is not None else None
                xyxy   = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                tracked_boxes.append(xyxy)
                if cls_id == 0 and tid is not None and on_field(xyxy, field_mask, h, w):
                    person_dets.append((tid, cx, cy, xyxy))
                elif cls_id == 32:
                    ball_dets.append((tid, cx, cy, xyxy, float(box.conf[0])))

        # ── Assignation robuste en batch ──────────────────────────────────────
        tid_to_pno = batch_assign_players(state, person_dets, frame)

        # ── Traitement joueurs ────────────────────────────────────────────────
        for tid, cx, cy, xyxy in person_dets:
            pno   = tid_to_pno[tid]
            jcol  = get_jersey_color(frame, xyxy)
            if jcol is not None:
                state.jersey_colors[pno].append(jcol)
                if len(state.jersey_colors[pno]) > JERSEY_MAX_HIST:
                    state.jersey_colors[pno] = state.jersey_colors[pno][-JERSEY_MAX_HIST:]

            color = player_color(state, pno)
            speed = min(compute_speed(state, f'p_{pno}', cx, cy, cam_dx, cam_dy, cam_resp, fps), 38.0)
            state.player_speeds[pno].append(speed)
            state.player_frame_count[pno] += 1
            players_this_frame[pno] = (cx, cy)

            # Distance parcourue (déplacement corrigé mouvement caméra)
            prev_pos = state.player_prev_pos.get(pno)
            if prev_pos is not None:
                state.player_distance[pno] += (
                    math.hypot(cx - prev_pos[0] - cam_dx, cy - prev_pos[1] - cam_dy)
                    / PX_PER_METER
                )
            state.player_prev_pos[pno] = (cx, cy)

            # Historique position terrain (sous-échantillonné)
            if state.frame_count % 5 == 0:
                state.player_positions[pno].append((cx, cy))

            # Score live [0-100] = vitesse(40) + passes(30) + distance(30)
            _att   = state.passes_attempted[pno]
            _score = (min(speed / SCORE_SPEED_DIV, 1.0) * 40.0
                      + ((state.passes_made[pno] / _att * 30.0) if _att > 0 else 15.0)
                      + min(state.player_distance[pno] / 80.0, 1.0) * 30.0)

            underperf = (_score < 35 and state.player_frame_count[pno] >= 50
                         and not state.is_referee.get(pno))

            suffix      = " Arb" if state.is_referee.get(pno) else ""
            perf_suffix = " !SP" if underperf else ""
            passes_lbl  = f" {state.passes_made[pno]}/{state.passes_attempted[pno]}p"
            thickness   = 4 if underperf else 2
            draw_label(display, xyxy,
                       f'P{pno} {int(_score)}{suffix} {int(round(speed))}k{passes_lbl}{perf_suffix}',
                       color, thickness)

        # ── Traitement ballon ─────────────────────────────────────────────────
        if ball_dets:
            ball_found         = True
            tid, cx, cy, xyxy, _ = max(ball_dets, key=lambda b: b[4])
            key                = f'b_{tid}' if tid is not None else 'b_0'
            ball_speed         = min(compute_speed(state, key, cx, cy, cam_dx, cam_dy, cam_resp, fps, is_ball=True), 140.0)
            ball_cx, ball_cy   = cx, cy
            draw_label(display, xyxy, f'Balle {int(round(ball_speed))}km/h', BALL_COLOR)

        if not ball_found:
            ball_res = ball_model.predict(
                source=frame, conf=YOLO_CONF_BALL, classes=[32],
                verbose=False, imgsz=YOLO_IMGSZ, device=DEVICE,
            )
            if ball_res[0].boxes is not None and len(ball_res[0].boxes):
                best             = max(ball_res[0].boxes, key=lambda b: float(b.conf[0]))
                xyxy             = best.xyxy[0].tolist()
                x1, y1, x2, y2  = xyxy
                cx, cy           = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                ball_speed       = min(compute_speed(state, 'b_0', cx, cy, cam_dx, cam_dy, cam_resp, fps, is_ball=True), 140.0)
                ball_cx, ball_cy = cx, cy
                draw_label(display, xyxy, f'Balle {int(round(ball_speed))}km/h', BALL_COLOR)

        # ── Possession ───────────────────────────────────────────────────────
        if ball_cx is not None and players_this_frame:
            near_pno, near_dist = closest_player_to_ball(ball_cx, ball_cy, players_this_frame)
            if near_pno is not None and near_dist <= POSSESSION_MAX_DIST:
                state.possession_frames[near_pno] += 1

        # Timeout passe en vol : déclencher même si la balle est absente ce frame
        if (state.pass_state == "in_flight" and state.pass_in_flight_since is not None
                and state.frame_count - state.pass_in_flight_since > PASS_MAX_FLIGHT_FRAMES):
            state.pass_state           = "idle"
            state.pass_from_pno        = None
            state.pass_in_flight_since = None

        if ball_cx is not None and players_this_frame:
            update_pass_detection(state, ball_speed, ball_cx, ball_cy, players_this_frame)

        # ── Détection balle supplémentaire (slice) ────────────────────────────
        if state.frame_count % 3 == 0:
            s_boxes, s_confs, s_cls = slice_detect(ball_model, frame)
            for sb, sc, scls in zip(s_boxes, s_confs, s_cls):
                if sc < SLICE_MIN_CONF:
                    continue
                if any(iou(sb, tb) > 0.3 for tb in tracked_boxes):
                    continue
                if scls == 0 and not on_field(sb, field_mask, h, w):
                    continue
                x1, y1, x2, y2 = [int(round(v)) for v in sb]
                col = (180, 180, 180) if scls == 0 else BALL_COLOR
                for i in range(x1, x2, 10):
                    cv2.line(display, (i, y1), (min(i + 5, x2), y1), col, 1)
                    cv2.line(display, (i, y2), (min(i + 5, x2), y2), col, 1)
                for i in range(y1, y2, 10):
                    cv2.line(display, (x1, i), (x1, min(i + 5, y2)), col, 1)
                    cv2.line(display, (x2, i), (x2, min(i + 5, y2)), col, 1)

        writer.write(display)

        preview     = cv2.resize(display, (display.shape[1] // 2, display.shape[0] // 2))
        ok, buf     = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if not ok:
            ok, buf = cv2.imencode('.png', preview)
        if ok:
            tmp_path = PREVIEW_PATH + ".tmp"
            with open(tmp_path, 'wb') as _f:
                _f.write(buf.tobytes())
            os.replace(tmp_path, PREVIEW_PATH)  # atomic

        pct = state.frame_count / max(total_frames, 1) * 100
        print(f"  {state.frame_count}/{total_frames} frames ({pct:.1f}%)")

except KeyboardInterrupt:
    print(f"\nInterrompu à la frame {state.frame_count}.")

finally:
    cap.release()
    writer.release()
    if os.path.exists(PREVIEW_PATH):
        os.remove(PREVIEW_PATH)
    save_stats(state, VIDEO_PATH, STATS_PATH, fps, state.frame_count)

    # Re-encode to H.264 so browsers can play the video
    import subprocess as _sp, os as _os
    h264_path = OUTPUT_PATH.replace(".mp4", "_h264.mp4")
    ret = _sp.run(
        ["ffmpeg", "-y", "-i", OUTPUT_PATH,
         "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
         "-movflags", "+faststart", h264_path],
        capture_output=True,
    )
    if ret.returncode == 0:
        _os.replace(h264_path, OUTPUT_PATH)
        print(f"Vidéo sauvegardée  : {OUTPUT_PATH}")
    else:
        print(f"Vidéo sauvegardée  : {OUTPUT_PATH} (ré-encodage H.264 échoué)")
