import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

import sys
import cv2
import json
import numpy as np
import math
import torch
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

# ── Vidéo en argument ou valeur par défaut ────────────────────────────────────
VIDEO_PATH  = sys.argv[1] if len(sys.argv) > 1 else "match_extrait.mp4"
base        = VIDEO_PATH.rsplit(".", 1)[0]
OUTPUT_PATH = base + "_tracking.mp4"
STATS_PATH  = base + "_stats.json"

MODEL_PATH  = "yolov8m.pt"
TRACKER_CFG = "bytetrack.yaml"

# ── GPU ───────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda:0"
    print(f"GPU détecté : {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    print("Aucun GPU — utilisation CPU.")

PX_PER_METER = 55.0
LABEL_SCALE  = 0.35
LABEL_THICK  = 1

BALL_COLOR = (0, 255, 255)

# ── État tracking ─────────────────────────────────────────────────────────────
_prev_gray_small = None
_prev_centers    = {}
_speed_ema       = {}
_frame_count     = 0

# ── ReID : mapping tid ↔ pno ─────────────────────────────────────────────────
# _player_state[pno] = {cx, cy, vx, vy, frame}   (vx/vy en px/frame)
# _player_appearances[pno] = vecteur HSV normalisé (EMA)
_player_by_tid    = {}   # tracker-id → player-no
_tid_last_seen    = {}   # tracker-id → dernière frame vue
_player_state     = {}   # pno → {cx, cy, vx, vy, frame}
_player_appearances = {} # pno → feature vector (np.ndarray, normalisé)
_next_player_no   = 1

REID_MAX_FRAMES  = 120   # ~4 s à 30 fps
REID_MAX_DIST    = 150   # pixels : distance max position (prédite) pour accepter un match
REID_POS_WEIGHT  = 0.30  # poids position dans le coût
REID_APP_WEIGHT  = 0.70  # poids apparence dans le coût
REID_MAX_COST    = 0.40  # coût max pour accepter un réassignement (sinon nouveau joueur)
JERSEY_MAX_HIST  = 150   # cap sur l'historique couleur maillot

# Détection de switch interne du tracker (même tid → joueur différent)
TRACKER_SWITCH_MAX_DIST = 120  # pixels : saut max toléré pour un tid déjà connu
TRACKER_SWITCH_APP_MIN  = 0.30 # similarité apparence minimale pour garder un tid connu

# ── Stats par joueur ──────────────────────────────────────────────────────────
_player_speeds    = defaultdict(list)
_passes_made      = defaultdict(int)
_passes_received  = defaultdict(int)
_passes_attempted = defaultdict(int)

# ── Sprints ───────────────────────────────────────────────────────────────────
SPRINT_THRESH     = 25.0
SPRINT_END_THRESH = 20.0

_sprint_active = {}
_sprint_count  = defaultdict(int)
_sprint_frames = defaultdict(int)

# ── Distance + Fatigue + Position ─────────────────────────────────────────────
_player_distance    = defaultdict(float)   # pno → mètres parcourus (corrigé caméra)
_player_speed_hist  = defaultdict(list)    # pno → [(frame, speed_kmh), ...]
_player_positions   = defaultdict(list)    # pno → [(cx, cy), ...] (sous-échantillonné)
_player_prev_pos    = {}                   # pno → (cx, cy) frame précédente (pixels bruts)
_player_frame_count = defaultdict(int)     # pno → nombre de frames de présence effective

# ── Possession ────────────────────────────────────────────────────────────────
POSSESSION_MAX_DIST  = 80          # pixels : distance max pour attribuer la possession
_possession_frames   = defaultdict(int)   # pno → nb de frames avec la balle
_player_team         = {}          # pno → 1 | 2 | None (arbitre/inconnu)

# ── Couleurs maillot (passes + détection arbitres) ────────────────────────────
_jersey_colors      = defaultdict(list)
_is_referee         = {}

JERSEY_MIN_SAMPLES  = 5
REF_UPDATE_INTERVAL = 15

SAME_TEAM_COLOR_DIST = 32.0

# ── Détection de passes ───────────────────────────────────────────────────────
PASS_KICK_THRESH      = 12.0
PASS_ARRIVE_THRESH    = 8.0
BALL_PROX_PX          = 110
PASS_MAX_FLIGHT_FRAMES = 90   # ~3 s : timeout si la balle n'arrive pas

_pass_state           = "idle"
_pass_from_pno        = None
_last_ball_spd        = 0.0
_pass_in_flight_since = None

# ── Masque terrain ────────────────────────────────────────────────────────────
_field_mask_cache = None
_field_mask_frame = -1
FIELD_MASK_REFRESH  = 5
MIN_PLAYER_FRAMES   = 15   # frames minimum pour apparaître dans les stats


# ═══════════════════════════════════════════════════════════════════════════════
# Couleur joueur
# ═══════════════════════════════════════════════════════════════════════════════

_player_colors: dict = {}   # pno → couleur BGR assignée à la première apparition

def player_color(pno):
    if pno not in _player_colors:
        hue = (pno * 47) % 180
        hsv = np.array([[[hue, 220, 220]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        _player_colors[pno] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    return _player_colors[pno]


# ═══════════════════════════════════════════════════════════════════════════════
# ReID robuste
# ═══════════════════════════════════════════════════════════════════════════════

def get_appearance_feature(frame, xyxy):
    """
    Histogramme HSV 32 bins par région (maillot + short) → vecteur 128d normalisé L2.
    Séparer maillot et short permet de distinguer les coéquipiers portant le même jersey.
    """
    x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
    h_box = max(1, y2 - y1)
    w_box = max(1, x2 - x1)
    px1 = max(0, x1 + int(0.12 * w_box))
    px2 = max(0, x2 - int(0.12 * w_box))

    # Maillot : 15 %–52 % de la hauteur
    top = frame[y1 + int(0.15 * h_box) : y1 + int(0.52 * h_box), px1:px2]
    # Short  : 52 %–82 % de la hauteur
    bot = frame[y1 + int(0.52 * h_box) : y1 + int(0.82 * h_box), px1:px2]

    parts = []
    for crop in (top, bot):
        if crop.size == 0:
            parts.append(np.zeros(64, dtype=np.float32))
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
        h_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
        parts.append(np.concatenate([h_h, h_s]).astype(np.float32))

    feat = np.concatenate(parts)
    norm = np.linalg.norm(feat)
    return feat / norm if norm > 1e-6 else None


def _predicted_pos(pno):
    """Position prédite du joueur à la frame courante (extrapolation linéaire)."""
    st = _player_state.get(pno)
    if st is None:
        return None, None
    dt   = _frame_count - st['frame']
    pred_cx = st['cx'] + st.get('vx', 0.0) * dt
    pred_cy = st['cy'] + st.get('vy', 0.0) * dt
    return pred_cx, pred_cy


def _reid_cost(cx, cy, feat, pno):
    """
    Coût [0, +inf] de l'assignation (cx,cy,feat) → pno.
    Combine distance à la position prédite + distance d'apparence.
    """
    pred_cx, pred_cy = _predicted_pos(pno)
    if pred_cx is None:
        return float('inf')

    pos_dist = math.hypot(cx - pred_cx, cy - pred_cy)
    if pos_dist > REID_MAX_DIST:
        return float('inf')

    pos_cost = pos_dist / REID_MAX_DIST          # [0, 1]

    app_cost = 0.5                               # neutre si pas de feature
    prev_feat = _player_appearances.get(pno)
    if feat is not None and prev_feat is not None:
        sim      = float(np.dot(feat, prev_feat))    # cosine similarity ∈ [0,1]
        app_cost = max(0.0, 1.0 - sim)

    return REID_POS_WEIGHT * pos_cost + REID_APP_WEIGHT * app_cost


def _update_player_state(pno, cx, cy, feat):
    """Met à jour position, vitesse (EMA) et apparence d'un joueur."""
    prev = _player_state.get(pno)
    if prev is not None and _frame_count > prev['frame']:
        dt = _frame_count - prev['frame']
        new_vx = (cx - prev['cx']) / dt
        new_vy = (cy - prev['cy']) / dt
        alpha  = 0.4
        vx = alpha * new_vx + (1 - alpha) * prev.get('vx', 0.0)
        vy = alpha * new_vy + (1 - alpha) * prev.get('vy', 0.0)
    else:
        vx, vy = 0.0, 0.0

    _player_state[pno] = {'cx': cx, 'cy': cy, 'vx': vx, 'vy': vy, 'frame': _frame_count}

    if feat is not None:
        prev_feat = _player_appearances.get(pno)
        _player_appearances[pno] = (feat if prev_feat is None
                                    else 0.65 * prev_feat + 0.35 * feat)


def cleanup_old_tids():
    old = [tid for tid, f in _tid_last_seen.items()
           if _frame_count - f > REID_MAX_FRAMES]
    for tid in old:
        _player_by_tid.pop(tid, None)
        _tid_last_seen.pop(tid, None)


def batch_assign_players(person_dets, frame):
    """
    person_dets : list of (tid, cx, cy, xyxy)  — toutes les détections joueurs de la frame.
    Retourne    : dict tid → pno

    Algorithme :
      1. Tids déjà connus  → réassignation directe après vérification de cohérence
                             (détecte les switches internes du tracker ByteTrack).
      2. Tids nouveaux     → assignation optimale via Hungarian algorithm
                             (scipy.optimize.linear_sum_assignment).
      3. Sans correspondance → nouveau numéro de joueur.
    """
    global _next_player_no
    result = {}

    # ── 1. Tids connus — avec vérification anti-switch ────────────────────────
    for tid, cx, cy, xyxy in person_dets:
        if tid not in _player_by_tid:
            continue
        pno = _player_by_tid[tid]
        feat = get_appearance_feature(frame, xyxy)

        # Vérifier que le tid n'a pas sauté sur un autre joueur (tracker switch)
        pred_cx, pred_cy = _predicted_pos(pno)
        if pred_cx is not None:
            jump = math.hypot(cx - pred_cx, cy - pred_cy)
            if jump > TRACKER_SWITCH_MAX_DIST:
                # Saut de position trop grand → vérifier aussi l'apparence
                prev_feat = _player_appearances.get(pno)
                app_ok = (feat is None or prev_feat is None
                          or float(np.dot(feat, prev_feat)) >= TRACKER_SWITCH_APP_MIN)
                if not app_ok:
                    # Switch confirmé : libérer ce tid pour la réidentification
                    del _player_by_tid[tid]
                    continue

        _tid_last_seen[tid] = _frame_count
        _update_player_state(pno, cx, cy, feat)
        result[tid] = pno

    new_dets = [(tid, cx, cy, xyxy) for tid, cx, cy, xyxy in person_dets
                if tid not in result]
    if not new_dets:
        return result

    # ── 2. Joueurs inactifs candidats ─────────────────────────────────────────
    taken_pnos    = set(result.values())
    inactive_pnos = [
        pno for pno, st in _player_state.items()
        if pno not in taken_pnos
        and _frame_count - st['frame'] <= REID_MAX_FRAMES
    ]

    # ── 3. Précalcul features + matrice de coûts complète ────────────────────
    feat_cache = {tid: get_appearance_feature(frame, xyxy)
                  for tid, cx, cy, xyxy in new_dets}

    assigned_pnos = set()
    if inactive_pnos:
        n = len(new_dets)
        m = len(inactive_pnos)
        # Initialiser au-dessus du seuil (détections sans match valide)
        cost_matrix = np.full((n, m), REID_MAX_COST + 1.0, dtype=np.float64)

        for i, (tid, cx, cy, _) in enumerate(new_dets):
            for j, pno in enumerate(inactive_pnos):
                c = _reid_cost(cx, cy, feat_cache[tid], pno)
                if c < REID_MAX_COST:
                    cost_matrix[i, j] = c

        # ── 4. Assignation optimale (Hungarian) ──────────────────────────────
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] >= REID_MAX_COST:
                continue
            tid, cx, cy, xyxy = new_dets[i]
            pno = inactive_pnos[j]
            _player_by_tid[tid] = pno
            _tid_last_seen[tid]  = _frame_count
            _update_player_state(pno, cx, cy, feat_cache[tid])
            result[tid] = pno
            assigned_pnos.add(pno)

    # ── 5. Nouveaux joueurs sans correspondance ────────────────────────────────
    for tid, cx, cy, xyxy in new_dets:
        if tid in result:
            continue
        pno = _next_player_no
        _next_player_no += 1
        _player_by_tid[tid] = pno
        _tid_last_seen[tid]  = _frame_count
        _update_player_state(pno, cx, cy, feat_cache.get(tid))
        result[tid] = pno

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Détection terrain
# ═══════════════════════════════════════════════════════════════════════════════

def detect_field(frame):
    global _field_mask_cache, _field_mask_frame
    if (_field_mask_cache is not None
            and _frame_count - _field_mask_frame < FIELD_MASK_REFRESH):
        return _field_mask_cache

    h, w = frame.shape[:2]
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    green = cv2.inRange(hsv, np.array([30, 30, 30]), np.array([98, 255, 255]))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((9, 9),  np.uint8))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN,  np.ones((5, 5),  np.uint8))

    white          = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 50, 255]))
    green_expanded = cv2.dilate(green, np.ones((40, 40), np.uint8))
    white_on_field = cv2.bitwise_and(white, green_expanded)

    combined = cv2.bitwise_or(green, white_on_field)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h, w), dtype=np.uint8)
    if contours:
        cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    else:
        mask = cv2.dilate(green, np.ones((25, 25), np.uint8))

    _field_mask_cache = mask
    _field_mask_frame = _frame_count
    return mask


def on_field(xyxy, mask, h, w):
    x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
    cx   = max(0, min(w - 1, int((x1 + x2) / 2)))
    foot = min(h - 1, int(y2))
    py1  = max(0, foot - 5);  py2 = min(h - 1, foot + 8)
    px1  = max(0, cx   - 8);  px2 = min(w - 1, cx   + 8)
    patch = mask[py1:py2, px1:px2]
    return float((patch > 0).mean()) > 0.3 or mask[foot, cx] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Couleurs maillot / arbitres / passes
# ═══════════════════════════════════════════════════════════════════════════════

def get_jersey_color(frame, xyxy):
    x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
    h_box = max(1, y2 - y1);  w_box = max(1, x2 - x1)
    cy1 = y1 + int(0.18 * h_box);  cy2 = y1 + int(0.52 * h_box)
    cx1 = x1 + int(0.18 * w_box);  cx2 = x1 + int(0.82 * w_box)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    valid  = (~((h_ch >= 35) & (h_ch <= 85) & (s_ch >= 35))
              & ~(v_ch < 35) & ~(s_ch < 25))
    lab    = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    pixels = lab[valid] if valid.sum() >= 20 else lab.reshape(-1, 3)
    return np.median(pixels, axis=0)


def update_referee_detection():
    pnos = [p for p, cols in _jersey_colors.items() if len(cols) >= JERSEY_MIN_SAMPLES]
    if len(pnos) < 6:
        return
    features = np.array([np.median(_jersey_colors[p], axis=0) for p in pnos],
                        dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, _ = cv2.kmeans(features, 3, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labels   = labels.flatten()
    counts   = [int(np.sum(labels == i)) for i in range(3)]
    ref_cls  = int(np.argmin(counts))
    team_cls = [i for i in range(3) if i != ref_cls]

    # Le cluster arbitre ne doit pas dépasser 20 % des joueurs détectés :
    # au-delà, le k-means confond une équipe avec des arbitres.
    ref_valid = (counts[ref_cls] / max(sum(counts), 1)) <= 0.20

    for pno, lbl in zip(pnos, labels):
        lbl = int(lbl)
        if ref_valid and lbl == ref_cls:
            _is_referee[pno]  = True
            _player_team[pno] = None
        else:
            _is_referee[pno]  = False
            _player_team[pno] = 1 if lbl == team_cls[0] else 2


def median_jersey(pno):
    cols = _jersey_colors.get(pno)
    if not cols or len(cols) < JERSEY_MIN_SAMPLES:
        return None
    return np.median(cols, axis=0)


def same_team(pno_a, pno_b, players_pos):
    if _is_referee.get(pno_a) or _is_referee.get(pno_b):
        return False
    ca, cb = median_jersey(pno_a), median_jersey(pno_b)
    if ca is not None and cb is not None:
        dist = float(np.linalg.norm(ca - cb))
        if dist < SAME_TEAM_COLOR_DIST:
            return True
        if dist > SAME_TEAM_COLOR_DIST * 2:
            return False
    # Fallback spatial (gardiens)
    active = [(pno, cx, cy) for pno, (cx, cy) in players_pos.items()
              if not _is_referee.get(pno)]
    if len(active) < 4 or pno_a not in players_pos or pno_b not in players_pos:
        return True
    pts   = np.array([[cx, cy] for _, cx, cy in active], dtype=np.float32)
    pnos  = [pno for pno, _, _ in active]
    crit  = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, _ = cv2.kmeans(pts, 2, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    lmap  = {pno: int(l) for pno, l in zip(pnos, labels.flatten())}
    return lmap.get(pno_a) == lmap.get(pno_b)


def closest_player_to_ball(bcx, bcy, players):
    best_pno, best_d = None, float('inf')
    for pno, (pcx, pcy) in players.items():
        d = math.hypot(bcx - pcx, bcy - pcy)
        if d < best_d:
            best_d, best_pno = d, pno
    return best_pno, best_d


def update_pass_detection(ball_speed, bcx, bcy, players_this_frame):
    global _pass_state, _pass_from_pno, _last_ball_spd, _pass_in_flight_since
    near_pno, near_dist = closest_player_to_ball(bcx, bcy, players_this_frame)
    if _pass_state == "idle":
        if ball_speed >= PASS_KICK_THRESH and _last_ball_spd < PASS_KICK_THRESH:
            if near_dist <= BALL_PROX_PX and near_pno is not None:
                _pass_from_pno        = near_pno
                _pass_state           = "in_flight"
                _pass_in_flight_since = _frame_count
                _passes_attempted[near_pno] += 1
    elif _pass_state == "in_flight":
        # Timeout : si la balle n'arrive pas dans le délai imparti, annuler
        if _frame_count - (_pass_in_flight_since or _frame_count) > PASS_MAX_FLIGHT_FRAMES:
            _pass_state           = "idle"
            _pass_from_pno        = None
            _pass_in_flight_since = None
        elif ball_speed <= PASS_ARRIVE_THRESH:
            if near_dist <= BALL_PROX_PX and near_pno is not None and near_pno != _pass_from_pno:
                if same_team(_pass_from_pno, near_pno, players_this_frame):
                    _passes_made[_pass_from_pno] += 1
                    _passes_received[near_pno]   += 1
            _pass_state           = "idle"
            _pass_from_pno        = None
            _pass_in_flight_since = None
    _last_ball_spd = ball_speed


# ═══════════════════════════════════════════════════════════════════════════════
# Vitesse
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_cam_motion(gray, h, w):
    global _prev_gray_small
    small_w = 320
    scale   = small_w / max(float(w), 1.0)
    small_h = max(1, int(h * scale))
    small   = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA).astype(np.float32)
    small   = cv2.GaussianBlur(small, (5, 5), 0)
    dx, dy, resp = 0.0, 0.0, 0.0
    if _prev_gray_small is not None and _prev_gray_small.shape == small.shape:
        (dx_s, dy_s), r = cv2.phaseCorrelate(_prev_gray_small, small)
        dx, dy = float(dx_s) / max(scale, 1e-6), float(dy_s) / max(scale, 1e-6)
        resp   = float(r)
        if abs(dx) >= 0.20 * w or abs(dy) >= 0.20 * h:
            dx, dy = 0.0, 0.0
    _prev_gray_small = small
    return dx, dy, resp


def ema_smooth(key, val, alpha=0.4, max_jump=18.0):
    prev = _speed_ema.get(key)
    if prev is not None and val > prev + max_jump:
        val = prev + max_jump
    out = val if prev is None else alpha * val + (1.0 - alpha) * prev
    _speed_ema[key] = out
    return out


def compute_speed(key, cx, cy, cam_dx, cam_dy, cam_resp, fps, is_ball=False):
    t   = _frame_count / max(fps, 1.0)
    kmh = 0.0
    if key in _prev_centers:
        pc     = _prev_centers[key]
        dt     = max(t - pc['t'], 1.0 / fps)
        obj_dx = (cx - pc['x']) - cam_dx
        obj_dy = (cy - pc['y']) - cam_dy
        kmh    = (math.hypot(obj_dx, obj_dy) / dt / PX_PER_METER) * 3.6
        if cam_resp < 0.08:
            raw = (math.hypot(cx - pc['x'], cy - pc['y']) / dt / PX_PER_METER) * 3.6
            kmh = 0.65 * kmh + 0.35 * raw
    _prev_centers[key] = {'x': cx, 'y': cy, 't': t}
    alpha    = 0.45 if not is_ball else 0.55
    max_jump = 10.0  if not is_ball else 45.0
    return ema_smooth(key, max(0.0, kmh), alpha, max_jump)


# ═══════════════════════════════════════════════════════════════════════════════
# Rendu
# ═══════════════════════════════════════════════════════════════════════════════

def draw_label(frame, xyxy, label, color, box_thick=2):
    x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thick)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, LABEL_THICK)
    ty = max(y1 - 3, th + 4)
    cv2.rectangle(frame, (x1, ty - th - 2), (x1 + tw + 4, ty + 2), color, -1)
    lum     = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
    txt_col = (0, 0, 0) if lum > 140 else (255, 255, 255)
    cv2.putText(frame, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                LABEL_SCALE, txt_col, LABEL_THICK, cv2.LINE_AA)


def slice_detect(mdl, frame, slice_sz=640, overlap=0.25, conf=0.06):
    h, w  = frame.shape[:2]
    step  = int(slice_sz * (1.0 - overlap))
    boxes, confs, cls_ids = [], [], []
    for y in range(0, h, step):
        for x in range(0, w, step):
            tile = frame[y:y + slice_sz, x:x + slice_sz]
            res  = mdl.predict(tile, conf=conf, classes=[0, 32], verbose=False)
            for r in res:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                    boxes.append([x + bx1, y + by1, x + bx2, y + by2])
                    confs.append(float(b.conf[0]))
                    cls_ids.append(int(b.cls[0]))
    return boxes, confs, cls_ids


def iou(a, b):
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aa + ab - inter, 1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# Stats
# ═══════════════════════════════════════════════════════════════════════════════

def _fatigue_index(pno):
    """
    Indice de fatigue [0–100] basé sur le déclin de vitesse moyenne.
    Compare le premier quart de présence du joueur au dernier quart.
    Retourne None si pas assez de données (<40 frames).
    """
    hist = _player_speed_hist[pno]
    if len(hist) < 40:
        return None
    speeds = [s for _, s in hist]
    q = max(1, len(speeds) // 4)
    first_avg = sum(speeds[:q]) / q
    last_avg  = sum(speeds[-q:]) / q
    if first_avg < 1.0:   # joueur quasi-immobile → indice non pertinent
        return None
    return round(max(0.0, min(100.0, (1.0 - last_avg / first_avg) * 100.0)), 1)


def _estimate_poste(pno, avg_pos, std_x_all):
    """
    Estime le poste d'un joueur à partir de :
      - la variance de sa position x (faible variance → gardien)
      - son rang de position x au sein de son équipe (défenseur/milieu/attaquant)
    Retourne None si pas assez de données.
    """
    if _is_referee.get(pno):
        return "Arbitre"
    pos = _player_positions.get(pno, [])
    if len(pos) < 5:
        return None

    std_x = float(np.std([p[0] for p in pos]))
    # Seuil = percentile 15 de tous les joueurs (les plus statiques = gardiens)
    threshold = float(np.percentile(std_x_all, 15)) if len(std_x_all) >= 4 else 40.0
    threshold = max(threshold, 25.0)
    if std_x <= threshold:
        return "Gardien"

    team = _player_team.get(pno)
    if team is None or pno not in avg_pos:
        return None

    team_pnos = [p for p in avg_pos if _player_team.get(p) == team and not _is_referee.get(p)]
    if len(team_pnos) < 2:
        return None

    sorted_by_x = sorted(team_pnos, key=lambda p: avg_pos[p][0])
    try:
        rank = sorted_by_x.index(pno)
    except ValueError:
        return None
    n   = len(sorted_by_x)
    pct = rank / max(n - 1, 1)

    if pct < 0.30:
        return "Défenseur"
    elif pct < 0.65:
        return "Milieu"
    else:
        return "Attaquant"


def save_stats(video_path, stats_path, fps, frame_count):
    # Filtre : exclure les joueurs fantômes (présents < MIN_PLAYER_FRAMES frames)
    all_pnos = {pno for pno in (set(_player_speeds.keys())
                                | set(_passes_attempted.keys())
                                | set(_passes_received.keys())
                                | set(_possession_frames.keys()))
                if _player_frame_count.get(pno, 0) >= MIN_PLAYER_FRAMES}

    players  = {}
    arbitres = {}

    total_possession = max(sum(_possession_frames.values()), 1)
    duration_sec     = frame_count / max(fps, 1.0)

    # Pré-calcul position moyenne et std-x (pour estimation des postes)
    avg_pos = {}
    for pno in all_pnos:
        pos = _player_positions.get(pno, [])
        if pos:
            avg_pos[pno] = (float(np.mean([p[0] for p in pos])),
                            float(np.mean([p[1] for p in pos])))

    std_x_all = [float(np.std([p[0] for p in _player_positions[pno]]))
                 for pno in all_pnos
                 if len(_player_positions.get(pno, [])) >= 5
                 and not _is_referee.get(pno)]

    for pno in sorted(all_pnos):
        speeds      = _player_speeds[pno]
        attempted   = _passes_attempted[pno]
        made        = _passes_made[pno]
        rate        = round(made / attempted * 100, 1) if attempted > 0 else 0.0
        vitesse_moy = round(sum(speeds) / len(speeds), 2) if speeds else 0.0
        vitesse_max = round(max(speeds), 2) if speeds else 0.0
        poss_pct    = round(_possession_frames[pno] / total_possession * 100, 1)
        distance_m  = round(_player_distance[pno], 1)
        fatigue     = _fatigue_index(pno)
        poste       = _estimate_poste(pno, avg_pos, std_x_all)

        # Score final [0-100] : vitesse(40) + passes(30) + distance(30)
        expected_dist = max(duration_sec * 3.0, 1.0)   # distance attendue pour ce clip
        sp_pts  = min(vitesse_moy / 12.0, 1.0) * 40.0
        pp_pts  = (rate / 100.0 * 30.0) if attempted > 0 else 15.0
        dp_pts  = min(distance_m / expected_dist, 1.0) * 30.0
        score   = round(sp_pts + pp_pts + dp_pts, 1)
        underperf = score < 35

        if _is_referee.get(pno):
            arbitres[str(pno)] = {
                "vitesse_moyenne_kmh": vitesse_moy,
                "vitesse_max_kmh":     vitesse_max,
                "distance_m":          distance_m,
            }
        else:
            players[str(pno)] = {
                "player_id":            pno,
                "equipe":               _player_team.get(pno),
                "poste":                poste,
                "score":                score,
                "sous_performance":     underperf,
                "vitesse_moyenne_kmh":  vitesse_moy,
                "vitesse_max_kmh":      vitesse_max,
                "distance_m":           distance_m,
                "sprints":              _sprint_count[pno],
                "temps_sprint_sec":     round(_sprint_frames[pno] / max(fps, 1.0), 1),
                "passes_tentees":       attempted,
                "passes_reussies":      made,
                "passes_recues":        _passes_received[pno],
                "taux_reussite_passes": rate,
                "possession_pct":       poss_pct,
                "fatigue_index":        fatigue,
            }

    # Classement par score décroissant (rang 1 = meilleur)
    sorted_pnos = sorted(players, key=lambda k: players[k]["score"], reverse=True)
    for rank, k in enumerate(sorted_pnos, start=1):
        players[k]["classement"] = rank

    # Possession par équipe
    team_poss = defaultdict(int)
    for pno, frames in _possession_frames.items():
        team = _player_team.get(pno)
        if team:
            team_poss[team] += frames
    possession_equipes = {
        f"equipe_{t}": round(team_poss[t] / total_possession * 100, 1)
        for t in sorted(team_poss)
    }

    stats = {
        "video":              video_path,
        "duree_secondes":     round(duration_sec, 1),
        "frames_traitees":    frame_count,
        "joueurs":            players,
        "arbitres":           arbitres,
        "total_passes":       sum(_passes_made.values()),
        "possession_equipes": possession_equipes,
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n" + "═" * 110)
    print(f"  STATS — {video_path}  ({round(duration_sec, 1)}s)")
    print("═" * 110)
    print(f"  {'#':<4} {'Joueur':<8} {'Eq':>3} {'Poste':<12} {'Score':>6} {'Vit.moy':>8} "
          f"{'Vit.max':>8} {'Dist(m)':>8} {'Spr':>5} {'P.tent':>7} {'P.réus':>7} "
          f"{'Taux':>7} {'Poss.':>7} {'Fatigue':>8}")
    print("─" * 110)
    for k in sorted_pnos:
        p      = players[k]
        pno    = p["player_id"]
        eq     = f"E{p['equipe']}" if p['equipe'] else "  ?"
        poste  = (p['poste'] or "?")[:11]
        fat    = f"{p['fatigue_index']:>5.1f}%" if p['fatigue_index'] is not None else "    N/A"
        flag   = " !" if p['sous_performance'] else "  "
        print(f"  {p['classement']:<3}{flag} P{pno:<6} {eq:>3} {poste:<12} {p['score']:>5.1f}  "
              f"{p['vitesse_moyenne_kmh']:>7.1f}  {p['vitesse_max_kmh']:>7.1f}  "
              f"{p['distance_m']:>7.0f}  {p['sprints']:>4}  {p['passes_tentees']:>6}  "
              f"{p['passes_reussies']:>6}  {p['taux_reussite_passes']:>5.1f}%  "
              f"{p['possession_pct']:>5.1f}%  {fat}")
    if arbitres:
        print("─" * 110)
        print(f"  Arbitres : {', '.join(f'P{k}' for k in sorted(int(k) for k in arbitres))}")
    print("─" * 110)
    print(f"  Total passes réussies : {stats['total_passes']}")
    if possession_equipes:
        parts = "  /  ".join(f"Équipe {k[-1]} : {v}%" for k, v in possession_equipes.items())
        print(f"  Possession — {parts}")
    # Joueurs en sous-performance
    sp_list = [f"P{p['player_id']}" for p in players.values() if p['sous_performance']]
    if sp_list:
        print(f"  Sous-performance (!) : {', '.join(sp_list)}")
    print("═" * 110)
    print(f"\nStats sauvegardées : {stats_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

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

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        _frame_count += 1
        h, w = frame.shape[:2]

        cleanup_old_tids()
        if _frame_count % REF_UPDATE_INTERVAL == 0:
            update_referee_detection()

        field_mask               = detect_field(frame)
        gray                     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cam_dx, cam_dy, cam_resp = estimate_cam_motion(gray, h, w)

        results = model.track(
            source=frame,
            persist=True,
            tracker=TRACKER_CFG,
            conf=0.25,
            classes=[0, 32],
            verbose=False,
            imgsz=1280,
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
        tid_to_pno = batch_assign_players(person_dets, frame)

        # ── Traitement joueurs ────────────────────────────────────────────────
        for tid, cx, cy, xyxy in person_dets:
            pno   = tid_to_pno[tid]
            jcol  = get_jersey_color(frame, xyxy)
            if jcol is not None:
                _jersey_colors[pno].append(jcol)
                if len(_jersey_colors[pno]) > JERSEY_MAX_HIST:
                    _jersey_colors[pno] = _jersey_colors[pno][-JERSEY_MAX_HIST:]

            color = player_color(pno)
            speed = min(compute_speed(f'p_{pno}', cx, cy, cam_dx, cam_dy, cam_resp, fps), 38.0)
            _player_speeds[pno].append(speed)
            _player_frame_count[pno] += 1
            players_this_frame[pno] = (cx, cy)

            # Distance parcourue (déplacement corrigé mouvement caméra)
            prev_pos = _player_prev_pos.get(pno)
            if prev_pos is not None:
                _player_distance[pno] += (
                    math.hypot(cx - prev_pos[0] - cam_dx, cy - prev_pos[1] - cam_dy)
                    / PX_PER_METER
                )
            _player_prev_pos[pno] = (cx, cy)

            # Historique vitesse (calcul fatigue) + position terrain (sous-échantillonné)
            _player_speed_hist[pno].append((_frame_count, speed))
            if _frame_count % 5 == 0:
                _player_positions[pno].append((cx, cy))

            was_sprinting = _sprint_active.get(pno, False)
            threshold     = SPRINT_END_THRESH if was_sprinting else SPRINT_THRESH
            is_sprinting  = speed >= threshold
            if is_sprinting and not was_sprinting:
                _sprint_count[pno] += 1
            _sprint_active[pno] = is_sprinting
            if is_sprinting:
                _sprint_frames[pno] += 1

            # Score live [0-100] = vitesse(40) + passes(30) + distance(30)
            _att      = _passes_attempted[pno]
            _score    = (min(speed / 15.0, 1.0) * 40.0
                         + ((_passes_made[pno] / _att * 30.0) if _att > 0 else 15.0)
                         + min(_player_distance[pno] / 80.0, 1.0) * 30.0)

            underperf = (_score < 35 and _player_frame_count[pno] >= 50
                         and not _is_referee.get(pno))

            suffix      = " Arb" if _is_referee.get(pno) else (" SPR" if is_sprinting else "")
            perf_suffix = " !SP" if underperf else ""
            passes_lbl  = f" {_passes_made[pno]}/{_passes_attempted[pno]}p"
            # Couleur individuelle toujours conservée ; box plus épaisse si sous-perf ou sprint
            thickness   = 4 if underperf else (3 if is_sprinting else 2)
            draw_label(display, xyxy,
                       f'P{pno} {int(_score)}{suffix} {int(round(speed))}k{passes_lbl}{perf_suffix}',
                       color, thickness)

        # ── Traitement ballon ─────────────────────────────────────────────────
        if ball_dets:
            ball_found         = True
            tid, cx, cy, xyxy, _ = max(ball_dets, key=lambda b: b[4])
            key                = f'b_{tid}' if tid is not None else 'b_0'
            ball_speed         = min(compute_speed(key, cx, cy, cam_dx, cam_dy, cam_resp, fps, is_ball=True), 140.0)
            ball_cx, ball_cy   = cx, cy
            draw_label(display, xyxy, f'Balle {int(round(ball_speed))}km/h', BALL_COLOR)

        if not ball_found:
            ball_res = ball_model.predict(
                source=frame, conf=0.03, classes=[32],
                verbose=False, imgsz=1280, device=DEVICE,
            )
            if ball_res[0].boxes is not None and len(ball_res[0].boxes):
                best             = max(ball_res[0].boxes, key=lambda b: float(b.conf[0]))
                xyxy             = best.xyxy[0].tolist()
                x1, y1, x2, y2  = xyxy
                cx, cy           = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                ball_speed       = min(compute_speed('b_0', cx, cy, cam_dx, cam_dy, cam_resp, fps, is_ball=True), 140.0)
                ball_cx, ball_cy = cx, cy
                draw_label(display, xyxy, f'Balle {int(round(ball_speed))}km/h', BALL_COLOR)

        # ── Possession ───────────────────────────────────────────────────────
        if ball_cx is not None and players_this_frame:
            near_pno, near_dist = closest_player_to_ball(ball_cx, ball_cy, players_this_frame)
            if near_pno is not None and near_dist <= POSSESSION_MAX_DIST:
                _possession_frames[near_pno] += 1

        # Timeout passe en vol : déclencher même si la balle est absente ce frame
        if (_pass_state == "in_flight" and _pass_in_flight_since is not None
                and _frame_count - _pass_in_flight_since > PASS_MAX_FLIGHT_FRAMES):
            _pass_state           = "idle"
            _pass_from_pno        = None
            _pass_in_flight_since = None

        if ball_cx is not None and players_this_frame:
            update_pass_detection(ball_speed, ball_cx, ball_cy, players_this_frame)

        # ── Détection balle supplémentaire (slice) ────────────────────────────
        if _frame_count % 3 == 0:
            s_boxes, s_confs, s_cls = slice_detect(ball_model, frame)
            for sb, sc, scls in zip(s_boxes, s_confs, s_cls):
                if sc < 0.12:
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

        if _frame_count % 100 == 0:
            pct = _frame_count / max(total_frames, 1) * 100
            print(f"  {_frame_count}/{total_frames} frames ({pct:.1f}%)")

except KeyboardInterrupt:
    print(f"\nInterrompu à la frame {_frame_count}.")

finally:
    cap.release()
    writer.release()
    save_stats(VIDEO_PATH, STATS_PATH, fps, _frame_count)
    print(f"Vidéo sauvegardée  : {OUTPUT_PATH}")
