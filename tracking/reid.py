import math
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from tracking.config import (
    REID_MAX_FRAMES, REID_MAX_DIST, REID_POS_WEIGHT, REID_APP_WEIGHT, REID_MAX_COST,
    TRACKER_SWITCH_MAX_DIST, TRACKER_SWITCH_APP_MIN,
)


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


def predicted_pos(state, pno):
    """Position prédite du joueur à la frame courante (extrapolation linéaire)."""
    st = state.player_state.get(pno)
    if st is None:
        return None, None
    dt      = state.frame_count - st['frame']
    pred_cx = st['cx'] + st.get('vx', 0.0) * dt
    pred_cy = st['cy'] + st.get('vy', 0.0) * dt
    return pred_cx, pred_cy


def reid_cost(state, cx, cy, feat, pno):
    """
    Coût [0, +inf] de l'assignation (cx,cy,feat) → pno.
    Combine distance à la position prédite + distance d'apparence.
    """
    pred_cx, pred_cy = predicted_pos(state, pno)
    if pred_cx is None:
        return float('inf')

    pos_dist = math.hypot(cx - pred_cx, cy - pred_cy)
    if pos_dist > REID_MAX_DIST:
        return float('inf')

    pos_cost = pos_dist / REID_MAX_DIST          # [0, 1]

    app_cost = 0.5                               # neutre si pas de feature
    prev_feat = state.player_appearances.get(pno)
    if feat is not None and prev_feat is not None:
        sim      = float(np.dot(feat, prev_feat))    # cosine similarity ∈ [0,1]
        app_cost = max(0.0, 1.0 - sim)

    return REID_POS_WEIGHT * pos_cost + REID_APP_WEIGHT * app_cost


def update_player_state(state, pno, cx, cy, feat):
    """Met à jour position, vitesse (EMA) et apparence d'un joueur."""
    prev = state.player_state.get(pno)
    if prev is not None and state.frame_count > prev['frame']:
        dt     = state.frame_count - prev['frame']
        new_vx = (cx - prev['cx']) / dt
        new_vy = (cy - prev['cy']) / dt
        alpha  = 0.4
        vx = alpha * new_vx + (1 - alpha) * prev.get('vx', 0.0)
        vy = alpha * new_vy + (1 - alpha) * prev.get('vy', 0.0)
    else:
        vx, vy = 0.0, 0.0

    state.player_state[pno] = {'cx': cx, 'cy': cy, 'vx': vx, 'vy': vy, 'frame': state.frame_count}

    if feat is not None:
        prev_feat = state.player_appearances.get(pno)
        state.player_appearances[pno] = (feat if prev_feat is None
                                         else 0.65 * prev_feat + 0.35 * feat)


def cleanup_old_tids(state):
    old = [tid for tid, f in state.tid_last_seen.items()
           if state.frame_count - f > REID_MAX_FRAMES]
    for tid in old:
        state.player_by_tid.pop(tid, None)
        state.tid_last_seen.pop(tid, None)


def batch_assign_players(state, person_dets, frame):
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
    result = {}

    # ── 1. Tids connus — avec vérification anti-switch ────────────────────────
    for tid, cx, cy, xyxy in person_dets:
        if tid not in state.player_by_tid:
            continue
        pno  = state.player_by_tid[tid]
        feat = get_appearance_feature(frame, xyxy)

        # Vérifier que le tid n'a pas sauté sur un autre joueur (tracker switch)
        pred_cx, pred_cy = predicted_pos(state, pno)
        if pred_cx is not None:
            jump = math.hypot(cx - pred_cx, cy - pred_cy)
            if jump > TRACKER_SWITCH_MAX_DIST:
                # Saut de position trop grand → vérifier aussi l'apparence
                prev_feat = state.player_appearances.get(pno)
                app_ok = (feat is None or prev_feat is None
                          or float(np.dot(feat, prev_feat)) >= TRACKER_SWITCH_APP_MIN)
                if not app_ok:
                    # Switch confirmé : libérer ce tid pour la réidentification
                    del state.player_by_tid[tid]
                    continue

        state.tid_last_seen[tid] = state.frame_count
        update_player_state(state, pno, cx, cy, feat)
        result[tid] = pno

    new_dets = [(tid, cx, cy, xyxy) for tid, cx, cy, xyxy in person_dets
                if tid not in result]
    if not new_dets:
        return result

    # ── 2. Joueurs inactifs candidats ─────────────────────────────────────────
    taken_pnos    = set(result.values())
    inactive_pnos = [
        pno for pno, st in state.player_state.items()
        if pno not in taken_pnos
        and state.frame_count - st['frame'] <= REID_MAX_FRAMES
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
                c = reid_cost(state, cx, cy, feat_cache[tid], pno)
                if c < REID_MAX_COST:
                    cost_matrix[i, j] = c

        # ── 4. Assignation optimale (Hungarian) ──────────────────────────────
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] >= REID_MAX_COST:
                continue
            tid, cx, cy, xyxy = new_dets[i]
            pno = inactive_pnos[j]
            state.player_by_tid[tid] = pno
            state.tid_last_seen[tid] = state.frame_count
            update_player_state(state, pno, cx, cy, feat_cache[tid])
            result[tid] = pno
            assigned_pnos.add(pno)

    # ── 5. Nouveaux joueurs sans correspondance ────────────────────────────────
    for tid, cx, cy, xyxy in new_dets:
        if tid in result:
            continue
        pno = state.next_player_no
        state.next_player_no += 1
        state.player_by_tid[tid] = pno
        state.tid_last_seen[tid] = state.frame_count
        update_player_state(state, pno, cx, cy, feat_cache.get(tid))
        result[tid] = pno

    return result
