import math
import numpy as np
import cv2

from tracking.config import (
    JERSEY_MIN_SAMPLES, SAME_TEAM_COLOR_DIST,
)


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


def update_referee_detection(state):
    pnos = [p for p, cols in state.jersey_colors.items() if len(cols) >= JERSEY_MIN_SAMPLES]
    if len(pnos) < 6:
        return
    features = np.array([np.median(state.jersey_colors[p], axis=0) for p in pnos],
                        dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)

    # ── Essai k=3 (équipe1 + équipe2 + arbitre) ──────────────────────────────
    _, labels3, centers3 = cv2.kmeans(features, 3, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labels3  = labels3.flatten()
    counts3  = [int(np.sum(labels3 == i)) for i in range(3)]
    ref_cls  = int(np.argmin(counts3))
    team_cls = [i for i in range(3) if i != ref_cls]
    ref_valid = (counts3[ref_cls] / max(sum(counts3), 1)) <= 0.20

    n1 = counts3[team_cls[0]]
    n2 = counts3[team_cls[1]]

    # ── Fallback k=2 si les équipes sont trop déséquilibrées (ratio > 3:1) ──
    if max(n1, n2) / max(min(n1, n2), 1) > 3.0:
        _, labels2, centers2 = cv2.kmeans(features, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        labels2 = labels2.flatten()
        # Stabiliser : cluster le plus sombre (L* bas) → équipe 1
        if centers2[0][0] > centers2[1][0]:
            labels2 = 1 - labels2
        for pno, lbl in zip(pnos, labels2):
            state.is_referee[pno]  = False
            state.player_team[pno] = int(lbl) + 1
        return

    # ── Stabiliser l'ordre équipe1/équipe2 par luminosité L* ─────────────────
    # Cluster le plus sombre (L* le plus faible) → équipe 1
    if centers3[team_cls[0]][0] > centers3[team_cls[1]][0]:
        team_cls = team_cls[::-1]

    for pno, lbl in zip(pnos, labels3):
        lbl = int(lbl)
        if ref_valid and lbl == ref_cls:
            state.is_referee[pno]  = True
            state.player_team[pno] = None
        else:
            state.is_referee[pno]  = False
            state.player_team[pno] = 1 if lbl == team_cls[0] else 2


def median_jersey(state, pno):
    cols = state.jersey_colors.get(pno)
    if not cols or len(cols) < JERSEY_MIN_SAMPLES:
        return None
    return np.median(cols, axis=0)


def same_team(state, pno_a, pno_b, players_pos):
    if state.is_referee.get(pno_a) or state.is_referee.get(pno_b):
        return False
    ca, cb = median_jersey(state, pno_a), median_jersey(state, pno_b)
    if ca is not None and cb is not None:
        dist = float(np.linalg.norm(ca - cb))
        if dist < SAME_TEAM_COLOR_DIST:
            return True
        if dist > SAME_TEAM_COLOR_DIST * 2:
            return False
    # Fallback spatial (gardiens)
    active = [(pno, cx, cy) for pno, (cx, cy) in players_pos.items()
              if not state.is_referee.get(pno)]
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


def player_color(state, pno):
    if pno not in state.player_colors:
        hue = (pno * 47) % 180
        hsv = np.array([[[hue, 220, 220]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        state.player_colors[pno] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    return state.player_colors[pno]
