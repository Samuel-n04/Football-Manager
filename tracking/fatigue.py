"""
tracking/fatigue.py
===================
Calcul du Fatigue Score (0-100) pour chaque joueur.

Formule :
  fatigue = 25% distance_normalisée
           + 25% vitesse_max_normalisée
           + 25% intensité_haute (% frames en sprint)
           + 25% décélération_accumulée

Seuils :
  < 30  → Frais
  30-50 → Modéré
  50-70 → Fatigué
  > 70  → Épuisé (candidat substitution)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tracking.state import TrackingState

# ── Constantes ────────────────────────────────────────────────────────────────

# Distance maximale attendue sur une vidéo courte (mètres)
# Pour une vidéo de 3 min ≈ 300-400m de course pour un joueur actif
FATIGUE_DIST_MAX   = 400.0   # m  → 100 % contribution distance

# Vitesse de sprint (km/h) — au-delà = "effort intense"
SPRINT_THRESHOLD   = 20.0    # km/h

# Vitesse max normalisée (km/h) — joueur à 30 km/h = 100 % contribution
FATIGUE_SPEED_MAX  = 30.0    # km/h

# Fenêtre glissante pour détecter les déccélérations (frames)
ACCEL_WINDOW       = 10

# Poids de chaque composante (doivent sommer à 1)
W_DISTANCE  = 0.30
W_INTENSITY = 0.30
W_SPEED_MAX = 0.20
W_DECEL     = 0.20


@dataclass
class FatigueTracker:
    """État de fatigue temps-réel pour UN joueur."""
    sprint_frames:    int   = 0          # nombre de frames en sprint
    total_frames:     int   = 0          # frames totales détectées
    max_speed:        float = 0.0        # vitesse max observée (km/h)
    speed_window:     deque = field(default_factory=lambda: deque(maxlen=ACCEL_WINDOW))
    decel_accum:      float = 0.0        # décélérations accumulées (km/h/frame)

    # Score calculé à la dernière frame
    score:            float = 0.0        # 0-100
    label:            str   = "Frais"


def _fatigue_label(score: float) -> str:
    if score < 30:
        return "Frais"
    if score < 50:
        return "Modéré"
    if score < 70:
        return "Fatigué"
    return "Épuisé"


def update_fatigue(tracker: FatigueTracker, speed_kmh: float, distance_m: float) -> None:
    """
    Met à jour le tracker de fatigue à chaque frame où le joueur est visible.

    Parameters
    ----------
    tracker   : FatigueTracker du joueur
    speed_kmh : vitesse instantanée (km/h)
    distance_m: distance totale parcourue depuis le début (m)
    """
    tracker.total_frames += 1

    # Sprints
    if speed_kmh >= SPRINT_THRESHOLD:
        tracker.sprint_frames += 1

    # Vitesse max
    if speed_kmh > tracker.max_speed:
        tracker.max_speed = speed_kmh

    # Décélération accumulée (différence négative entre frames consécutives)
    tracker.speed_window.append(speed_kmh)
    if len(tracker.speed_window) >= 2:
        delta = tracker.speed_window[-1] - tracker.speed_window[-2]
        if delta < 0:                        # décélération
            tracker.decel_accum += abs(delta)

    # ── Calcul du score ───────────────────────────────────────────────────────
    comp_distance  = min(distance_m / FATIGUE_DIST_MAX, 1.0)

    intensity_ratio = (tracker.sprint_frames / tracker.total_frames
                       if tracker.total_frames > 0 else 0.0)
    comp_intensity = min(intensity_ratio * 3.0, 1.0)   # 33 % frames sprint → 100 %

    comp_speed_max = min(tracker.max_speed / FATIGUE_SPEED_MAX, 1.0)

    # Normaliser les décélérations accumulées (valeur typique après 3 min ≈ 500-1000)
    comp_decel = min(tracker.decel_accum / 800.0, 1.0)

    tracker.score = (
        W_DISTANCE  * comp_distance  +
        W_INTENSITY * comp_intensity +
        W_SPEED_MAX * comp_speed_max +
        W_DECEL     * comp_decel
    ) * 100.0

    tracker.label = _fatigue_label(tracker.score)


def compute_all_fatigue(state: "TrackingState") -> dict[int, dict]:
    """
    Calcule les scores de fatigue finaux pour tous les joueurs
    en partant de l'état de tracking complet.

    Retourne un dict  {pno: {"score": float, "label": str, ...}}
    """
    results = {}
    for pno, tracker in state.fatigue_trackers.items():
        if state.is_referee.get(pno):
            continue
        results[pno] = {
            "fatigue_score":    round(tracker.score, 1),
            "fatigue_label":    tracker.label,
            "sprint_frames":    tracker.sprint_frames,
            "total_frames_fat": tracker.total_frames,
            "sprint_pct":       round(
                tracker.sprint_frames / tracker.total_frames * 100
                if tracker.total_frames > 0 else 0.0, 1
            ),
            "max_speed_kmh":    round(tracker.max_speed, 2),
            "decel_accum":      round(tracker.decel_accum, 1),
        }
    return results