from collections import defaultdict
from tracking.fatigue import FatigueTracker


class TrackingState:
    def __init__(self):
        # Speed / camera
        self.prev_gray_small = None
        self.prev_centers    = {}
        self.speed_ema       = {}
        self.frame_count     = 0

        # ReID : mapping tid ↔ pno
        # player_state[pno] = {cx, cy, vx, vy, frame}   (vx/vy en px/frame)
        # player_appearances[pno] = vecteur HSV normalisé (EMA)
        self.player_by_tid    = {}   # tracker-id → player-no
        self.tid_last_seen    = {}   # tracker-id → dernière frame vue
        self.player_state     = {}   # pno → {cx, cy, vx, vy, frame}
        self.player_appearances = {} # pno → feature vector (np.ndarray, normalisé)
        self.next_player_no   = 1

        # Stats par joueur
        self.player_speeds    = defaultdict(list)
        self.passes_made      = defaultdict(int)
        self.passes_received  = defaultdict(int)
        self.passes_attempted = defaultdict(int)

        # Distance + Position
        self.player_distance    = defaultdict(float)  # pno → mètres parcourus (corrigé caméra)
        self.player_positions   = defaultdict(list)   # pno → [(cx, cy), ...] (sous-échantillonné)
        self.player_prev_pos    = {}                  # pno → (cx, cy) frame précédente (pixels bruts)
        self.player_frame_count = defaultdict(int)    # pno → nombre de frames de présence effective

        # Possession
        self.possession_frames = defaultdict(int)  # pno → nb de frames avec la balle
        self.player_team       = {}                # pno → 1 | 2 | None (arbitre/inconnu)

        # Couleurs maillot (passes + détection arbitres)
        self.jersey_colors = defaultdict(list)
        self.is_referee    = {}

        # Pass detection
        self.pass_state           = "idle"
        self.pass_from_pno        = None
        self.last_ball_spd        = 0.0
        self.pass_in_flight_since = None

        # Field mask
        self.field_mask_cache = None
        self.field_mask_frame = -1

        # Rendering
        self.player_colors: dict = {}  # pno → couleur BGR assignée à la première apparition

        # ── Semaine 2 — Fatigue trackers ──────────────────────────────────────
        # fatigue_trackers[pno] = FatigueTracker (mis à jour frame par frame)
        self.fatigue_trackers: dict = defaultdict(FatigueTracker)