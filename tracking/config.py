PX_PER_METER = 55.0
LABEL_SCALE  = 0.35
LABEL_THICK  = 1

BALL_COLOR = (0, 255, 255)

REID_MAX_FRAMES  = 120   # ~4 s à 30 fps
REID_MAX_DIST    = 150   # pixels : distance max position (prédite) pour accepter un match
REID_POS_WEIGHT  = 0.30  # poids position dans le coût
REID_APP_WEIGHT  = 0.70  # poids apparence dans le coût
REID_MAX_COST    = 0.32  # coût max pour accepter un réassignement (sinon nouveau joueur)
JERSEY_MAX_HIST  = 150   # cap sur l'historique couleur maillot

# Détection de switch interne du tracker (même tid → joueur différent)
TRACKER_SWITCH_MAX_DIST = 120  # pixels : saut max toléré pour un tid déjà connu
TRACKER_SWITCH_APP_MIN  = 0.30 # similarité apparence minimale pour garder un tid connu

POSSESSION_MAX_DIST = 80  # pixels : distance max pour attribuer la possession

JERSEY_MIN_SAMPLES  = 5
REF_UPDATE_INTERVAL = 15
SAME_TEAM_COLOR_DIST = 32.0

PASS_KICK_THRESH       = 12.0
PASS_ARRIVE_THRESH     = 8.0
BALL_PROX_PX           = 110
PASS_MAX_FLIGHT_FRAMES = 90   # ~3 s : timeout si la balle n'arrive pas

FIELD_MASK_REFRESH = 5
MIN_PLAYER_FRAMES  = 90    # frames minimum pour apparaître dans les stats (~3s à 30fps)

SUBST_MIN_FRAMES   = 150    # ~5s à 30fps — minimum pour une recommandation fiable
SUBST_SCORE_THRESH = 35.0   # score en dessous duquel on recommande un changement

# YOLO inference
YOLO_CONF_TRACK = 0.40   # confidence seuil tracker principal
YOLO_CONF_BALL  = 0.03   # confidence seuil modèle balle seule
YOLO_IMGSZ      = 1280
SLICE_MIN_CONF  = 0.12   # confidence seuil pour détections slice

# Formule de score [0-100] : vitesse(40) + passes(30) + distance(30)
SCORE_SPEED_DIV = 12.0   # km/h de référence pour la composante vitesse
