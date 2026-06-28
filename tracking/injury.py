"""
tracking/ml_model.py
====================
Semaine 2 — Tâches 2.5 & 2.6

• Tâche 2.5 : Random Forest + XGBoost → prédiction niveau risque blessure
              (Faible / Modéré / Élevé)

• Tâche 2.6 : MLP simple (Deep Learning) → prédiction performance future
              à partir des 5 dernières "fenêtres" de stats du joueur

Fonctionnement :
  - Un dataset synthétique est généré à partir des stats réelles du match
    (augmentées avec bruit gaussien) pour avoir assez d'exemples.
  - Les modèles sont entraînés en mémoire et leurs résultats sauvegardés
    dans le JSON de stats.
  - Une comparaison ML vs DL est produite (accuracy, temps d'entraînement).

Dépendances :
  scikit-learn (déjà installé via ultralytics)
  numpy (idem)
  Les imports XGBoost et torch sont protégés (optional).
"""

from __future__ import annotations

import time
import numpy as np
from typing import Any

# ── Imports optionnels ────────────────────────────────────────────────────────

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

# ── Labels injury ─────────────────────────────────────────────────────────────
INJURY_LABELS    = ["Faible", "Modéré", "Élevé"]
INJURY_LABEL_MAP = {l: i for i, l in enumerate(INJURY_LABELS)}


# ══════════════════════════════════════════════════════════════════════════════
#  Génération du dataset synthétique
# ══════════════════════════════════════════════════════════════════════════════

def _augment_player(stats: dict, n_aug: int = 20, seed: int = 0) -> list[dict]:
    """
    Crée n_aug variantes bruitées d'un enregistrement joueur.
    Utilisé pour grossir le dataset quand on n'a que peu de joueurs.
    """
    rng   = np.random.default_rng(seed)
    rows  = []
    for _ in range(n_aug):
        noise = rng.normal(0, 0.08)   # ±8 % bruit
        rows.append({
            "vitesse_moyenne":  max(0, stats["vitesse_moyenne_kmh"] * (1 + noise)),
            "distance":         max(0, stats["distance_m"]          * (1 + noise)),
            "sprint_pct":       max(0, min(100, stats.get("sprint_pct", 0)  * (1 + noise))),
            "fatigue_score":    max(0, min(100, stats.get("fatigue_score", 50) + rng.normal(0, 5))),
            "decel_accum":      max(0, stats.get("decel_accum", 0)  * (1 + noise)),
            "play_time_s":      max(0, stats.get("play_time_s", 60) * (1 + noise)),
            "performance_score":max(0, min(100, stats["score"]       * (1 + noise))),
            "injury_label":     stats.get("injury_label", "Faible"),
        })
    return rows


def build_dataset(all_player_stats: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construit X_injury, y_injury (pour RF/XGB)
    et X_perf, y_perf (pour MLP performance future).

    Retourne (X_injury, y_injury, X_perf, y_perf)
    """
    rows = []
    for p in all_player_stats:
        rows.extend(_augment_player(p))

    features_injury = []
    labels_injury   = []
    features_perf   = []
    labels_perf     = []

    for r in rows:
        feat = [
            r["vitesse_moyenne"],
            r["distance"],
            r["sprint_pct"],
            r["fatigue_score"],
            r["decel_accum"],
            r["play_time_s"],
        ]
        features_injury.append(feat)
        labels_injury.append(INJURY_LABEL_MAP.get(r["injury_label"], 0))

        # Pour la performance future : on cible le score actuel
        # (simulation d'une prédiction sur base des indicateurs physiques)
        features_perf.append(feat)
        labels_perf.append(r["performance_score"])

    X_inj  = np.array(features_injury, dtype=np.float32)
    y_inj  = np.array(labels_injury,   dtype=np.int64)
    X_perf = np.array(features_perf,   dtype=np.float32)
    y_perf = np.array(labels_perf,     dtype=np.float32)

    return X_inj, y_inj, X_perf, y_perf


# ══════════════════════════════════════════════════════════════════════════════
#  Tâche 2.5 — Random Forest + XGBoost
# ══════════════════════════════════════════════════════════════════════════════

def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Entraîne un Random Forest et retourne métriques + prédictions."""
    if not SKLEARN_OK:
        return {"error": "scikit-learn non disponible"}

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)

    t0  = time.perf_counter()
    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=seed, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    train_time = time.perf_counter() - t0

    y_pred   = clf.predict(X_te)
    accuracy = accuracy_score(y_te, y_pred)
    report   = classification_report(y_te, y_pred, target_names=INJURY_LABELS, output_dict=True, zero_division=0)

    importances = dict(zip(
        ["vitesse_moy", "distance", "sprint_pct", "fatigue", "decel_accum", "play_time"],
        clf.feature_importances_.round(4).tolist()
    ))

    return {
        "model":        "Random Forest",
        "accuracy":     round(accuracy, 4),
        "train_time_s": round(train_time, 4),
        "report":       report,
        "feature_importance": importances,
        "model_obj":    (clf, scaler),   # gardé en mémoire pour prédictions
    }


def train_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Entraîne XGBoost et retourne métriques."""
    if not SKLEARN_OK:
        return {"error": "scikit-learn non disponible"}
    if not XGB_OK:
        return {"error": "xgboost non installé — pip install xgboost"}

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)

    t0  = time.perf_counter()
    clf = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=seed, verbosity=0,
    )
    clf.fit(X_tr, y_tr)
    train_time = time.perf_counter() - t0

    y_pred   = clf.predict(X_te)
    accuracy = accuracy_score(y_te, y_pred)
    report   = classification_report(y_te, y_pred, target_names=INJURY_LABELS, output_dict=True, zero_division=0)

    return {
        "model":        "XGBoost",
        "accuracy":     round(accuracy, 4),
        "train_time_s": round(train_time, 4),
        "report":       report,
        "model_obj":    (clf, scaler),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Tâche 2.6 — Deep Learning MLP (PyTorch)
# ══════════════════════════════════════════════════════════════════════════════

class _MLP(nn.Module):
    """MLP simple : 6 → 64 → 32 → 1 (régression score performance 0-100)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 60,
    lr: float = 1e-3,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    Entraîne le MLP pour prédire le score de performance (0-100).
    Retourne MAE, RMSE, temps d'entraînement et courbe de perte.
    """
    if not TORCH_OK:
        return {"error": "PyTorch non disponible"}

    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(len(idx) * (1 - test_size))
    tr_idx, te_idx = idx[:split], idx[split:]

    # Normalisation
    mean, std = X[tr_idx].mean(0), X[tr_idx].std(0) + 1e-8
    X_tr = torch.tensor((X[tr_idx] - mean) / std, dtype=torch.float32)
    X_te = torch.tensor((X[te_idx] - mean) / std, dtype=torch.float32)
    y_tr = torch.tensor(y[tr_idx], dtype=torch.float32)
    y_te = torch.tensor(y[te_idx], dtype=torch.float32)

    model = _MLP()
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses  = []

    t0 = time.perf_counter()
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(X_tr)
        loss = loss_fn(pred, y_tr)
        loss.backward()
        opt.step()
        if ep % 5 == 0:
            losses.append(round(loss.item(), 4))

    train_time = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        preds = model(X_te).numpy()
    targets = y_te.numpy()

    mae  = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))

    return {
        "model":        "MLP (Deep Learning)",
        "mae":          round(mae,  2),
        "rmse":         round(rmse, 2),
        "train_time_s": round(train_time, 4),
        "loss_curve":   losses,
        "model_obj":    (model, mean, std),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Comparaison ML vs DL + prédictions individuelles
# ══════════════════════════════════════════════════════════════════════════════

def _safe(d: dict, key: str, default=None):
    return d.get(key, default)


def compare_models(rf_res: dict, xgb_res: dict, mlp_res: dict) -> dict:
    """Génère un tableau de comparaison résumé."""
    comparison = []

    for res in [rf_res, xgb_res]:
        if "error" not in res:
            comparison.append({
                "modèle":       res["model"],
                "tâche":        "Classification risque blessure",
                "métrique":     f"Accuracy = {res['accuracy']*100:.1f}%",
                "temps_train":  f"{res['train_time_s']:.3f}s",
            })

    if "error" not in mlp_res:
        comparison.append({
            "modèle":       mlp_res["model"],
            "tâche":        "Régression performance future",
            "métrique":     f"MAE = {mlp_res['mae']:.1f} pts | RMSE = {mlp_res['rmse']:.1f}",
            "temps_train":  f"{mlp_res['train_time_s']:.3f}s",
        })

    return {"comparaison_modeles": comparison}


def predict_injury_risk_ml(
    player_features: list[float],
    rf_res: dict,
) -> str | None:
    """
    Prédit le niveau de risque blessure avec Random Forest pour un joueur.
    player_features = [vitesse_moy, distance, sprint_pct, fatigue, decel, play_time_s]
    """
    if "error" in rf_res or rf_res.get("model_obj") is None:
        return None
    clf, scaler = rf_res["model_obj"]
    X = scaler.transform(np.array([player_features], dtype=np.float32))
    idx = clf.predict(X)[0]
    return INJURY_LABELS[idx]


def predict_future_performance_mlp(
    player_features: list[float],
    mlp_res: dict,
) -> float | None:
    """
    Prédit le score de performance futur avec le MLP.
    """
    if not TORCH_OK or "error" in mlp_res or mlp_res.get("model_obj") is None:
        return None
    model, mean, std = mlp_res["model_obj"]
    x = torch.tensor(
        (np.array([player_features], dtype=np.float32) - mean) / (std + 1e-8),
        dtype=torch.float32,
    )
    model.eval()
    with torch.no_grad():
        pred = model(x).item()
    return round(min(max(pred, 0), 100), 1)


# ══════════════════════════════════════════════════════════════════════════════
#  Point d'entrée principal appelé depuis stats.py
# ══════════════════════════════════════════════════════════════════════════════

def run_ml_pipeline(all_player_stats: list[dict]) -> dict:
    """
    Lance tout le pipeline ML S2 sur la liste des stats joueurs.

    Parameters
    ----------
    all_player_stats : liste de dicts avec les clés attendues par build_dataset()

    Returns
    -------
    dict complet avec résultats RF, XGB, MLP, comparaison et prédictions
    """
    if len(all_player_stats) < 2:
        return {"error": "Pas assez de joueurs pour entraîner les modèles (min 2)"}

    X_inj, y_inj, X_perf, y_perf = build_dataset(all_player_stats)

    rf_res  = train_random_forest(X_inj,  y_inj)
    xgb_res = train_xgboost(X_inj,  y_inj)
    mlp_res = train_mlp(X_perf, y_perf)

    comparison = compare_models(rf_res, xgb_res, mlp_res)

    # Prédictions individuelles sur chaque joueur réel
    player_predictions = {}
    for p in all_player_stats:
        pno   = p.get("player_id")
        feats = [
            p.get("vitesse_moyenne_kmh", 0),
            p.get("distance_m", 0),
            p.get("sprint_pct", 0),
            p.get("fatigue_score", 0),
            p.get("decel_accum", 0),
            p.get("play_time_s", 60),
        ]
        inj_ml  = predict_injury_risk_ml(feats, rf_res)
        perf_dl = predict_future_performance_mlp(feats, mlp_res)
        player_predictions[str(pno)] = {
            "injury_risk_ml":         inj_ml,
            "future_performance_mlp": perf_dl,
        }

    # Nettoyer les objets modèle (non-sérialisables JSON)
    for res in [rf_res, xgb_res, mlp_res]:
        res.pop("model_obj", None)

    return {
        "random_forest":       rf_res,
        "xgboost":             xgb_res,
        "mlp":                 mlp_res,
        **comparison,
        "predictions_joueurs": player_predictions,
    }