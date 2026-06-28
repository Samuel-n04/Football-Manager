"""
tracking/reinforcement.py
=========================
Semaine 4 — Reinforcement Learning

Tâche 4.1 : Environnement simulé de match de football
Tâche 4.2 : Q-Learning (entraînement + Q-Table)
Tâche 4.3 : Comparaison décision Coach humain vs IA
"""

from __future__ import annotations
import os
import random
import numpy as np

os.makedirs("models", exist_ok=True)

Q_TABLE_PATH = "models/tactical_q_table.npy"

# ── Actions ───────────────────────────────────────────────────────────────────
ACTIONS = {
    0: "Remplacer joueur",
    1: "Attaquer",
    2: "Défendre",
    3: "Changer tactique",
}

# ══════════════════════════════════════════════════════════════════════════════
#  Tâche 4.1 — Environnement simulé
# ══════════════════════════════════════════════════════════════════════════════

class FootballEnv:
    """
    Environnement de simulation d'un match de football.

    Espace d'états (discret) :
      - minute     : 0 (0-30 min)  · 1 (30-60 min)  · 2 (60-90 min)
      - score_diff : -2, -1, 0, +1, +2  (clamped)
      - fatigue    : 0 (< 40%)  · 1 (40-75%)  · 2 (> 75%)
      - possession : 0 (adversaire) · 1 (notre équipe)

    Actions : 0=Remplacer · 1=Attaquer · 2=Défendre · 3=Changer tactique
    """

    def __init__(self):
        self.action_space = ACTIONS
        self.reset()

    def reset(self):
        self.minute     = 0
        self.score_diff = 0
        self.fatigue    = 0
        self.possession = 1
        return self._state()

    def _state(self):
        return (self.minute,
                int(np.clip(self.score_diff, -2, 2)),
                self.fatigue,
                self.possession)

    def step(self, action: int):
        reward = 0
        self.minute = min(self.minute + 1, 2)

        if action == 0:   # Remplacer joueur
            self.fatigue    = max(0, self.fatigue - 1)
            reward         += 2
            # Légère chance de marquer après un remplacement offensif
            if self.possession == 1 and random.random() > 0.75:
                self.score_diff += 1
                reward          += 5

        elif action == 1:  # Attaquer
            if self.possession == 1 and self.fatigue < 2:
                if random.random() > 0.40:
                    self.score_diff += 1
                    reward          += 10
                else:
                    self.possession  = 0
                    reward          -= 1
            else:
                self.possession = 0
                reward         -= 3

        elif action == 2:  # Défendre
            if self.possession == 0:
                if random.random() > 0.45:
                    self.possession = 1
                reward += 3
            else:
                reward -= 1   # Pénalité légère si on défend quand on a la balle

        elif action == 3:  # Changer tactique
            reward += 1
            if self.fatigue == 2:
                reward -= 2   # Inutile sans gestion physique d'abord

        # Pénalité fatigue non gérée
        if self.fatigue == 2 and action != 0:
            reward -= 4
            if random.random() > 0.65:
                self.score_diff -= 1
                reward          -= 3

        # Bonus résultat final
        done = (self.minute >= 2)
        if done:
            if self.score_diff > 0:
                reward += 20
            elif self.score_diff == 0:
                reward += 5
            else:
                reward -= 10

        return self._state(), reward, done


# ══════════════════════════════════════════════════════════════════════════════
#  Tâche 4.2 — Q-Learning
# ══════════════════════════════════════════════════════════════════════════════

def train_q_learning(
    n_episodes: int = 8000,
    alpha: float = 0.15,
    gamma: float = 0.95,
    epsilon_start: float = 0.5,
    epsilon_end: float = 0.05,
    force_retrain: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Entraîne l'agent Q-Learning sur n_episodes matchs simulés.

    Q-Table shape : (3, 5, 3, 2, 4)
      minutes(3) × score_diff(5) × fatigue(3) × possession(2) × actions(4)

    Retourne (q_table, training_stats).
    """
    # Si déjà entraîné et pas de forçage, on charge directement
    if not force_retrain and os.path.exists(Q_TABLE_PATH):
        q_table = np.load(Q_TABLE_PATH)
        return q_table, {"loaded_from_cache": True}

    env     = FootballEnv()
    q_table = np.zeros((3, 5, 3, 2, len(ACTIONS)))

    rewards_per_episode = []
    epsilon_decay = (epsilon_start - epsilon_end) / n_episodes

    for ep in range(n_episodes):
        state = env.reset()
        done  = False
        total_reward = 0
        epsilon = max(epsilon_end, epsilon_start - ep * epsilon_decay)

        while not done:
            m, s, f, p = state
            s_idx = s + 2   # -2..+2 → 0..4

            # Epsilon-greedy
            if random.random() < epsilon:
                action = random.randrange(len(ACTIONS))
            else:
                action = int(np.argmax(q_table[m, s_idx, f, p]))

            next_state, reward, done = env.step(action)
            nm, ns, nf, np_ = next_state
            ns_idx = ns + 2

            # Bellman update
            old_val  = q_table[m, s_idx, f, p, action]
            next_max = float(np.max(q_table[nm, ns_idx, nf, np_]))
            q_table[m, s_idx, f, p, action] = (
                old_val + alpha * (reward + gamma * next_max - old_val)
            )

            state        = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

    np.save(Q_TABLE_PATH, q_table)
    print(f"[RL] Q-Learning entraîné ({n_episodes} épisodes). Q-Table sauvegardée.")

    # Stats d'entraînement (moyennes glissantes)
    rpe = np.array(rewards_per_episode)
    training_stats = {
        "n_episodes":         n_episodes,
        "reward_mean":        round(float(rpe.mean()), 2),
        "reward_last_500":    round(float(rpe[-500:].mean()), 2),
        "reward_curve_sample": rpe[::max(1, n_episodes // 50)].round(2).tolist(),
        "loaded_from_cache":  False,
    }
    return q_table, training_stats


# ══════════════════════════════════════════════════════════════════════════════
#  Consultation de la Q-Table
# ══════════════════════════════════════════════════════════════════════════════

def get_ia_action(
    q_table: np.ndarray,
    minute: int,
    score_diff: int,
    fatigue_pct: float,
    possession: int,
) -> dict:
    """
    Retourne la décision optimale de l'IA pour un état de match donné.

    Parameters
    ----------
    minute      : 0-90
    score_diff  : différence de buts (négatif = on perd)
    fatigue_pct : fatigue globale en % (0-100)
    possession  : 1 = notre équipe  · 0 = adversaire

    Returns
    -------
    dict avec action_id, action_label, q_values, confiance
    """
    m_idx = 0 if minute < 30 else (1 if minute < 60 else 2)
    s_idx = int(np.clip(score_diff, -2, 2)) + 2
    f_idx = 0 if fatigue_pct < 40 else (1 if fatigue_pct < 75 else 2)
    p_idx = int(bool(possession))

    q_vals   = q_table[m_idx, s_idx, f_idx, p_idx]
    best_act = int(np.argmax(q_vals))

    # Confiance = écart entre la meilleure et la 2ème action
    sorted_q = np.sort(q_vals)[::-1]
    gap      = float(sorted_q[0] - sorted_q[1]) if len(sorted_q) > 1 else 0.0
    confiance = min(round(gap / max(abs(sorted_q[0]) + 1e-6, 1) * 100, 1), 100.0)

    return {
        "action_id":    best_act,
        "action_label": ACTIONS[best_act],
        "q_values":     {ACTIONS[i]: round(float(q_vals[i]), 3) for i in range(len(ACTIONS))},
        "confiance":    confiance,
        "etat":         {"minute": minute, "score_diff": score_diff,
                         "fatigue_pct": fatigue_pct, "possession": possession},
    }


def load_or_train_qtable(force_retrain: bool = False) -> tuple[np.ndarray, dict]:
    """Charge la Q-Table si elle existe, sinon l'entraîne."""
    return train_q_learning(force_retrain=force_retrain)


# ══════════════════════════════════════════════════════════════════════════════
#  Tâche 4.3 — Comparaison Coach humain vs IA
# ══════════════════════════════════════════════════════════════════════════════

COACH_SCENARIOS = [
    {"minute": 75, "score_diff": -1, "fatigue_pct": 80, "possession": 0,
     "coach_action": 0, "context": "On perd 0-1 à la 75e, équipe épuisée, adversaire domine"},
    {"minute": 30, "score_diff":  1, "fatigue_pct": 20, "possession": 1,
     "coach_action": 1, "context": "On mène 1-0 à la 30e, frais, possession"},
    {"minute": 60, "score_diff":  0, "fatigue_pct": 55, "possession": 0,
     "coach_action": 2, "context": "Match nul à la 60e, équipe modérément fatiguée, adversaire en possession"},
    {"minute": 85, "score_diff":  1, "fatigue_pct": 75, "possession": 1,
     "coach_action": 2, "context": "On mène 1-0 à la 85e, équipe fatiguée, on a la balle"},
    {"minute": 45, "score_diff": -2, "fatigue_pct": 40, "possession": 0,
     "coach_action": 1, "context": "On perd 0-2 à la mi-temps, fatigue modérée"},
]


def compare_coach_vs_ia(q_table: np.ndarray) -> list[dict]:
    """
    Tâche 4.3 — Compare les décisions du coach humain (défini dans COACH_SCENARIOS)
    avec celles de l'IA Q-Learning sur 5 situations types.

    Retourne une liste de dicts prêts pour l'affichage.
    """
    results = []
    agreements = 0

    for sc in COACH_SCENARIOS:
        ia = get_ia_action(
            q_table,
            minute=sc["minute"],
            score_diff=sc["score_diff"],
            fatigue_pct=sc["fatigue_pct"],
            possession=sc["possession"],
        )
        coach_lbl = ACTIONS.get(sc["coach_action"], "?")
        ia_lbl    = ia["action_label"]
        accord    = (sc["coach_action"] == ia["action_id"])
        if accord:
            agreements += 1

        results.append({
            "contexte":       sc["context"],
            "minute":         sc["minute"],
            "score_diff":     sc["score_diff"],
            "fatigue_pct":    sc["fatigue_pct"],
            "possession":     "Notre équipe" if sc["possession"] else "Adversaire",
            "coach":          coach_lbl,
            "ia":             ia_lbl,
            "accord":         accord,
            "confiance_ia":   ia["confiance"],
            "q_values":       ia["q_values"],
        })

    taux_accord = round(agreements / len(COACH_SCENARIOS) * 100, 1)
    for r in results:
        r["taux_accord_global"] = taux_accord

    return results