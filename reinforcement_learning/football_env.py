import numpy as np
from typing import Tuple, Dict


ACTIONS = {
    0: 'substitute',
    1: 'attack',
    2: 'defend',
    3: 'change_tactic',
}

ACTION_NAMES = list(ACTIONS.values())


class FootballEnv:
    """
    Environnement simulé de match de football pour Q-Learning.

    État  : (minute_bin, score_diff, fatigue_level, possession_bin)
    Actions : 4 — substitute / attack / defend / change_tactic

    L'environnement simule l'évolution d'un match minute par minute.
    """

    MINUTE_BINS = [0, 15, 30, 45, 60, 75, 90]   # 6 bins
    FATIGUE_BINS = [0, 25, 50, 75, 100]           # 4 bins
    POSSESSION_BINS = [0, 35, 50, 65, 100]        # 4 bins
    SCORE_RANGE = range(-3, 4)                     # -3 … +3 → 7 values

    N_MINUTE_BINS = 6
    N_FATIGUE_BINS = 4
    N_POSSESSION_BINS = 4
    N_SCORE = 7                                    # -3..+3 shifted by 3

    N_STATES = N_MINUTE_BINS * N_SCORE * N_FATIGUE_BINS * N_POSSESSION_BINS
    N_ACTIONS = 4

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> int:
        self.minute = 0
        self.score_diff = 0
        self.avg_fatigue = self.rng.uniform(10, 30)
        self.possession = self.rng.uniform(40, 60)
        self.subs_remaining = 3
        self.done = False
        return self._state_index()

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        assert not self.done, "Episode finished. Call reset()."
        assert action in ACTIONS, f"Invalid action {action}."

        reward = self._apply_action(action)

        # Simulate one time-step (≈ 5 minutes)
        self.minute = min(self.minute + 5, 90)
        self._simulate_dynamics()

        self.done = self.minute >= 90

        if self.done:
            reward += self._final_reward()

        info = {
            'minute': self.minute,
            'score_diff': self.score_diff,
            'avg_fatigue': round(self.avg_fatigue, 1),
            'possession': round(self.possession, 1),
            'action_name': ACTIONS[action],
        }
        return self._state_index(), reward, self.done, info

    # ── private helpers ───────────────────────────────────────────────────────

    def _apply_action(self, action: int) -> float:
        reward = 0.0

        if action == 0:  # substitute
            if self.subs_remaining > 0:
                fatigue_reduction = self.rng.uniform(5, 15)
                self.avg_fatigue = max(0, self.avg_fatigue - fatigue_reduction)
                self.subs_remaining -= 1
                reward += 0.3
            else:
                reward -= 0.5  # penalty: no subs left

        elif action == 1:  # attack
            # Higher possession, chance to score, but defensive risk
            self.possession = min(80, self.possession + self.rng.uniform(2, 8))
            self.avg_fatigue = min(100, self.avg_fatigue + self.rng.uniform(1, 4))
            if self.rng.random() < 0.20:
                self.score_diff += 1
                reward += 1.5
            if self.rng.random() < 0.12:  # concede on counter
                self.score_diff -= 1
                reward -= 1.0

        elif action == 2:  # defend
            self.possession = max(30, self.possession - self.rng.uniform(2, 6))
            self.avg_fatigue = max(0, self.avg_fatigue - self.rng.uniform(0, 2))
            if self.score_diff > 0:
                reward += 0.4   # reward holding a lead
            if self.rng.random() < 0.06:  # opponent scores anyway
                self.score_diff -= 1
                reward -= 0.8

        elif action == 3:  # change tactic
            # Uncertain outcome — disrupts current flow but may unlock the game
            self.possession += self.rng.uniform(-5, 5)
            self.possession = np.clip(self.possession, 30, 80)
            self.avg_fatigue = min(100, self.avg_fatigue + self.rng.uniform(0, 3))
            reward += self.rng.uniform(-0.2, 0.4)

        return reward

    def _simulate_dynamics(self):
        # Natural fatigue increase
        self.avg_fatigue = min(100, self.avg_fatigue + self.rng.uniform(0.5, 2.0))
        # Possession drift toward 50%
        self.possession += (50 - self.possession) * 0.05
        self.possession += self.rng.uniform(-3, 3)
        self.possession = np.clip(self.possession, 30, 75)

    def _final_reward(self) -> float:
        if self.score_diff > 0:
            return 3.0
        if self.score_diff == 0:
            return 1.0
        return -1.0

    def _state_index(self) -> int:
        mb = np.searchsorted(self.MINUTE_BINS[1:], self.minute, side='right')
        mb = min(mb, self.N_MINUTE_BINS - 1)

        sd = int(np.clip(self.score_diff + 3, 0, 6))

        fb = np.searchsorted(self.FATIGUE_BINS[1:], self.avg_fatigue, side='right')
        fb = min(fb, self.N_FATIGUE_BINS - 1)

        pb = np.searchsorted(self.POSSESSION_BINS[1:], self.possession, side='right')
        pb = min(pb, self.N_POSSESSION_BINS - 1)

        return (mb * self.N_SCORE * self.N_FATIGUE_BINS * self.N_POSSESSION_BINS
                + sd * self.N_FATIGUE_BINS * self.N_POSSESSION_BINS
                + fb * self.N_POSSESSION_BINS
                + pb)

    @property
    def state_description(self) -> Dict:
        return {
            'minute': self.minute,
            'score_diff': self.score_diff,
            'avg_fatigue': round(self.avg_fatigue, 1),
            'possession': round(self.possession, 1),
            'subs_remaining': self.subs_remaining,
        }
