import numpy as np
import pandas as pd
import pickle
import os
from typing import List, Dict
from .football_env import FootballEnv, ACTIONS, ACTION_NAMES


class QLearningAgent:
    """
    Agent Q-Learning pour optimiser les décisions tactiques.

    Hyperparamètres :
        alpha    : taux d'apprentissage
        gamma    : facteur d'actualisation
        epsilon  : exploration initiale (décroît vers epsilon_min)
    """

    def __init__(
        self,
        n_states: int = FootballEnv.N_STATES,
        n_actions: int = FootballEnv.N_ACTIONS,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.Q = np.zeros((n_states, n_actions))
        self.episode_rewards: List[float] = []
        self.episode_outcomes: List[str] = []

    # ── training ─────────────────────────────────────────────────────────────

    def train(self, n_episodes: int = 2000, seed: int = 42) -> Dict:
        env = FootballEnv(seed=seed)

        for ep in range(n_episodes):
            state = env.reset()
            total_reward = 0.0

            while True:
                action = self._choose_action(state)
                next_state, reward, done, _ = env.step(action)

                # Q-update
                best_next = np.max(self.Q[next_state])
                self.Q[state, action] += self.alpha * (
                    reward + self.gamma * best_next - self.Q[state, action]
                )

                state = next_state
                total_reward += reward

                if done:
                    break

            self.episode_rewards.append(total_reward)
            score_diff = env.score_diff
            self.episode_outcomes.append(
                'win' if score_diff > 0 else ('draw' if score_diff == 0 else 'loss')
            )

            # Decay exploration
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return self._training_summary(n_episodes)

    # ── inference ─────────────────────────────────────────────────────────────

    def decide(self, state: int) -> Dict:
        action = int(np.argmax(self.Q[state]))
        return {
            'action_id': action,
            'action_name': ACTIONS[action],
            'q_values': {ACTIONS[i]: round(float(self.Q[state, i]), 4) for i in range(self.n_actions)},
            'confidence': round(float(self._softmax(self.Q[state])[action]), 4),
        }

    def compare_with_human(self, human_action: int, state: int) -> Dict:
        ai_decision = self.decide(state)
        ai_action = ai_decision['action_id']
        ai_q = self.Q[state, ai_action]
        human_q = self.Q[state, human_action]
        return {
            'minute': '?',
            'human_action': ACTIONS[human_action],
            'ai_action': ACTIONS[ai_action],
            'human_q_value': round(float(human_q), 4),
            'ai_q_value': round(float(ai_q), 4),
            'advantage': round(float(ai_q - human_q), 4),
            'verdict': 'IA supérieure' if ai_q > human_q else ('Égal' if ai_q == human_q else 'Humain supérieur'),
        }

    def policy_table(self) -> pd.DataFrame:
        best_actions = np.argmax(self.Q, axis=1)
        rows = [
            {'state': s, 'best_action': ACTIONS[a], 'q_value': round(float(self.Q[s, a]), 4)}
            for s, a in enumerate(best_actions)
        ]
        return pd.DataFrame(rows)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str = 'stubs/q_agent.pkl') -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str = 'stubs/q_agent.pkl') -> 'QLearningAgent':
        with open(path, 'rb') as f:
            return pickle.load(f)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _choose_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    @staticmethod
    def _softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
        e = np.exp((x - np.max(x)) / temp)
        return e / e.sum()

    def _training_summary(self, n_episodes: int) -> Dict:
        outcomes = self.episode_outcomes
        last_200 = outcomes[-200:] if len(outcomes) >= 200 else outcomes
        return {
            'episodes_trained': n_episodes,
            'win_rate': round(last_200.count('win') / len(last_200) * 100, 1),
            'draw_rate': round(last_200.count('draw') / len(last_200) * 100, 1),
            'loss_rate': round(last_200.count('loss') / len(last_200) * 100, 1),
            'avg_reward_last_200': round(float(np.mean(self.episode_rewards[-200:])), 3),
            'final_epsilon': round(self.epsilon, 4),
            'most_chosen_action': ACTIONS[
                int(np.argmax(np.bincount(np.argmax(self.Q, axis=1))))
            ],
        }
