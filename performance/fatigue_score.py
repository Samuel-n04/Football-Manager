import pandas as pd
import numpy as np


class FatigueScoreCalculator:
    """
    Calcule le Fatigue Score (0-100) à partir des indicateurs physiques.
    Score élevé = joueur très fatigué.

    Formule :
        35 % distance + 30 % nombre de sprints + 20 % vitesse moyenne + 15 % temps de jeu
    """

    # Valeurs max de référence pour la normalisation
    MAX_DISTANCE = 13000       # mètres
    MAX_SPRINTS = 45
    MAX_SPEED = 28             # km/h
    MAX_MINUTES = 95

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['FatigueScore'] = df.apply(self._fatigue_row, axis=1).round(2)
        return df

    def _fatigue_row(self, row) -> float:
        d = min(float(row.get('Distance', 0)) / self.MAX_DISTANCE * 100, 100)
        s = min(float(row.get('SprintCount', 0)) / self.MAX_SPRINTS * 100, 100)
        v = min(float(row.get('SpeedAvg', 0)) / self.MAX_SPEED * 100, 100)
        t = min(float(row.get('MinutesPlayed', 0)) / self.MAX_MINUTES * 100, 100)
        return 0.35 * d + 0.30 * s + 0.20 * v + 0.15 * t

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        bins = [0, 30, 55, 70, 100]
        labels = ['Frais', 'Modéré', 'Fatigué', 'Épuisé']
        df['FatigueLevel'] = pd.cut(df['FatigueScore'], bins=bins, labels=labels)
        return df

    def detect_high_fatigue(self, df: pd.DataFrame, threshold: float = 70) -> pd.DataFrame:
        return df[df['FatigueScore'] > threshold].copy()

    def compute_team_fatigue(self, df: pd.DataFrame) -> dict:
        """Return average fatigue score per team."""
        if 'Team' not in df.columns:
            return {}
        return df.groupby('Team')['FatigueScore'].mean().round(2).to_dict()
