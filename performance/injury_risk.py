import pandas as pd
import numpy as np


class InjuryRiskCalculator:
    """
    Calcule l'Injury Risk Score (0-100).

    Formule :
        40 % fatigue + 20 % temps de jeu + 20 % accélérations + 20 % distance
    """

    MAX_MINUTES = 95
    MAX_ACCELERATIONS = 110
    MAX_DISTANCE = 13000

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['InjuryRiskScore'] = df.apply(self._risk_row, axis=1).round(2)
        df['InjuryRisk'] = df['InjuryRiskScore'].apply(self._classify_score)
        return df

    def _risk_row(self, row) -> float:
        fatigue = float(row.get('FatigueScore', 50))
        t = min(float(row.get('MinutesPlayed', 0)) / self.MAX_MINUTES * 100, 100)
        a = min(float(row.get('Accelerations', 0)) / self.MAX_ACCELERATIONS * 100, 100)
        d = min(float(row.get('Distance', 0)) / self.MAX_DISTANCE * 100, 100)
        return 0.40 * fatigue + 0.20 * t + 0.20 * a + 0.20 * d

    @staticmethod
    def _classify_score(score: float) -> str:
        if score < 33:
            return 'Faible'
        if score < 66:
            return 'Moyen'
        return 'Eleve'

    def high_risk_players(self, df: pd.DataFrame, threshold: float = 66) -> pd.DataFrame:
        return df[df['InjuryRiskScore'] >= threshold].copy()

    def risk_summary(self, df: pd.DataFrame) -> dict:
        counts = df['InjuryRisk'].value_counts().to_dict()
        return {
            'Faible': counts.get('Faible', 0),
            'Moyen': counts.get('Moyen', 0),
            'Eleve': counts.get('Eleve', 0),
        }
