import numpy as np
import pandas as pd
from typing import Dict, List

FORMATIONS = {
    '4-3-3': {
        'defenders': 4, 'midfielders': 3, 'attackers': 3,
        'possession_base': 55, 'pressing_base': 65,
        'xG_base': 1.8, 'defensive_risk_base': 35,
        'description': 'Formation offensive équilibrée. Bon pressing, ailiers rapides.',
    },
    '4-4-2': {
        'defenders': 4, 'midfielders': 4, 'attackers': 2,
        'possession_base': 50, 'pressing_base': 55,
        'xG_base': 1.5, 'defensive_risk_base': 30,
        'description': 'Formation classique. Bloc compact, bons duels au milieu.',
    },
    '3-5-2': {
        'defenders': 3, 'midfielders': 5, 'attackers': 2,
        'possession_base': 60, 'pressing_base': 60,
        'xG_base': 1.6, 'defensive_risk_base': 45,
        'description': 'Domination du milieu, piston offensif, défense à 3 risquée.',
    },
    '3-4-3': {
        'defenders': 3, 'midfielders': 4, 'attackers': 3,
        'possession_base': 52, 'pressing_base': 70,
        'xG_base': 2.1, 'defensive_risk_base': 55,
        'description': 'Ultra-offensif. Prise de risque défensive élevée.',
    },
}


class TacticalSimulator:
    """
    Simule et compare des formations tactiques.

    Les indicateurs simulés :
        - Possession (%)
        - Pressing intensity (0-100)
        - xG simplifié (expected goals)
        - Risque défensif (0-100)
        - Score global de formation (plus élevé = meilleur)
    """

    def simulate_all(self, player_df: pd.DataFrame, context: dict | None = None) -> pd.DataFrame:
        """
        Simulate all four formations given the current player stats.
        context: {'minute': int, 'score_diff': int}
        """
        context = context or {'minute': 0, 'score_diff': 0}
        rows = []
        for formation, base in FORMATIONS.items():
            metrics = self._simulate_formation(formation, base, player_df, context)
            rows.append(metrics)
        df = pd.DataFrame(rows)
        df['GlobalScore'] = self._global_score(df)
        df = df.sort_values('GlobalScore', ascending=False).reset_index(drop=True)
        return df

    def recommend_formation(self, player_df: pd.DataFrame, context: dict | None = None) -> Dict:
        results = self.simulate_all(player_df, context)
        best = results.iloc[0]
        second = results.iloc[1] if len(results) > 1 else None
        return {
            'recommended_formation': best['Formation'],
            'global_score': round(float(best['GlobalScore']), 1),
            'possession': round(float(best['Possession']), 1),
            'pressing': round(float(best['Pressing']), 1),
            'xG': round(float(best['xG']), 2),
            'defensive_risk': round(float(best['DefensiveRisk']), 1),
            'description': FORMATIONS[best['Formation']]['description'],
            'alternatives': [
                {
                    'formation': second['Formation'],
                    'global_score': round(float(second['GlobalScore']), 1),
                }
            ] if second is not None else [],
            'all_results': results.to_dict(orient='records'),
        }

    # ── private helpers ───────────────────────────────────────────────────────

    def _simulate_formation(self, name: str, base: dict, df: pd.DataFrame, ctx: dict) -> dict:
        avg_fatigue = float(df['FatigueScore'].mean()) if 'FatigueScore' in df else 50
        avg_perf = float(df['PerformanceScore'].mean()) if 'PerformanceScore' in df else 65
        avg_speed = float(df['SpeedAvg'].mean()) if 'SpeedAvg' in df else 15
        avg_pass = float(df['PassAccuracy'].mean()) if 'PassAccuracy' in df else 75

        freshness = 1 - avg_fatigue / 100       # 0-1
        perf_factor = avg_perf / 100             # 0-1
        speed_bonus = (avg_speed - 10) / 20      # 0-1 approx

        minute = ctx.get('minute', 0)
        score_diff = ctx.get('score_diff', 0)    # positive = winning

        # Fatigue reduces all metrics as match progresses
        fatigue_penalty = avg_fatigue * 0.3 * (minute / 90)

        possession = np.clip(
            base['possession_base'] + perf_factor * 10 + avg_pass / 100 * 5 - fatigue_penalty,
            30, 75
        )
        pressing = np.clip(
            base['pressing_base'] + freshness * 15 - fatigue_penalty * 0.5,
            20, 90
        )
        xg = np.clip(
            base['xG_base'] + speed_bonus * 0.4 + perf_factor * 0.3 - fatigue_penalty * 0.01,
            0.5, 4.0
        )

        # If losing, defensive risk tolerated more; if winning, lower risk preferred
        tactical_bias = -score_diff * 3
        defensive_risk = np.clip(
            base['defensive_risk_base'] + (1 - freshness) * 15 + tactical_bias,
            10, 80
        )

        return {
            'Formation': name,
            'Possession': round(float(possession), 1),
            'Pressing': round(float(pressing), 1),
            'xG': round(float(xg), 2),
            'DefensiveRisk': round(float(defensive_risk), 1),
            'Description': base['description'],
        }

    @staticmethod
    def _global_score(df: pd.DataFrame) -> pd.Series:
        # Higher possession, pressing, xG → better; lower defensive risk → better
        score = (
            df['Possession'] * 0.25
            + df['Pressing'] * 0.25
            + df['xG'] * 15          # scale xG to ~0-60
            - df['DefensiveRisk'] * 0.25
        )
        return score.round(2)
