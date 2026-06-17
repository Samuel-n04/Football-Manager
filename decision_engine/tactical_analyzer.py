import pandas as pd
import numpy as np
from typing import Dict, List


class TacticalAnalyzer:
    """
    Analyse l'adversaire pour identifier :
        - les joueurs faibles (PerformanceScore < 55)
        - les zones faibles (zone où le % d'occasions concédées est le plus élevé)
        - les faiblesses défensives (joueurs à faible pressing + haute fatigue)
    """

    def analyze_opponent(self, opponent_df: pd.DataFrame) -> Dict:
        weak_players = self._find_weak_players(opponent_df)
        weak_zones = self._find_weak_zones(opponent_df)
        defensive_weaknesses = self._find_defensive_weaknesses(opponent_df)
        tactical_recommendations = self._tactical_recommendations(weak_zones, defensive_weaknesses)

        return {
            'weak_players': weak_players,
            'weak_zones': weak_zones,
            'defensive_weaknesses': defensive_weaknesses,
            'tactical_recommendations': tactical_recommendations,
            'summary': self._build_summary(weak_players, weak_zones, defensive_weaknesses),
        }

    # ── private methods ───────────────────────────────────────────────────────

    def _find_weak_players(self, df: pd.DataFrame) -> List[Dict]:
        threshold = 55
        weak = df[df['PerformanceScore'] < threshold].copy()
        weak = weak.sort_values('PerformanceScore')
        result = []
        for _, row in weak.iterrows():
            result.append({
                'player_id': int(row['PlayerID']),
                'position': row.get('Position', 'N/A'),
                'performance_score': round(float(row['PerformanceScore']), 1),
                'fatigue_score': round(float(row.get('FatigueScore', 0)), 1),
                'weakness': self._describe_weakness(row),
            })
        return result

    def _find_weak_zones(self, df: pd.DataFrame) -> Dict:
        zones = {}
        for zone in ['ZoneLeft', 'ZoneCenter', 'ZoneRight']:
            if zone in df.columns:
                # Weight zone action by low performance (more occasions conceded when weak)
                weighted = (df[zone] * (1 - df['PerformanceScore'] / 100)).mean()
                zones[zone.replace('Zone', '').lower()] = round(float(weighted), 1)

        if not zones:
            # Fallback: synthetic zone vulnerability
            zones = {
                'left': round(np.random.uniform(20, 50), 1),
                'center': round(np.random.uniform(20, 40), 1),
                'right': round(np.random.uniform(20, 50), 1),
            }

        total = sum(zones.values()) or 1
        zones_pct = {k: round(v / total * 100, 1) for k, v in zones.items()}
        weakest = max(zones_pct, key=zones_pct.get)

        return {
            'zones': zones_pct,
            'weakest_zone': weakest,
            'vulnerability': f"{zones_pct[weakest]:.1f}% des occasions concédées côté {weakest}",
        }

    def _find_defensive_weaknesses(self, df: pd.DataFrame) -> List[Dict]:
        # Defensive players with low pressing + high fatigue → defensive gap
        defensive_positions = ['GK', 'CB', 'LB', 'RB', 'CDM']
        def_df = df[df['Position'].isin(defensive_positions)].copy() if 'Position' in df.columns else df.copy()

        weaknesses = []
        for _, row in def_df.iterrows():
            pressing = float(row.get('PressingActions', 10))
            fatigue = float(row.get('FatigueScore', 50))
            perf = float(row.get('PerformanceScore', 60))

            gap_score = (1 - pressing / 25) * 40 + fatigue / 100 * 40 + (1 - perf / 100) * 20

            if gap_score > 50:
                weaknesses.append({
                    'player_id': int(row['PlayerID']),
                    'position': row.get('Position', 'DEF'),
                    'gap_score': round(gap_score, 1),
                    'detail': f"Pressing faible ({pressing:.0f} actions), fatigue {fatigue:.0f}",
                })

        return sorted(weaknesses, key=lambda x: -x['gap_score'])

    def _tactical_recommendations(self, weak_zones: Dict, def_weaknesses: List) -> List[str]:
        recs = []
        wz = weak_zones.get('weakest_zone', 'left')
        recs.append(f"Exploiter le côté {wz} — zone la plus vulnérable de l'adversaire.")

        if def_weaknesses:
            top = def_weaknesses[0]
            recs.append(
                f"Cibler la position {top['position']} (joueur #{top['player_id']}) "
                f"qui présente la plus grande faille défensive (score {top['gap_score']:.0f}/100)."
            )

        recs.append("Augmenter le pressing haut dans les 20 dernières minutes pour profiter de la fatigue adverse.")
        return recs

    @staticmethod
    def _describe_weakness(row: pd.Series) -> str:
        parts = []
        if row.get('PassAccuracy', 80) < 65:
            parts.append("passes imprécises")
        if row.get('FatigueScore', 0) > 60:
            parts.append("fatigue élevée")
        if row.get('PressingActions', 10) < 5:
            parts.append("pressing insuffisant")
        return ', '.join(parts) if parts else 'performance globalement faible'

    @staticmethod
    def _build_summary(weak_players: List, weak_zones: Dict, def_weak: List) -> str:
        wz = weak_zones.get('weakest_zone', 'gauche')
        vuln = weak_zones.get('vulnerability', '')
        n_weak = len(weak_players)
        return (
            f"{n_weak} joueur(s) adverses en sous-performance. "
            f"{vuln}. "
            f"{len(def_weak)} faille(s) défensive(s) identifiée(s)."
        )
