import pandas as pd
import numpy as np
from typing import Optional, List, Dict


class SubstitutionRecommender:
    """
    Détermine quel joueur remplacer, par qui, et pourquoi.

    Le score de compatibilité tactique combine :
        40 % fraîcheur physique (1 - Fatigue/100)
        30 % performance récente
        30 % complémentarité positionnelle
    """

    POSITION_COMPATIBILITY: dict = {
        'GK':  ['GK'],
        'CB':  ['CB', 'LB', 'RB'],
        'LB':  ['LB', 'CB', 'CAM'],
        'RB':  ['RB', 'CB', 'CAM'],
        'CDM': ['CDM', 'CM'],
        'CM':  ['CM', 'CDM', 'CAM'],
        'CAM': ['CAM', 'CM', 'LW', 'RW'],
        'LW':  ['LW', 'CAM', 'CF'],
        'RW':  ['RW', 'CAM', 'CF'],
        'CF':  ['CF', 'ST', 'LW', 'RW'],
        'ST':  ['ST', 'CF'],
    }

    def recommend(
        self,
        field_players: pd.DataFrame,
        bench_players: pd.DataFrame,
        max_subs: int = 3,
    ) -> List[Dict]:
        """
        field_players : DataFrame des joueurs sur le terrain (PlayerID, Position,
                        FatigueScore, PerformanceScore, InjuryRisk, …)
        bench_players : DataFrame des remplaçants disponibles.
        Retourne une liste ordonnée de recommandations de remplacement.
        """
        recommendations = []
        remaining_subs = max_subs

        # Sort field players: prioritise highest fatigue + lowest performance
        field_sorted = field_players.copy()
        field_sorted['SubPriority'] = (
            field_sorted['FatigueScore'] * 0.5
            - field_sorted['PerformanceScore'] * 0.5
        )
        field_sorted = field_sorted.sort_values('SubPriority', ascending=False)

        already_subbed_out: List = []

        for _, out_player in field_sorted.iterrows():
            if remaining_subs == 0:
                break

            fatigue = out_player.get('FatigueScore', 0)
            perf = out_player.get('PerformanceScore', 100)
            injury = out_player.get('InjuryRisk', 'Faible')

            # Only recommend if there is a genuine concern
            if fatigue <= 55 and perf >= 55 and injury != 'Eleve':
                continue

            position = out_player.get('Position', 'CM')
            compatible_positions = self.POSITION_COMPATIBILITY.get(position, [position])

            candidates = bench_players[
                bench_players['Position'].isin(compatible_positions)
            ].copy()

            if candidates.empty:
                candidates = bench_players.copy()

            candidates['TacticalCompatibilityScore'] = candidates.apply(
                lambda r: self._tactical_score(r, position), axis=1
            )
            candidates = candidates.sort_values('TacticalCompatibilityScore', ascending=False)

            if candidates.empty:
                continue

            best_sub = candidates.iloc[0]

            recommendations.append({
                'out_player_id': int(out_player['PlayerID']),
                'out_position': position,
                'out_fatigue': round(float(fatigue), 1),
                'out_performance': round(float(perf), 1),
                'out_injury_risk': injury,
                'in_player_id': int(best_sub['PlayerID']),
                'in_position': best_sub.get('Position', 'CM'),
                'in_fatigue': round(float(best_sub.get('FatigueScore', 20)), 1),
                'in_performance': round(float(best_sub.get('PerformanceScore', 70)), 1),
                'tactical_score': round(float(best_sub['TacticalCompatibilityScore']), 1),
                'reason': self._build_reason(out_player, best_sub),
                'urgency': self._urgency(fatigue, perf, injury),
            })

            already_subbed_out.append(int(out_player['PlayerID']))
            bench_players = bench_players[bench_players['PlayerID'] != best_sub['PlayerID']]
            remaining_subs -= 1

        return recommendations

    # ── helpers ──────────────────────────────────────────────────────────────

    def _tactical_score(self, bench_row, out_position: str) -> float:
        freshness = (1 - bench_row.get('FatigueScore', 20) / 100) * 40
        perf = bench_row.get('PerformanceScore', 70) / 100 * 30
        compat = 30 if bench_row.get('Position', '') in self.POSITION_COMPATIBILITY.get(out_position, []) else 15
        return freshness + perf + compat

    @staticmethod
    def _build_reason(out: pd.Series, inp: pd.Series) -> str:
        parts = []
        if out.get('FatigueScore', 0) > 70:
            parts.append(f"fatigue = {out['FatigueScore']:.0f}")
        if out.get('PerformanceScore', 100) < 50:
            parts.append(f"performance = {out['PerformanceScore']:.0f}")
        if out.get('InjuryRisk', '') == 'Eleve':
            parts.append("risque blessure = élevé")
        reason_out = ', '.join(parts) if parts else 'surveillance préventive'
        return (
            f"Remplacer Joueur #{int(out['PlayerID'])} "
            f"(position : {out.get('Position','?')}) — {reason_out}. "
            f"Faire entrer Joueur #{int(inp['PlayerID'])} "
            f"(fraîcheur physique : {100 - inp.get('FatigueScore', 20):.0f}%)."
        )

    @staticmethod
    def _urgency(fatigue: float, perf: float, injury: str) -> str:
        if injury == 'Eleve' or (fatigue > 80 and perf < 40):
            return 'IMMEDIATE'
        if fatigue > 70 or perf < 50:
            return 'HAUTE'
        return 'NORMALE'
