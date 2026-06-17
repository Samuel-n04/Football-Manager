import json
import os
from datetime import datetime
from typing import Dict, Any, List


class ReportGenerator:
    """
    Génère un rapport de match structuré (JSON + texte lisible) couvrant :
        - résumé du match
        - performances individuelles
        - décisions recommandées et explications
        - erreurs tactiques identifiées
        - résultats du Reinforcement Learning
    """

    def generate(
        self,
        player_stats,          # pd.DataFrame
        substitutions: List[Dict],
        tactical_recommendation: Dict,
        opponent_analysis: Dict,
        rl_summary: Dict,
        match_context: Dict | None = None,
    ) -> Dict[str, Any]:
        match_context = match_context or {'team': 'Équipe A', 'opponent': 'Équipe B', 'score': '?-?'}
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

        report = {
            'timestamp': timestamp,
            'match': match_context,
            'summary': self._build_summary(player_stats, match_context),
            'top_performers': self._top_performers(player_stats),
            'underperformers': self._underperformers(player_stats),
            'fatigue_report': self._fatigue_report(player_stats),
            'injury_risk_report': self._injury_report(player_stats),
            'substitution_decisions': self._format_subs(substitutions),
            'tactical_decision': self._format_tactic(tactical_recommendation),
            'opponent_analysis': self._format_opponent(opponent_analysis),
            'reinforcement_learning': rl_summary,
            'tactical_errors': self._identify_tactical_errors(player_stats, substitutions),
            'conclusion': self._conclusion(player_stats, rl_summary),
        }
        return report

    def save_json(self, report: Dict, path: str = 'output_videos/match_report.json') -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        return path

    def save_text(self, report: Dict, path: str = 'output_videos/match_report.txt') -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        lines = self._render_text(report)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        return path

    # ── private formatters ────────────────────────────────────────────────────

    def _build_summary(self, df, ctx: Dict) -> str:
        try:
            avg_perf = df['PerformanceScore'].mean()
            avg_fat = df['FatigueScore'].mean()
            n = len(df)
        except Exception:
            return "Données insuffisantes pour générer un résumé."
        return (
            f"Match : {ctx.get('team','Équipe A')} vs {ctx.get('opponent','Équipe B')} "
            f"— Score : {ctx.get('score','?-?')}. "
            f"{n} joueurs analysés. Performance moyenne : {avg_perf:.1f}/100. "
            f"Fatigue moyenne : {avg_fat:.1f}/100."
        )

    def _top_performers(self, df) -> List[Dict]:
        try:
            top = df.nlargest(3, 'PerformanceScore')[['PlayerID', 'PerformanceScore', 'FatigueScore']]
            return top.round(1).to_dict(orient='records')
        except Exception:
            return []

    def _underperformers(self, df) -> List[Dict]:
        try:
            under = df[df['PerformanceScore'] < 50][['PlayerID', 'PerformanceScore', 'FatigueScore', 'InjuryRisk']]
            return under.round(1).to_dict(orient='records')
        except Exception:
            return []

    def _fatigue_report(self, df) -> Dict:
        try:
            return {
                'avg_fatigue': round(float(df['FatigueScore'].mean()), 1),
                'max_fatigue': round(float(df['FatigueScore'].max()), 1),
                'players_above_70': int((df['FatigueScore'] > 70).sum()),
                'most_fatigued_player': int(df.loc[df['FatigueScore'].idxmax(), 'PlayerID']),
            }
        except Exception:
            return {}

    def _injury_report(self, df) -> Dict:
        try:
            counts = df['InjuryRisk'].value_counts().to_dict()
            return {
                'Faible': counts.get('Faible', 0),
                'Moyen': counts.get('Moyen', 0),
                'Eleve': counts.get('Eleve', 0),
                'high_risk_players': df[df['InjuryRisk'] == 'Eleve']['PlayerID'].tolist(),
            }
        except Exception:
            return {}

    def _format_subs(self, subs: List[Dict]) -> List[Dict]:
        formatted = []
        for s in subs:
            formatted.append({
                'sortant': f"Joueur #{s.get('out_player_id', '?')} ({s.get('out_position', '?')})",
                'entrant': f"Joueur #{s.get('in_player_id', '?')} ({s.get('in_position', '?')})",
                'urgence': s.get('urgency', 'NORMALE'),
                'explication': s.get('reason', ''),
                'score_compatibilite': s.get('tactical_score', 0),
            })
        return formatted

    def _format_tactic(self, tactic: Dict) -> Dict:
        return {
            'formation_recommandee': tactic.get('recommended_formation', 'N/A'),
            'score_global': tactic.get('global_score', 0),
            'possession_estimee': f"{tactic.get('possession', 0):.1f}%",
            'xG_estime': tactic.get('xG', 0),
            'risque_defensif': tactic.get('defensive_risk', 0),
            'description': tactic.get('description', ''),
        }

    def _format_opponent(self, opp: Dict) -> Dict:
        return {
            'resume': opp.get('summary', ''),
            'zone_vulnerable': opp.get('weak_zones', {}).get('weakest_zone', 'N/A'),
            'joueurs_faibles': len(opp.get('weak_players', [])),
            'failles_defensives': len(opp.get('defensive_weaknesses', [])),
            'recommandations': opp.get('tactical_recommendations', []),
        }

    def _identify_tactical_errors(self, df, subs: List[Dict]) -> List[str]:
        errors = []
        try:
            if (df['FatigueScore'] > 80).sum() > 3:
                errors.append("Trop de joueurs épuisés en fin de match — remplacements tardifs.")
            if (df['PerformanceScore'] < 40).sum() > 2:
                errors.append("Plusieurs joueurs en sous-performance critique non remplacés.")
            if not subs:
                high_risk = (df['InjuryRisk'] == 'Eleve').sum()
                if high_risk > 0:
                    errors.append(f"{high_risk} joueur(s) à haut risque de blessure sans remplacement effectué.")
        except Exception:
            pass
        return errors if errors else ["Aucune erreur tactique majeure identifiée."]

    def _conclusion(self, df, rl: Dict) -> str:
        try:
            win_rate = rl.get('win_rate', 'N/A')
            best_action = rl.get('most_chosen_action', 'N/A')
            avg_perf = df['PerformanceScore'].mean()
            return (
                f"Performance collective : {avg_perf:.1f}/100. "
                f"L'agent RL recommande principalement l'action « {best_action} » "
                f"avec un taux de victoire simulé de {win_rate}%. "
                f"Appliquer ces recommandations lors des prochains matchs."
            )
        except Exception:
            return "Rapport généré. Consultez les sections détaillées ci-dessus."

    @staticmethod
    def _render_text(report: Dict) -> List[str]:
        sep = '=' * 70
        lines = [
            sep,
            f"  RAPPORT DE MATCH — {report.get('timestamp', '')}",
            sep,
            '',
            f"  {report.get('summary', '')}",
            '',
        ]

        for section, title in [
            ('top_performers',       'MEILLEURES PERFORMANCES'),
            ('underperformers',      'SOUS-PERFORMANCES'),
            ('fatigue_report',       'BILAN FATIGUE'),
            ('injury_risk_report',   'RISQUES DE BLESSURE'),
            ('substitution_decisions', 'DÉCISIONS DE REMPLACEMENT'),
            ('tactical_decision',    'TACTIQUE RECOMMANDÉE'),
            ('opponent_analysis',    'ANALYSE ADVERSE'),
            ('tactical_errors',      'ERREURS TACTIQUES'),
            ('reinforcement_learning', 'REINFORCEMENT LEARNING'),
            ('conclusion',           'CONCLUSION'),
        ]:
            lines += ['', f'  {title}', '-' * 70]
            data = report.get(section)
            if isinstance(data, list):
                for item in data:
                    lines.append(f"  • {item}")
            elif isinstance(data, dict):
                for k, v in data.items():
                    lines.append(f"  {k}: {v}")
            else:
                lines.append(f"  {data}")

        lines += ['', sep]
        return lines
