import pandas as pd
from typing import List, Dict


class RulesEngine:
    """
    Moteur de règles IF-THEN pour les décisions tactiques en temps réel.

    Règles :
        R1 : Fatigue > 70  ET  Performance < 50  → Remplacement immédiat
        R2 : Fatigue > 80                         → Alerte fatigue élevée
        R3 : InjuryRisk == 'Eleve'               → Alerte risque blessure
        R4 : Performance < 40                     → Sous-performance critique
        R5 : Fatigue > 60  ET  Performance < 60  → Surveillance renforcée
    """

    RULES: List[Dict] = [
        {
            'id': 'R1',
            'label': 'Remplacement recommandé',
            'severity': 'CRITIQUE',
            'condition': lambda r: r.get('FatigueScore', 0) > 70 and r.get('PerformanceScore', 100) < 50,
            'action': 'SUBSTITUTION',
        },
        {
            'id': 'R2',
            'label': 'Alerte fatigue élevée',
            'severity': 'HAUTE',
            'condition': lambda r: r.get('FatigueScore', 0) > 80,
            'action': 'WATCH_FATIGUE',
        },
        {
            'id': 'R3',
            'label': 'Risque de blessure élevé',
            'severity': 'HAUTE',
            'condition': lambda r: r.get('InjuryRisk', '') == 'Eleve',
            'action': 'INJURY_RISK',
        },
        {
            'id': 'R4',
            'label': 'Sous-performance critique',
            'severity': 'MOYENNE',
            'condition': lambda r: r.get('PerformanceScore', 100) < 40,
            'action': 'UNDERPERFORMING',
        },
        {
            'id': 'R5',
            'label': 'Surveillance renforcée',
            'severity': 'BASSE',
            'condition': lambda r: r.get('FatigueScore', 0) > 60 and r.get('PerformanceScore', 100) < 60,
            'action': 'MONITOR',
        },
    ]

    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all rules to each player row.
        Returns a DataFrame with columns: PlayerID, triggered rules (list), highest severity action.
        """
        alerts = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            triggered = []
            for rule in self.RULES:
                try:
                    if rule['condition'](row_dict):
                        triggered.append({
                            'rule_id': rule['id'],
                            'label': rule['label'],
                            'severity': rule['severity'],
                            'action': rule['action'],
                        })
                except Exception:
                    pass
            alerts.append({
                'PlayerID': row_dict.get('PlayerID'),
                'Team': row_dict.get('Team', row_dict.get('team', 0)),
                'FatigueScore': row_dict.get('FatigueScore', 0),
                'PerformanceScore': row_dict.get('PerformanceScore', 0),
                'InjuryRisk': row_dict.get('InjuryRisk', 'N/A'),
                'Alerts': triggered,
                'AlertCount': len(triggered),
                'TopAction': triggered[0]['action'] if triggered else None,
            })
        return pd.DataFrame(alerts)

    def get_substitution_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        results = self.evaluate(df)
        return results[results['TopAction'] == 'SUBSTITUTION'].copy()

    def summary(self, df: pd.DataFrame) -> Dict:
        results = self.evaluate(df)
        return {
            'total_players': len(results),
            'substitution_needed': int((results['TopAction'] == 'SUBSTITUTION').sum()),
            'high_fatigue': int((results['FatigueScore'] > 80).sum()),
            'injury_risk_high': int((results['InjuryRisk'] == 'Eleve').sum()),
            'underperforming': int((results['PerformanceScore'] < 50).sum()),
        }
