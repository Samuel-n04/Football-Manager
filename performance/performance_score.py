import pandas as pd
import numpy as np


class PerformanceScoreCalculator:
    """
    Calcule le Performance Score (0-100) selon la formule du cahier des charges :
        30 % passes + 20 % tirs + 20 % récupération + 20 % pressing + 10 % discipline
    """

    WEIGHTS = {
        'passes': 0.30,
        'shots': 0.20,
        'recoveries': 0.20,
        'pressing': 0.20,
        'discipline': 0.10,
    }

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df with a PerformanceScore column (re-calculated from raw stats)."""
        df = df.copy()
        df['PerformanceScore'] = df.apply(self._score_row, axis=1).round(2)
        return df

    def _score_row(self, row) -> float:
        pass_score = float(row.get('PassAccuracy', 70))
        shot_score = min(float(row.get('Shots', 0)) / 8 * 100, 100)
        rec_score = min(float(row.get('Recoveries', 0)) / 18 * 100, 100)
        press_score = min(float(row.get('PressingActions', 0)) / 25 * 100, 100)
        disc_score = max(0.0, 100 - float(row.get('Fouls', 0)) * 10
                         - float(row.get('YellowCards', 0)) * 20)

        return (self.WEIGHTS['passes'] * pass_score
                + self.WEIGHTS['shots'] * shot_score
                + self.WEIGHTS['recoveries'] * rec_score
                + self.WEIGHTS['pressing'] * press_score
                + self.WEIGHTS['discipline'] * disc_score)

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add PerformanceLevel column: Faible / Moyen / Bon / Excellent."""
        df = df.copy()
        bins = [0, 40, 60, 80, 100]
        labels = ['Faible', 'Moyen', 'Bon', 'Excellent']
        df['PerformanceLevel'] = pd.cut(df['PerformanceScore'], bins=bins, labels=labels)
        return df

    def detect_underperformers(self, df: pd.DataFrame, threshold: float = 50) -> pd.DataFrame:
        """Return only rows where PerformanceScore < threshold."""
        return df[df['PerformanceScore'] < threshold].copy()

    @staticmethod
    def from_tracks(tracks: dict, frame_rate: int = 24) -> pd.DataFrame:
        """
        Build a minimal player DataFrame from tracker output.
        Fills tactical stats (passes, shots …) with neutral defaults since
        those are not observable from video alone.
        """
        player_stats: dict = {}

        for frame_data in tracks.get('players', []):
            for pid, info in frame_data.items():
                if pid not in player_stats:
                    player_stats[pid] = {
                        'PlayerID': pid,
                        'Team': info.get('team', 0),
                        'SpeedSamples': [],
                        'Distance': 0.0,
                        'SprintCount': 0,
                        'PassAccuracy': 75.0,
                        'Shots': 0,
                        'Recoveries': 0,
                        'PressingActions': 0,
                        'Fouls': 0,
                        'YellowCards': 0,
                        'MinutesPlayed': len(tracks.get('players', [])) / (frame_rate * 60),
                    }

                speed = info.get('speed')
                if speed is not None:
                    player_stats[pid]['SpeedSamples'].append(speed)
                    if speed > 20:
                        player_stats[pid]['SprintCount'] += 1

                dist = info.get('distance')
                if dist is not None:
                    player_stats[pid]['Distance'] = max(player_stats[pid]['Distance'], dist)

        rows = []
        for pid, s in player_stats.items():
            samples = s.pop('SpeedSamples')
            s['SpeedAvg'] = float(np.mean(samples)) if samples else 0.0
            s['SpeedMax'] = float(np.max(samples)) if samples else 0.0
            rows.append(s)

        return pd.DataFrame(rows)
