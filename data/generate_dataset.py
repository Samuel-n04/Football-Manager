import pandas as pd
import numpy as np

POSITIONS = ['GK', 'CB', 'LB', 'RB', 'CDM', 'CM', 'CAM', 'LW', 'RW', 'CF', 'ST']


def generate_player_dataset(n_players=22, n_matches=15, seed=42):
    """Synthetic player dataset covering physical and tactical KPIs across multiple matches."""
    np.random.seed(seed)
    rows = []
    for match_id in range(n_matches):
        for player_id in range(1, n_players + 1):
            position = np.random.choice(POSITIONS)
            minutes_played = np.random.randint(45, 95)
            speed_avg = np.random.uniform(8, 28)
            speed_max = speed_avg + np.random.uniform(5, 15)
            sprint_count = np.random.randint(5, 45)
            distance = np.random.uniform(4500, 13000)
            accelerations = np.random.randint(15, 110)
            pass_accuracy = np.random.uniform(58, 97)
            shots = np.random.randint(0, 9)
            recoveries = np.random.randint(0, 18)
            pressing_actions = np.random.randint(0, 25)
            fouls = np.random.randint(0, 6)
            yellow_cards = np.random.randint(0, 2)

            fatigue = _compute_fatigue(distance, sprint_count, speed_avg, minutes_played)
            performance = _compute_performance(pass_accuracy, shots, recoveries, pressing_actions, fouls, yellow_cards)
            injury_score = _compute_injury_score(fatigue, minutes_played, accelerations, distance)

            if injury_score < 33:
                injury_risk = 'Faible'
            elif injury_score < 66:
                injury_risk = 'Moyen'
            else:
                injury_risk = 'Eleve'

            rows.append({
                'PlayerID': player_id,
                'MatchID': match_id,
                'Position': position,
                'MinutesPlayed': minutes_played,
                'SpeedAvg': round(speed_avg, 2),
                'SpeedMax': round(speed_max, 2),
                'SprintCount': sprint_count,
                'Distance': round(distance, 2),
                'Accelerations': accelerations,
                'PassAccuracy': round(pass_accuracy, 2),
                'Shots': shots,
                'Recoveries': recoveries,
                'PressingActions': pressing_actions,
                'Fouls': fouls,
                'YellowCards': yellow_cards,
                'FatigueScore': round(fatigue, 2),
                'PerformanceScore': round(performance, 2),
                'InjuryRiskScore': round(injury_score, 2),
                'InjuryRisk': injury_risk,
            })
    return pd.DataFrame(rows)


def generate_match_events(n_events=200, seed=42):
    """Synthetic in-match events (goals, fouls, cards, substitutions)."""
    np.random.seed(seed)
    event_types = ['goal', 'foul', 'yellow_card', 'corner', 'offside', 'substitution', 'save']
    events = []
    for _ in range(n_events):
        minute = np.random.randint(1, 91)
        team = np.random.randint(1, 3)
        player_id = np.random.randint(1, 23)
        event_type = np.random.choice(event_types, p=[0.05, 0.25, 0.08, 0.15, 0.10, 0.07, 0.30])
        events.append({'Minute': minute, 'Team': team, 'PlayerID': player_id, 'EventType': event_type})
    return pd.DataFrame(events).sort_values('Minute').reset_index(drop=True)


def generate_opponent_data(n_players=11, seed=99):
    """Synthetic opponent team dataset for pre-match analysis."""
    np.random.seed(seed)
    rows = []
    for player_id in range(1, n_players + 1):
        position = np.random.choice(POSITIONS)
        pass_accuracy = np.random.uniform(55, 95)
        speed_avg = np.random.uniform(8, 26)
        sprint_count = np.random.randint(5, 40)
        distance = np.random.uniform(4000, 12000)
        shots = np.random.randint(0, 8)
        recoveries = np.random.randint(0, 15)
        pressing_actions = np.random.randint(0, 20)
        fouls = np.random.randint(0, 5)
        yellow_cards = np.random.randint(0, 2)
        fatigue = np.random.uniform(20, 80)

        performance = _compute_performance(pass_accuracy, shots, recoveries, pressing_actions, fouls, yellow_cards)

        # Zone tendency: left, center, right (percentage of actions in each zone)
        zone_split = np.random.dirichlet([1, 1, 1])

        rows.append({
            'PlayerID': player_id,
            'Position': position,
            'PassAccuracy': round(pass_accuracy, 2),
            'SpeedAvg': round(speed_avg, 2),
            'SprintCount': sprint_count,
            'Distance': round(distance, 2),
            'Shots': shots,
            'Recoveries': recoveries,
            'PressingActions': pressing_actions,
            'Fouls': fouls,
            'YellowCards': yellow_cards,
            'FatigueScore': round(fatigue, 2),
            'PerformanceScore': round(performance, 2),
            'ZoneLeft': round(zone_split[0] * 100, 1),
            'ZoneCenter': round(zone_split[1] * 100, 1),
            'ZoneRight': round(zone_split[2] * 100, 1),
        })
    return pd.DataFrame(rows)


# ── internal helpers ─────────────────────────────────────────────────────────

def _compute_fatigue(distance, sprint_count, speed_avg, minutes_played):
    d_norm = min(distance / 13000 * 100, 100)
    s_norm = min(sprint_count / 45 * 100, 100)
    v_norm = min(speed_avg / 28 * 100, 100)
    t_norm = min(minutes_played / 95 * 100, 100)
    return 0.35 * d_norm + 0.30 * s_norm + 0.20 * v_norm + 0.15 * t_norm


def _compute_performance(pass_accuracy, shots, recoveries, pressing, fouls, yellow_cards):
    pass_score = pass_accuracy
    shot_score = min(shots / 8 * 100, 100)
    rec_score = min(recoveries / 18 * 100, 100)
    press_score = min(pressing / 25 * 100, 100)
    disc_score = max(0, 100 - fouls * 10 - yellow_cards * 20)
    return (0.30 * pass_score + 0.20 * shot_score + 0.20 * rec_score
            + 0.20 * press_score + 0.10 * disc_score)


def _compute_injury_score(fatigue, minutes_played, accelerations, distance):
    t_norm = min(minutes_played / 95 * 100, 100)
    a_norm = min(accelerations / 110 * 100, 100)
    d_norm = min(distance / 13000 * 100, 100)
    return 0.40 * fatigue + 0.20 * t_norm + 0.20 * a_norm + 0.20 * d_norm
