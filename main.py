import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np

from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

from data.generate_dataset import generate_player_dataset, generate_opponent_data, generate_match_events
from performance.performance_score import PerformanceScoreCalculator
from performance.fatigue_score import FatigueScoreCalculator
from performance.injury_risk import InjuryRiskCalculator
from ml_models.injury_predictor import InjuryPredictor
from ml_models.performance_predictor import PerformancePredictor
from decision_engine.rules_engine import RulesEngine
from decision_engine.substitution_recommender import SubstitutionRecommender
from decision_engine.tactical_analyzer import TacticalAnalyzer
from decision_engine.tactical_simulator import TacticalSimulator
from reinforcement_learning.q_learning_agent import QLearningAgent
from report.report_generator import ReportGenerator


def run_computer_vision(video_path: str, model_path: str) -> dict:
    """Semaine 1 — Détection YOLOv8, tracking ByteTrack, indicateurs physiques."""
    print(f"[CV] Lecture vidéo … (modèle: {model_path})")
    video_frames = read_video(video_path)

    tracker = Tracker(model_path)
    tracks = tracker.get_object_tracks(
        video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl'
    )
    tracker.add_position_to_tracks(tracks)

    camera_est = CameraMovementEstimator(video_frames[0])
    cam_move = camera_est.get_camera_movement(
        video_frames, read_from_stub=True, stub_path='stubs/camera_movement_stub.pkl'
    )
    camera_est.add_adjust_positions_to_tracks(tracks, cam_move)

    view = ViewTransformer()
    view.add_transformed_position_to_tracks(tracks)

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    speed_est = SpeedAndDistance_Estimator()
    speed_est.add_speed_and_distance_to_tracks(tracks)

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    for frame_num, player_track in enumerate(tracks['players']):
        for pid, info in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], info['bbox'], pid)
            tracks['players'][frame_num][pid]['team'] = team
            tracks['players'][frame_num][pid]['team_color'] = team_assigner.team_colors[team]

    ball_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned = ball_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned != -1:
            tracks['players'][frame_num][assigned]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)
    team_ball_control = np.array(team_ball_control)

    # Annotate output video
    out_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    out_frames = camera_est.draw_camera_movement(out_frames, cam_move)
    speed_est.draw_speed_and_distance(out_frames, tracks)
    save_video(out_frames, 'output_videos/output_video.avi')
    print("[CV] Output video saved to output_videos/output_video.avi")

    # Build player DataFrame from tracking data
    player_df = PerformanceScoreCalculator.from_tracks(tracks)
    return {'tracks': tracks, 'team_ball_control': team_ball_control, 'player_df': player_df}


def run_performance_intelligence(player_df, dataset_df):
    """Week 2 — Performance, Fatigue, Injury Risk, ML models."""
    print("\n[PERF] Computing scores …")

    # Merge video-extracted features with dataset (use dataset stats for tactical cols)
    if player_df is not None and len(player_df) > 0:
        cols_to_add = ['PassAccuracy', 'Shots', 'Recoveries', 'PressingActions',
                       'Fouls', 'YellowCards', 'Accelerations', 'Position']
        last_match = dataset_df[dataset_df['MatchID'] == dataset_df['MatchID'].max()].copy()
        # Fill in tactical columns from dataset (video tracks physical cols)
        for col in cols_to_add:
            if col not in player_df.columns and col in last_match.columns:
                player_df = player_df.merge(
                    last_match[['PlayerID', col]], on='PlayerID', how='left'
                )
        df = player_df.copy()
    else:
        df = dataset_df[dataset_df['MatchID'] == dataset_df['MatchID'].max()].copy()

    df = df.fillna(0)

    perf_calc = PerformanceScoreCalculator()
    fat_calc = FatigueScoreCalculator()
    inj_calc = InjuryRiskCalculator()

    df = perf_calc.calculate(df)
    df = fat_calc.calculate(df)
    df = inj_calc.calculate(df)
    df = perf_calc.classify(df)
    df = fat_calc.classify(df)

    print(f"[PERF] Players analysed: {len(df)}")
    print(f"[PERF] Avg performance: {df['PerformanceScore'].mean():.1f}")
    print(f"[PERF] Avg fatigue:     {df['FatigueScore'].mean():.1f}")

    # Under-performance detection
    rules = RulesEngine()
    summary = rules.summary(df)
    print(f"[RULES] {summary}")

    # ML — Injury risk prediction
    print("\n[ML] Training Injury Predictor (RF + XGBoost) …")
    inj_pred = InjuryPredictor()
    inj_results = inj_pred.train(dataset_df)
    print(f"[ML] RF accuracy: {inj_results['rf_accuracy']:.4f} | XGB: {inj_results['xgb_accuracy']:.4f}")
    print(f"[ML] Best model: {inj_results['best_model']}")

    # ML — Performance prediction (MLP + LSTM)
    print("\n[ML] Training Performance Predictor (MLP + LSTM) …")
    perf_pred_model = PerformancePredictor()
    perf_results = perf_pred_model.train(dataset_df)
    print(f"[ML] MLP MAE: {perf_results.get('mlp_mae', 'N/A')} | LSTM MAE: {perf_results.get('lstm_mae', 'N/A')}")
    print(f"[ML] Best model: {perf_results.get('best_model', 'N/A')}")

    return df, inj_pred, perf_pred_model


def run_decision_engine(df, dataset_df):
    """Week 3 — Tactical decisions, opponent analysis, formation simulation."""
    print("\n[DECISION] Running decision engine …")

    # Split field players / bench
    if 'Team' not in df.columns:
        df['Team'] = (df['PlayerID'] <= 11).map({True: 1, False: 2})

    team1 = df[df['Team'] == 1].copy()
    bench_df = dataset_df[dataset_df['MatchID'] == dataset_df['MatchID'].max() - 1].copy()
    bench_df['Team'] = (bench_df['PlayerID'] <= 11).map({True: 1, False: 2})
    bench1 = bench_df[bench_df['Team'] == 1].copy()

    # Substitution recommendations
    recommender = SubstitutionRecommender()
    subs = recommender.recommend(team1, bench1, max_subs=3)
    print(f"[DECISION] {len(subs)} substitution(s) recommended.")
    for s in subs:
        print(f"  → OUT #{s['out_player_id']} | IN #{s['in_player_id']} | {s['urgency']} | {s['reason'][:60]}…")

    # Opponent analysis
    opp_df = generate_opponent_data(n_players=11)
    analyzer = TacticalAnalyzer()
    opp_analysis = analyzer.analyze_opponent(opp_df)
    print(f"[DECISION] Opponent: {opp_analysis['summary']}")

    # Formation simulation
    simulator = TacticalSimulator()
    tac_rec = simulator.recommend_formation(team1, {'minute': 60, 'score_diff': 0})
    print(f"[DECISION] Recommended formation: {tac_rec['recommended_formation']} "
          f"(score {tac_rec['global_score']})")

    return subs, opp_analysis, tac_rec


def run_reinforcement_learning(n_episodes: int = 2000):
    """Week 4 — Q-Learning agent training."""
    print(f"\n[RL] Training Q-Learning agent ({n_episodes} episodes) …")
    agent = QLearningAgent()
    rl_results = agent.train(n_episodes=n_episodes)
    print(f"[RL] Win rate (last 200 ep): {rl_results['win_rate']:.1f}%")
    print(f"[RL] Most recommended action: {rl_results['most_chosen_action']}")
    agent.save()
    return agent, rl_results


def generate_report(df, subs, tac_rec, opp_analysis, rl_results, score_diff=0):
    """Generate and save the post-match report."""
    print("\n[REPORT] Generating match report …")
    gen = ReportGenerator()
    report = gen.generate(
        player_stats=df,
        substitutions=subs,
        tactical_recommendation=tac_rec,
        opponent_analysis=opp_analysis,
        rl_summary=rl_results,
        match_context={'team': 'Équipe A', 'opponent': 'Équipe B', 'score': f'{score_diff:+d}'},
    )
    json_path = gen.save_json(report)
    txt_path = gen.save_text(report)
    print(f"[REPORT] JSON saved: {json_path}")
    print(f"[REPORT] Text saved: {txt_path}")
    return report


def main():
    print("=" * 60)
    print("  Football Decision Intelligence AI — Pipeline")
    print("=" * 60)

    # ── Week 1: Computer Vision ───────────────────────────────────────────
    video_path = 'input_videos/08fd33_4.mp4'
    cv_player_df = None
    model_path = 'yolov8x.pt'  # YOLOv8 — téléchargé automatiquement par ultralytics

    if os.path.exists(video_path):
        cv_result = run_computer_vision(video_path, model_path)
        cv_player_df = cv_result['player_df']
    else:
        print(f"[CV] Video not found. Skipping CV pipeline.")
        print(f"     Place your match video at: {video_path}")

    # ── Week 2: Performance Intelligence ─────────────────────────────────
    print("\n[DATA] Generating player dataset …")
    dataset_df = generate_player_dataset(n_players=22, n_matches=15)
    print(f"[DATA] Dataset shape: {dataset_df.shape}")

    df, inj_pred, perf_pred_model = run_performance_intelligence(cv_player_df, dataset_df)

    # ── Week 3: Tactical Decision Engine ──────────────────────────────────
    subs, opp_analysis, tac_rec = run_decision_engine(df, dataset_df)

    # ── Week 4: Reinforcement Learning ────────────────────────────────────
    agent, rl_results = run_reinforcement_learning(n_episodes=2000)

    # ── Post-match Report ─────────────────────────────────────────────────
    report = generate_report(df, subs, tac_rec, opp_analysis, rl_results)

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("  Launch the dashboard with:  streamlit run dashboard.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
