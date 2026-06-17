# Football Decision Intelligence AI

Assistant tactique intelligent pour aider un coach à prendre des décisions avant, pendant et après un match.

## Architecture

```
Vidéo Match → Computer Vision → Tracking Joueurs → Extraction Statistiques
→ Performance Intelligence → Injury Risk → Decision Engine
→ Tactical Simulation → Reinforcement Learning → Recommandations → Dashboard Coach
```

## Structure du projet

```
football_analysis/
├── main.py                          # Pipeline complet (Semaines 1-4)
├── dashboard.py                     # Dashboard Streamlit (streamlit run dashboard.py)
├── data/
│   └── generate_dataset.py          # Génération du dataset joueurs/matchs/adversaire
├── performance/
│   ├── performance_score.py         # Performance Score (0-100)
│   ├── fatigue_score.py             # Fatigue Score (0-100)
│   └── injury_risk.py               # Injury Risk Score (0-100)
├── ml_models/
│   ├── injury_predictor.py          # Random Forest + XGBoost (Semaine 2)
│   └── performance_predictor.py     # MLP + LSTM (Semaine 2)
├── decision_engine/
│   ├── rules_engine.py              # Règles IF-THEN (Semaine 3)
│   ├── substitution_recommender.py  # Recommandation de remplacement
│   ├── tactical_analyzer.py         # Analyse adverse
│   └── tactical_simulator.py        # Simulation 4-3-3 / 4-4-2 / 3-5-2 / 3-4-3
├── reinforcement_learning/
│   ├── football_env.py              # Environnement simulé (Semaine 4)
│   └── q_learning_agent.py          # Agent Q-Learning
├── report/
│   └── report_generator.py          # Rapport automatique JSON + texte
├── trackers/tracker.py              # Détection + tracking (YOLOv8 + ByteTrack)
├── team_assigner/                   # Assignation des équipes par couleur
├── player_ball_assigner/            # Possession du ballon
├── camera_movement_estimator/       # Correction mouvement caméra
├── view_transformer/                # Transformation perspective (pixels → mètres)
├── speed_and_distance_estimator/    # Vitesse et distance par joueur
├── utils/                           # Utilitaires vidéo et bbox
├── stubs/                           # Cache pré-calculé (tracking, caméra, agent RL)
├── input_videos/08fd33_4.mp4        # Vidéo de match
└── requirements.txt
```

## Lancement

```bash
# Pipeline complet
python main.py

# Dashboard Coach
streamlit run dashboard.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Modèles

- **YOLOv8x** : téléchargé automatiquement par `ultralytics` au premier lancement (`yolov8x.pt`)
- **Modèle custom** : placer `best.pt` dans `models/` pour remplacer YOLOv8x

## Livrables

| Livrable | Semaine |
|---|---|
| Dataset football | S1 |
| Vidéo annotée YOLO | S1 |
| Tracking joueurs | S1 |
| Statistiques physiques | S1-S2 |
| Performance / Fatigue / Injury Risk Score | S2 |
| Modèle ML (RF + XGBoost) | S2 |
| Modèle Deep Learning (MLP + LSTM) | S2 |
| Tactical Decision Engine | S3 |
| Analyse adverse | S3 |
| Simulation tactique | S3 |
| Recommandations interprétées | S3 |
| Agent Q-Learning | S4 |
| Dashboard Streamlit | S4 |
| Rapport automatique | S4 |
