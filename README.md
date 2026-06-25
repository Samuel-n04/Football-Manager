# Football Decision Intelligence AI — Football Manager

Système d'analyse vidéo et d'aide à la décision tactique pour l'entraîneur de football.  
Le projet combine le tracking de joueurs par computer vision (YOLO + ByteTrack) avec un moteur de décision basé sur des modèles ML et des règles tactiques, le tout présenté dans un dashboard Streamlit interactif.

---

## Rapport du projet

### Objectif

Fournir à un entraîneur un outil complet qui :
- Analyse automatiquement une vidéo de match (tracking joueurs, ballon, équipes)
- Calcule des indicateurs de performance, fatigue et risque de blessure
- Recommande des substitutions, formations et ajustements tactiques
- Génère un rapport post-match exploitable

### Architecture

```
Football-Manager/
├── app.py                        # Dashboard Streamlit (point d'entrée)
├── tracking_foot.py              # Pipeline de tracking vidéo YOLO + ByteTrack
├── tracking/
│   ├── config.py                 # Paramètres globaux (seuils, codec, YOLO)
│   ├── state.py                  # État partagé du tracking (positions, passes, possession)
│   ├── reid.py                   # Ré-identification des joueurs entre frames
│   ├── team.py                   # Détection des équipes par couleur de maillot
│   ├── field.py                  # Masque du terrain (exclusion hors-jeu)
│   ├── passes.py                 # Détection automatique des passes
│   ├── speed.py                  # Estimation de vitesse (correction mouvement caméra)
│   ├── render.py                 # Rendu des annotations sur la vidéo
│   └── stats.py                  # Calcul et sauvegarde des statistiques JSON
├── input_videos/                 # Vidéos MP4/AVI à analyser (à placer ici)
├── output_videos/                # Vidéos annotées et fichiers stats générés
└── requirements.txt              # Dépendances Python
```

Les modules ML (`decision_engine/`, `ml_models/`, `performance/`, `data/`, `report/`) 

### Fonctionnalités

| Module | Description |
|---|---|
| Tracking YOLO | Détection joueurs + ballon avec YOLOv8m, tracker ByteTrack |
| Ré-identification | Maintien de l'identité des joueurs entre les frames (ReID couleur + position) |
| Équipes | Classification automatique par couleur de maillot (K-Means HSV) |
| Passes | Détection automatique des passes (vitesse balle + proximité joueur) |
| Vitesse | Estimation en km/h avec correction du mouvement de caméra (optical flow) |
| Score joueur | Performance [0-100] : vitesse (40%) + passes (30%) + distance (30%) |
| Fatigue | Calcul 0-100 : distance (35%) + sprints (30%) + vitesse (20%) + temps jeu (15%) |
| Risque blessure | Score 0-100 + classification (Faible / Moyen / Élevé) via Random Forest + XGBoost |
| Substitutions | Recommandations avec score de compatibilité tactique (fraîcheur + perf + poste) |
| Tactique | Simulation de 4 formations avec score global (possession, pressing, xG, risque défensif) |
| Rapport | Génération automatique du rapport post-match (JSON + affichage dashboard) |

### Dashboard — 4 onglets

1. **Avant Match** — Formation recommandée, joueur clé adverse, zones vulnérables, prédictions de performance
2. **Pendant Match** — KPIs en temps réel, scatter fatigue/performance, alertes (CRITIQUE / HAUTE / BASSE), remplacements proposés
3. **Après Match** — Rapport complet, meilleures/pires performances, erreurs tactiques, comparaison modèles ML, export JSON
4. **Vidéo annotée** — Lecture de la vidéo de tracking avec conversion H264 automatique

---

## Prérequis

- Python **3.10+**
- GPU AMD (ROCm) ou CPU (le tracking sera plus lent sur CPU)
- Les deux dossiers doivent être au même niveau :
  ```
  Football Decision Intelligence AI/
  ├── Football-Manager/     ← ce projet
  └── football_analysis/    ← modules ML (decision_engine, ml_models, etc.)
  ```

---

## Installation

### 1. Cloner / télécharger le projet

Placer le dossier `Football-Manager` dans :
```
C:\Users\<user>\Desktop\Football Decision Intelligence AI\
```

### 2. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# ou
source venv/bin/activate     # Linux / macOS
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

> **Note GPU AMD (ROCm)** : la première ligne de `requirements.txt` pointe vers ROCm 6.3.  
> Pour un GPU NVIDIA, supprimez `--extra-index-url https://download.pytorch.org/whl/rocm6.3` et installez PyTorch depuis [pytorch.org](https://pytorch.org/get-started/locally/).  
> Pour CPU uniquement, gardez le fichier tel quel.

### 4. Télécharger le modèle YOLO

Au premier lancement, `ultralytics` télécharge automatiquement `yolov8m.pt` (~52 MB).  
Si le réseau est limité, téléchargez-le manuellement et placez-le à la racine du projet :

```bash
# Téléchargement automatique via Python
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
```

### 5. (Optionnel mais recommandé) Installer FFmpeg

FFmpeg permet la conversion H264 automatique pour lire la vidéo annotée dans le navigateur.

```bash
# Windows (via winget)
winget install Gyan.FFmpeg

# Vérification
ffmpeg -version
```

Sans FFmpeg, la vidéo annotée sera disponible uniquement en téléchargement.

---

## Lancer le projet

### Dashboard Streamlit

```bash
cd "C:\Users\<user>\Desktop\Football Decision Intelligence AI\Football-Manager"
streamlit run app.py
```

L'interface s'ouvre automatiquement sur [http://localhost:8501](http://localhost:8501).

### Tracking seul (sans dashboard)

```bash
python tracking_foot.py input_videos/votre_video.mp4
```

Les résultats sont sauvegardés dans `output_videos/` :
- `votre_video_tracking.mp4` — vidéo annotée
- `votre_video_stats.json` — statistiques détaillées

---

## Utilisation du dashboard

### Étape 1 — Choisir une vidéo

- Sélectionnez une vidéo dans la liste déroulante de la barre latérale
- **OU** uploadez directement une vidéo (MP4 / AVI / MOV)

### Étape 2 — Lancer le tracking

Cliquez sur **▶ Lancer le tracking**.

> Le traitement prend environ :
> - **GPU** : 5-15 minutes pour une vidéo de 10 min
> - **CPU** : 30-120 minutes pour une vidéo de 10 min
>
> Ne cliquez pas "Arrêter" avant la fin — le fichier vidéo serait corrompu.

Une barre de progression indique l'avancement frame par frame.

### Étape 3 — Explorer l'analyse

Une fois le tracking terminé, les 4 onglets se remplissent automatiquement :

| Onglet | Ce qu'on y trouve |
|---|---|
| Avant Match | Tactique recommandée, adversaire, risques |
| Pendant Match | Fatigue, alertes, remplacements (ajuster la minute + score dans la sidebar) |
| Après Match | Rapport complet + export JSON |
| Vidéo annotée | Lecture de la vidéo avec annotations |

---

## Structure des statistiques générées

```json
{
  "video": "input_videos/match.mp4",
  "duree_secondes": 30.0,
  "frames_traitees": 750,
  "total_passes": 12,
  "possession_equipes": { "equipe_1": 54.3, "equipe_2": 45.7 },
  "joueurs": {
    "1": {
      "player_id": 1,
      "equipe": 1,
      "poste": "Milieu",
      "score": 67.4,
      "vitesse_moyenne_kmh": 8.2,
      "distance_m": 245.0,
      "passes_tentees": 5,
      "passes_reussies": 4,
      "taux_reussite_passes": 80.0,
      "possession_pct": 12.3,
      "sous_performance": false,
      "classement": 2
    }
  },
  "substitutions_recommandees": { ... }
}
```

---

## Problèmes courants

| Problème | Solution |
|---|---|
| `moov atom not found` sur la vidéo | Le tracking a été interrompu trop tôt — relancez et laissez terminer |
| Vidéo annotée illisible dans le navigateur | Installez FFmpeg (`winget install Gyan.FFmpeg`) puis relancez le tracking |
| `Aucun GPU — utilisation CPU` | Normal sans GPU AMD/ROCm. Le tracking sera lent mais fonctionnel |
| Erreur import `decision_engine` | Vérifiez que le dossier `football_analysis/` est au même niveau que `Football-Manager/` |
| `yolov8m.pt` non trouvé | Lancez `python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"` |
| Streamlit affiche une page blanche | Vérifiez que toutes les dépendances sont installées : `pip install -r requirements.txt` |

---

## Dépendances principales

| Bibliothèque | Rôle |
|---|---|
| `streamlit` | Dashboard interactif |
| `ultralytics` (YOLOv8) | Détection objets (joueurs, ballon) |
| `opencv-python` | Traitement vidéo frame par frame |
| `torch` + `torchvision` | Backend deep learning |
| `plotly` | Graphiques interactifs |
| `scikit-learn` | Random Forest (risque blessure) |
| `xgboost` | XGBoost (risque blessure) |
| `pandas` / `numpy` | Manipulation des données |
| `scipy` | Calculs scientifiques |
