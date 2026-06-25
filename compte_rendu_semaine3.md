# Compte Rendu — Semaine 3
## Projet : Football Decision Intelligence AI — Groupe 2

**Date :** 19 juin 2026
**Membres du groupe :**
- PIERRE Jean-Samuel
- CHAUVET Darren
- SASSI Ismail
- DAO Trung-Hieu
- COULIBALY Bourama

---

## 1. Rappel des objectifs — Semaine 3

Selon le sujet, la semaine 3 est dédiée à la construction du **Tactical Decision Engine**. Les tâches attendues sont :

| Tâche | Description |
|-------|-------------|
| 3.1 | Construction des règles de décision (SI Fatigue > 70 ET Performance < 50 ALORS remplacement) |
| 3.2 | Recommandation de remplacement (quel joueur sortir, lequel faire entrer, pourquoi) |
| 3.3 | Tactical Compatibility Score (compatibilité tactique, fraîcheur physique, complémentarité) |
| 3.4 | Analyse adverse (identifier joueur faible, zone faible, faiblesse défensive) |
| 3.5 | Simulation tactique (comparer 4-3-3, 4-4-2, 3-5-2, 3-4-3) |
| 3.6 | Interprétation des décisions (justifier chaque recommandation avec les métriques) |

**Livrables attendus fin S3 :** moteur de décision, recommandations tactiques, analyse adverse, simulation tactique, explications des recommandations.

---

## 2. Répartition du travail

### PIERRE Jean-Samuel — Réorganisation du projet
Jean-Samuel s'est chargé de la réorganisation structurelle des fichiers du projet. Le dépôt a été nettoyé et les répertoires ont été mis en place (`input_videos/`, `output_videos/`), permettant une meilleure lisibilité et une collaboration plus fluide au sein du groupe.

### SASSI Ismail — Veille et prototype de référence
Ismail a effectué une veille approfondie sur des projets similaires au nôtre (analyse vidéo football + dashboard IA). Il a identifié un prototype fonctionnel proche de nos objectifs et a réussi à en reproduire une version adaptée à notre contexte, notamment la partie interface Streamlit. Ce travail a servi de base concrète pour la suite.

### CHAUVET Darren — Adaptation du Streamlit et optimisation du tracking
Darren a récupéré la partie Streamlit du prototype réalisé par Ismail et l'a adaptée au projet de base existant, sans modifier le reste du projet. En parallèle, en s'appuyant sur les retours de tests fournis par Bourama, il a optimisé le code du tracking et l'a séparé en plusieurs fichiers afin d'améliorer à la fois la précision du tracking et la lisibilité du code.

### DAO Trung-Hieu — Recherches sur l'implémentation d'une vraie IA
Trung-Hieu a mené des recherches ciblées sur les techniques d'IA à intégrer dans les prochaines semaines : Q-Learning pour l'agent tactique (Semaine 4), modèles LSTM/MLP pour la prédiction de performance, et approches de Reinforcement Learning. Ces recherches constituent la base théorique pour l'implémentation de la semaine 4.

### COULIBALY Bourama — Tests
Bourama a réalisé les tests du pipeline complet : lancement du tracking sur une vidéo d'entrée, vérification des statistiques générées, validation de l'affichage dans le dashboard et identification des cas limites (joueurs non détectés, passes nulles, etc.).

---

## 3. Réalisations techniques

### 3.1 Pipeline de tracking fonctionnel
Le pipeline complet est opérationnel : vidéo d'entrée → détection YOLOv8 → tracking ByteTrack → extraction des statistiques → export JSON → affichage Streamlit. L'application peut être lancée avec `streamlit run app.py`.

### 3.2 Dashboard Streamlit — 4 onglets

#### Onglet "Joueurs" *(voir Capture 1)*
Tableau de classement de tous les joueurs détectés avec les métriques suivantes :
- Score de performance (0–100)
- Rang général
- Vitesse moyenne (km/h)
- Distance parcourue (m)
- Passes tentées / réussies / taux de réussite (%)
- Possession (%)
- Indicateur de sous-performance (fond rouge)

Les joueurs sont filtrables par équipe (E1/E2) et par poste (Gardien, Défenseur, Milieu, Attaquant). Sur la vidéo testée (`match_extrait.mp4`, 12.6s, 316 frames), **29 joueurs ont été détectés**.

#### Onglet "Vidéo annotée" *(voir Capture 2)*
Lecture de la vidéo de tracking avec annotations visuelles sur chaque joueur : identifiant, équipe colorée, statistiques en temps réel (distance, vitesse). Le ballon est également tracké et identifié.

#### Onglet "Substitutions" *(voir Capture 3)*
**10 substitutions recommandées** ont été générées automatiquement. Chaque recommandation inclut :
- L'identifiant du joueur, son équipe et son poste
- Son score de performance sur 100
- Les raisons explicitées (ex. : *"vitesse trop faible (1.9 km/h)"*, *"distance insuffisante (7 m)"*)

Ceci correspond aux **tâches 3.1, 3.2 et 3.6** du sujet : les règles de décision sont appliquées et chaque recommandation est justifiée avec les métriques du joueur.

#### Onglet "Graphiques" *(voir Capture 4)*
- Diagramme en barres de la **possession par équipe** (Équipe 1 ~65%, Équipe 2 ~35%)
- Graphiques comparatifs par métrique (Score, Vitesse, Distance, Possession) sélectionnables
- Nuage de points Vitesse moyenne vs Distance parcourue, coloré par équipe et dimensionné par score

---

## 4. Captures d'écran

### Capture 1 — Tableau des joueurs
![Tableau joueurs](semaine3_images/WhatsApp%20Image%202026-06-18%20at%2018.03.59.jpeg)
*Onglet "Joueurs" : classement des 29 joueurs détectés avec toutes les métriques, filtrés par équipe et poste.*

### Capture 2 — Vidéo annotée
![Vidéo annotée](semaine3_images/WhatsApp%20Image%202026-06-18%20at%2018.04.28.jpeg)
*Onglet "Vidéo annotée" : tracking en temps réel des joueurs avec bounding boxes colorées par équipe et annotations de statistiques.*

### Capture 3 — Recommandations de substitutions
![Substitutions](semaine3_images/WhatsApp%20Image%202026-06-18%20at%2018.04.56.jpeg)
*Onglet "Substitutions" : 10 recommandations de remplacement générées par le moteur de décision, chacune accompagnée d'une justification.*

### Capture 4 — Graphiques
![Graphiques](semaine3_images/WhatsApp%20Image%202026-06-18%20at%2018.05.30.jpeg)
*Onglet "Graphiques" : visualisation de la possession par équipe et des métriques comparatives.*

---

## 5. Bilan par rapport aux livrables attendus fin S3

| Livrable attendu | Statut | Commentaire |
|------------------|--------|-------------|
| Moteur de décision (règles SI/ALORS) | ✅ Réalisé | Règles basées sur score, vitesse, distance |
| Recommandations tactiques (remplacements) | ✅ Réalisé | 10 substitutions générées avec justification |
| Explications des recommandations | ✅ Réalisé | Raisons explicites par joueur dans l'onglet Substitutions |
| Analyse adverse (joueur/zone faible) | ⚠️ Partiel | Classement visible mais pas d'analyse adverse dédiée |
| Simulation tactique (4-3-3, 4-4-2…) | ❌ Non réalisé | À intégrer en S4 |
| Tactical Compatibility Score | ❌ Non réalisé | À intégrer en S4 |

---

## 6. Points bloquants et difficultés rencontrées

- La **détection des passes** reste à 0 sur la vidéo de test : la logique de détection de passes (proximité ballon–joueur) nécessite un affinage des seuils de distance.
- La vidéo de test (`match_extrait.mp4`) ne dure que 12.6 secondes, ce qui limite la pertinence statistique des métriques de distance et de vitesse.
- L'identification des deux équipes repose actuellement sur la couleur des maillots : cette classification peut être imprécise selon l'éclairage ou les angles de caméra.

---

## 7. Perspectives — Semaine 4

Pour la dernière semaine, le groupe prévoit de :
1. Implémenter la **simulation tactique** (comparaison de formations 4-3-3, 4-4-2, 3-5-2, 3-4-3)
2. Développer l'**analyse adverse** (identification des zones et joueurs faibles)
3. Intégrer un **agent Q-Learning** pour l'aide à la décision automatique (travaux de Trung-Hieu)
4. Finaliser le dashboard avec les onglets *Avant match*, *Pendant match*, *Après match*
5. Générer un **rapport automatique** post-match
6. Rédiger le rapport scientifique (8-10 pages) et préparer les slides de soutenance
