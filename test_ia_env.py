import sys
import os
import numpy as np
from collections import defaultdict

# On s'assure que le dossier courant est dans le chemin pour importer tes modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from tracking.state import TrackingState
from tracking.stats import save_stats

def run_mock_simulation():
    print("🏗️ Création de l'environnement de simulation de match...")
    state = TrackingState()
    
    # ── 1. CONFIGURATION DU MATCH SIMULÉ ──────────────────────────────────────
    fps = 30.0
    # On simule un match qui a duré 3000 frames (~100 secondes)
    state.frame_count = 3000 
    
    # ── 2. CRÉATION DES PROFILS DE JOUEURS (SCÉNARIOS DE TEST) ────────────────
    # Joueur 1 (Équipe 1) : L'infatigable (Très performant, court beaucoup)
    # Joueur 2 (Équipe 1) : Le fatigué/blessé (Vitesse en chute, beaucoup de frames)
    # Joueur 3 (Équipe 2) : L'attaquant adverse statique
    # Joueur 4 (Équipe 1) : Le Gardien (très faible mouvement sur X)
    
    pnos = [1, 2, 3, 4]
    
    # Présence sur le terrain (en nombre de frames)
    state.player_frame_count[1] = 2900
    state.player_frame_count[2] = 2950
    state.player_frame_count[3] = 2800
    state.player_frame_count[4] = 3000
    
    # Attribution des équipes (1 ou 2)
    state.player_team[1] = 1
    state.player_team[2] = 1
    state.player_team[4] = 1 # Gardien E1
    state.player_team[3] = 2 # Adversaire
    
    # Simulation des vitesses (historique de courses)
    state.player_speeds[1] = [14.0] * 100 + [12.0] * 100  # Rapide
    state.player_speeds[2] = [12.0] * 50 + [4.5] * 150    # Grosse chute de vitesse (Fatigue !)
    state.player_speeds[3] = [10.0] * 200
    state.player_speeds[4] = [2.0] * 200                 # Lent (Gardien)
    
    # Distances parcourues (en mètres)
    state.player_distance[1] = 450.0  # Énorme volume de jeu
    state.player_distance[2] = 380.0  # A beaucoup couru mais piétine à la fin
    state.player_distance[3] = 250.0
    state.player_distance[4] = 25.0   # Logique pour un gardien
    
    # Passes (Tentées, Réussies)
    state.passes_attempted[1] = 15; state.passes_made[1] = 13  # Excellent taux
    state.passes_attempted[2] = 8;  state.passes_made[2] = 2   # Rapprochement manqué (Sous-perf)
    state.passes_attempted[3] = 10; state.passes_made[3] = 7
    
    # Possession (Nombre de frames avec la balle)
    state.possession_frames[1] = 600
    state.possession_frames[2] = 100
    state.possession_frames[3] = 500
    
    # ── 3. SIMULATION DES POSITIONS POUR LES FORMATIONS (Semaine 3) ───────────
    # Rappel : l'axe X détermine le poste (ex: terrain de 0 à 1280 pixels)
    # On simule l'historique des coordonnées X moyennes pour tester le tri spatial
    state.player_positions[4] = [(50, 360)] * 20   # X très bas -> Gardien
    state.player_positions[2] = [(250, 200)] * 20  # X bas -> Défenseur
    state.player_positions[1] = [(600, 400)] * 20  # X moyen -> Milieu
    state.player_positions[3] = [(1000, 300)] * 20 # X haut -> Attaquant de l'autre équipe
    
    # ── 4. EXÉCUTION DU TEST ──────────────────────────────────────────────────
    print("🏃 Lancement du moteur de décision sur les données simulées...")
    
    video_mock = "input_videos/japon_tunisie.mp4"
    stats_json_mock = "output_videos/match_simulation_stats.json"
    
    # On appelle ta fonction save_stats corrigée
    save_stats(state, video_mock, stats_json_mock, fps, state.frame_count)
    
    print("\n✅ TEST DE L'ENVIRONNEMENT TERMINÉ !")
    print(f"Le fichier JSON de test a été généré ici : {stats_json_mock}")
    print("Tu peux maintenant ouvrir ton Dashboard Streamlit pour voir ce match virtuel sans avoir calculé de vidéo !")

if __name__ == "__main__":
    run_mock_simulation()