import json
import numpy as np
from collections import defaultdict

from tracking.config import MIN_PLAYER_FRAMES, SUBST_MIN_FRAMES, SUBST_SCORE_THRESH


def estimate_poste(state, pno, avg_pos, std_x_all):
    """
    Estime le poste d'un joueur à partir de :
      - la variance de sa position x (faible variance → gardien)
      - son rang de position x au sein de son équipe (défenseur/milieu/attaquant)
    Retourne None si pas assez de données.
    """
    if state.is_referee.get(pno):
        return "Arbitre"
    pos = state.player_positions.get(pno, [])
    if len(pos) < 5:
        return None

    std_x = float(np.std([p[0] for p in pos]))
    # Seuil = percentile 15 de tous les joueurs (les plus statiques = gardiens)
    threshold = float(np.percentile(std_x_all, 15)) if len(std_x_all) >= 4 else 40.0
    threshold = max(threshold, 25.0)

    if std_x <= threshold:
        return "Gardien"

    team = state.player_team.get(pno)
    if team is None or pno not in avg_pos:
        return None

    team_pnos = [p for p in avg_pos if state.player_team.get(p) == team and not state.is_referee.get(p)]
    if len(team_pnos) < 2:
        return None

    sorted_by_x = sorted(team_pnos, key=lambda p: avg_pos[p][0])
    try:
        rank = sorted_by_x.index(pno)
    except ValueError:
        return None
    n   = len(sorted_by_x)
    pct = rank / max(n - 1, 1)

    if pct < 0.30:
        return "Défenseur"
    elif pct < 0.65:
        return "Milieu"
    else:
        return "Attaquant"


def save_stats(state, video_path, stats_path, fps, frame_count):
    # Filtre : exclure les joueurs fantômes (présents < MIN_PLAYER_FRAMES frames)
    all_pnos = {pno for pno in (set(state.player_speeds.keys())
                                | set(state.passes_attempted.keys())
                                | set(state.passes_received.keys())
                                | set(state.possession_frames.keys()))
                if state.player_frame_count.get(pno, 0) >= MIN_PLAYER_FRAMES}

    players  = {}
    arbitres = {}

    total_possession = max(sum(state.possession_frames.values()), 1)
    duration_sec     = frame_count / max(fps, 1.0)
    expected_dist    = max(duration_sec * 3.0, 1.0)   # distance attendue pour ce clip (constante)

    # Pré-calcul position moyenne et std-x (pour estimation des postes)
    avg_pos = {}
    for pno in all_pnos:
        pos = state.player_positions.get(pno, [])
        if pos:
            avg_pos[pno] = (float(np.mean([p[0] for p in pos])),
                            float(np.mean([p[1] for p in pos])))

    std_x_all = [float(np.std([p[0] for p in state.player_positions[pno]]))
                 for pno in all_pnos
                 if len(state.player_positions.get(pno, [])) >= 5
                 and not state.is_referee.get(pno)]

    for pno in sorted(all_pnos):
        speeds      = state.player_speeds[pno]
        attempted   = state.passes_attempted[pno]
        made        = state.passes_made[pno]
        rate        = round(made / attempted * 100, 1) if attempted > 0 else 0.0
        vitesse_moy = round(sum(speeds) / len(speeds), 2) if speeds else 0.0
        poss_pct    = round(state.possession_frames[pno] / total_possession * 100, 1)
        distance_m  = round(state.player_distance[pno], 1)
        poste       = estimate_poste(state, pno, avg_pos, std_x_all)

        # Score final [0-100] : vitesse(40) + passes(30) + distance(30)
        sp_pts  = min(vitesse_moy / 12.0, 1.0) * 40.0
        pp_pts  = (rate / 100.0 * 30.0) if attempted > 0 else 15.0
        dp_pts  = min(distance_m / expected_dist, 1.0) * 30.0
        score   = round(sp_pts + pp_pts + dp_pts, 1)
        underperf = score < 35

        if state.is_referee.get(pno):
            arbitres[str(pno)] = {
                "vitesse_moyenne_kmh": vitesse_moy,
                "distance_m":          distance_m,
            }
        else:
            players[str(pno)] = {
                "player_id":            pno,
                "equipe":               state.player_team.get(pno),
                "poste":                poste,
                "score":                score,
                "sous_performance":     underperf,
                "vitesse_moyenne_kmh":  vitesse_moy,
                "distance_m":           distance_m,
                "passes_tentees":       attempted,
                "passes_reussies":      made,
                "passes_recues":        state.passes_received[pno],
                "taux_reussite_passes": rate,
                "possession_pct":       poss_pct,
            }

    # Classement par score décroissant (rang 1 = meilleur)
    sorted_pnos = sorted(players, key=lambda k: players[k]["score"], reverse=True)
    for rank, k in enumerate(sorted_pnos, start=1):
        players[k]["classement"] = rank

    # Possession par équipe
    team_poss = defaultdict(int)
    for pno, frames in state.possession_frames.items():
        team = state.player_team.get(pno)
        if team:
            team_poss[team] += frames
    possession_equipes = {
        f"equipe_{t}": round(team_poss[t] / total_possession * 100, 1)
        for t in sorted(team_poss)
    }

    # ── Recommandations de substitution ──────────────────────────────────────────
    # Conditions : données suffisantes (>= SUBST_MIN_FRAMES), score faible,
    # hors gardiens et arbitres (leur faible mobilité est structurelle).
    subst_reco = {}
    for pno_str, p in players.items():
        pno = p["player_id"]
        if state.player_frame_count.get(pno, 0) < SUBST_MIN_FRAMES:
            continue
        if p["poste"] in ("Gardien", "Arbitre"):
            continue
        if p["score"] >= SUBST_SCORE_THRESH:
            continue

        raisons = []
        if p["vitesse_moyenne_kmh"] < 4.0:
            raisons.append(f"vitesse trop faible ({p['vitesse_moyenne_kmh']:.1f} km/h)")
        if p["distance_m"] < expected_dist * 0.4:
            raisons.append(f"distance insuffisante ({p['distance_m']:.0f} m)")
        if p["passes_tentees"] >= 3 and p["taux_reussite_passes"] < 30.0:
            raisons.append(f"passes inefficaces ({p['taux_reussite_passes']:.0f}%)")
        if not raisons:
            raisons.append("score global insuffisant")

        subst_reco[pno_str] = {
            "player_id": pno,
            "equipe":    p["equipe"],
            "poste":     p["poste"],
            "score":     p["score"],
            "raisons":   raisons,
        }

    stats = {
        "video":                        video_path,
        "duree_secondes":               round(duration_sec, 1),
        "frames_traitees":              frame_count,
        "joueurs":                      players,
        "arbitres":                     arbitres,
        "total_passes":                 sum(state.passes_made.values()),
        "possession_equipes":           possession_equipes,
        "substitutions_recommandees":   subst_reco,
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n" + "═" * 95)
    print(f"  STATS — {video_path}  ({round(duration_sec, 1)}s)")
    print("═" * 95)
    print(f"  {'#':<4} {'Joueur':<8} {'Eq':>3} {'Poste':<12} {'Score':>6} {'Vit.moy':>8} "
          f"{'Dist(m)':>8} {'P.tent':>7} {'P.réus':>7} {'Taux':>7} {'Poss.':>7}")
    print("─" * 95)
    for k in sorted_pnos:
        p     = players[k]
        pno   = p["player_id"]
        eq    = f"E{p['equipe']}" if p['equipe'] else "  ?"
        poste = (p['poste'] or "?")[:11]
        flag  = " !" if p['sous_performance'] else "  "
        print(f"  {p['classement']:<3}{flag} P{pno:<6} {eq:>3} {poste:<12} {p['score']:>5.1f}  "
              f"{p['vitesse_moyenne_kmh']:>7.1f}  {p['distance_m']:>7.0f}  "
              f"{p['passes_tentees']:>6}  {p['passes_reussies']:>6}  "
              f"{p['taux_reussite_passes']:>5.1f}%  {p['possession_pct']:>5.1f}%")
    if arbitres:
        print("─" * 95)
        print(f"  Arbitres : {', '.join(f'P{k}' for k in sorted(int(k) for k in arbitres))}")
    print("─" * 95)
    print(f"  Total passes réussies : {stats['total_passes']}")
    if possession_equipes:
        parts = "  /  ".join(f"Équipe {k[-1]} : {v}%" for k, v in possession_equipes.items())
        print(f"  Possession — {parts}")
    # Joueurs en sous-performance
    sp_list = [f"P{p['player_id']}" for p in players.values() if p['sous_performance']]
    if sp_list:
        print(f"  Sous-performance (!) : {', '.join(sp_list)}")

    # Recommandations de substitution
    if subst_reco:
        print("─" * 95)
        print("  SUBSTITUTIONS RECOMMANDÉES :")
        for _, r in sorted(subst_reco.items(), key=lambda x: x[1]["score"]):
            eq    = f"E{r['equipe']}" if r['equipe'] else "  ?"
            poste = (r['poste'] or "?")[:11]
            print(f"    → P{r['player_id']:<5} {eq}  {poste:<12}  score {r['score']:>5.1f}  |  {', '.join(r['raisons'])}")
    else:
        print("─" * 95)
        print("  Aucune substitution recommandée (données insuffisantes ou tous les joueurs qualifiés).")

    print("═" * 95)
    print(f"\nStats sauvegardées : {stats_path}")
