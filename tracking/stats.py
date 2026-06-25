import json
import numpy as np
from collections import defaultdict

from tracking.config import MIN_PLAYER_FRAMES, SUBST_MIN_FRAMES, SUBST_SCORE_THRESH


def _fatigue_score(speeds):
    if len(speeds) < 10:
        return 0.0
    mid = len(speeds) // 2
    first_avg  = sum(speeds[:mid]) / mid
    second_avg = sum(speeds[mid:]) / (len(speeds) - mid)
    if first_avg < 0.5:
        return 0.0
    drop = max(0.0, (first_avg - second_avg) / first_avg)
    return round(min(drop * 150.0, 100.0), 1)


def _injury_risk(fatigue, distance_m, expected_dist, vitesse_moy):
    raw = (
        0.5 * fatigue
        + 0.3 * min(distance_m / max(expected_dist, 1.0) * 100.0, 100.0)
        + 0.2 * min(vitesse_moy / 20.0 * 100.0, 100.0)
    )
    score = round(min(raw, 100.0), 1)
    label = "Élevé" if score >= 60 else ("Moyen" if score >= 35 else "Faible")
    return score, label


def estimate_poste(state, pno, avg_pos, std_x_all):
    if state.is_referee.get(pno):
        return "Arbitre"
    pos = state.player_positions.get(pno, [])
    if len(pos) < 5:
        return None

    std_x     = float(np.std([p[0] for p in pos]))
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
    all_pnos = {pno for pno in (set(state.player_speeds.keys())
                                | set(state.passes_attempted.keys())
                                | set(state.passes_received.keys())
                                | set(state.possession_frames.keys()))
                if state.player_frame_count.get(pno, 0) >= MIN_PLAYER_FRAMES}

    players  = {}
    arbitres = {}

    total_possession = max(sum(state.possession_frames.values()), 1)
    duration_sec     = frame_count / max(fps, 1.0)
    expected_dist    = max(duration_sec * 3.0, 1.0)

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

        sp_pts  = min(vitesse_moy / 12.0, 1.0) * 40.0
        pp_pts  = (rate / 100.0 * 30.0) if attempted > 0 else 15.0
        dp_pts  = min(distance_m / expected_dist, 1.0) * 30.0
        score   = round(sp_pts + pp_pts + dp_pts, 1)
        underperf = score < 35

        fatigue              = _fatigue_score(list(speeds))
        inj_score, inj_label = _injury_risk(fatigue, distance_m, expected_dist, vitesse_moy)

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
                "fatigue_score":        fatigue,
                "injury_risk_score":    inj_score,
                "injury_risk_label":    inj_label,
            }

    sorted_pnos = sorted(players, key=lambda k: players[k]["score"], reverse=True)
    for rank, k in enumerate(sorted_pnos, start=1):
        players[k]["classement"] = rank

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

        fat      = p["fatigue_score"]
        inj_lbl  = p["injury_risk_label"]
        interp   = [f"Remplacer P{pno} ({p['poste'] or '?'}, E{p['equipe']}) — Score : {p['score']:.0f}/100"]
        for r in raisons:
            interp.append(f"• {r.capitalize()}")
        if fat >= 40:
            level = "élevée" if fat >= 60 else "modérée"
            interp.append(f"• Fatigue {level} ({fat:.0f}/100)")
        interp.append(f"• Risque blessure : {inj_lbl}")

        subst_reco[pno_str] = {
            "player_id":      pno,
            "equipe":         p["equipe"],
            "poste":          p["poste"],
            "score":          p["score"],
            "raisons":        raisons,
            "interpretation": "\n".join(interp),
        }

    # ── Analyse adverse ───────────────────────────────────────────────────────────
    all_x_vals = [pos[0] for k in players for pos in state.player_positions.get(int(k), [])]
    x_min = min(all_x_vals) if all_x_vals else 0.0
    x_max = max(all_x_vals) if all_x_vals else 1.0

    analyse_adverse = {}
    for team_id in [1, 2]:
        opponent_id = 3 - team_id
        candidates  = [
            p for p in players.values()
            if p["equipe"] == opponent_id
            and p["poste"] not in ("Gardien", "Arbitre")
            and p["poste"] is not None
        ]
        if not candidates:
            continue
        weakest = min(candidates, key=lambda p: p["score"])
        pno_w   = weakest["player_id"]
        pos_w   = state.player_positions.get(pno_w, [])
        if pos_w and x_max > x_min:
            avg_x = sum(p[0] for p in pos_w) / len(pos_w)
            rel   = (avg_x - x_min) / (x_max - x_min)
            zone  = "gauche" if rel < 0.33 else ("droite" if rel > 0.67 else "centre")
        else:
            zone = "inconnue"
        conseil = (
            f"Exploiter la faiblesse de P{pno_w} ({weakest['poste']}) "
            f"en zone {zone} — score {weakest['score']:.0f}/100, "
            f"vitesse {weakest['vitesse_moyenne_kmh']:.1f} km/h"
        )
        analyse_adverse[f"equipe_{team_id}"] = {
            "joueur_cible":    f"P{pno_w}",
            "poste":           weakest["poste"],
            "equipe_adverse":  f"E{opponent_id}",
            "score":           weakest["score"],
            "zone":            zone,
            "vitesse_kmh":     weakest["vitesse_moyenne_kmh"],
            "taux_passes_pct": weakest["taux_reussite_passes"],
            "conseil":         conseil,
        }

    # ── Rapport automatique ───────────────────────────────────────────────────────
    all_pl  = [p for p in players.values() if p["equipe"] is not None]
    best_p  = max(all_pl, key=lambda p: p["score"]) if all_pl else None
    worst_p = min(all_pl, key=lambda p: p["score"]) if all_pl else None
    dom_team = max(possession_equipes, key=possession_equipes.get) if possession_equipes else None

    subst_list = sorted(subst_reco.values(), key=lambda x: x["score"])

    txt = [
        f"RAPPORT DE MATCH — {video_path}",
        f"Durée : {round(duration_sec)}s | Frames traitées : {frame_count}",
        "",
        "▶ RÉSUMÉ",
    ]
    if best_p:
        txt.append(f"  Meilleur joueur  : P{best_p['player_id']} ({best_p['poste']}, E{best_p['equipe']}) — score {best_p['score']:.0f}/100")
    if worst_p:
        txt.append(f"  Joueur en diff.  : P{worst_p['player_id']} ({worst_p['poste']}, E{worst_p['equipe']}) — score {worst_p['score']:.0f}/100")
    if dom_team:
        txt.append(f"  Possession       : {dom_team.replace('equipe_', 'Équipe ')} dominante ({possession_equipes[dom_team]}%)")
    txt.append(f"  Passes réussies  : {sum(state.passes_made.values())}")

    if subst_list:
        txt += ["", f"▶ SUBSTITUTIONS RECOMMANDÉES ({len(subst_list)})"]
        for r in subst_list:
            txt.append(r["interpretation"])
            txt.append("")

    if analyse_adverse:
        txt.append("▶ ANALYSE ADVERSE")
        for adv in analyse_adverse.values():
            txt.append(f"  {adv['conseil']}")

    rapport_automatique = {
        "texte": "\n".join(txt),
        "meilleur_joueur": (
            {"id": f"P{best_p['player_id']}", "equipe": f"E{best_p['equipe']}",
             "poste": best_p["poste"], "score": best_p["score"]}
            if best_p else None
        ),
        "pire_joueur": (
            {"id": f"P{worst_p['player_id']}", "equipe": f"E{worst_p['equipe']}",
             "poste": worst_p["poste"], "score": worst_p["score"]}
            if worst_p else None
        ),
        "equipe_dominante": dom_team,
        "possession":       possession_equipes,
        "nb_substitutions": len(subst_reco),
        "substitutions": [
            {
                "joueur":         f"P{r['player_id']}",
                "equipe":         f"E{r['equipe']}",
                "score":          r["score"],
                "interpretation": r["interpretation"],
            }
            for r in subst_list
        ],
        "analyse_adverse": analyse_adverse,
    }

    stats = {
        "video":                      video_path,
        "duree_secondes":             round(duration_sec, 1),
        "frames_traitees":            frame_count,
        "joueurs":                    players,
        "arbitres":                   arbitres,
        "total_passes":               sum(state.passes_made.values()),
        "possession_equipes":         possession_equipes,
        "substitutions_recommandees": subst_reco,
        "analyse_adverse":            analyse_adverse,
        "rapport_automatique":        rapport_automatique,
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n" + "═" * 95)
    print(f"  STATS — {video_path}  ({round(duration_sec, 1)}s)")
    print("═" * 95)
    print(f"  {'#':<4} {'Joueur':<8} {'Eq':>3} {'Poste':<12} {'Score':>6} {'Vit.moy':>8} "
          f"{'Dist(m)':>8} {'P.tent':>7} {'P.réus':>7} {'Taux':>7} {'Poss.':>7} {'Fat.':>5} {'Risq.':>6}")
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
              f"{p['taux_reussite_passes']:>5.1f}%  {p['possession_pct']:>5.1f}%"
              f"  {p['fatigue_score']:>4.0f}  {p['injury_risk_label']:>6}")
    if arbitres:
        print("─" * 95)
        print(f"  Arbitres : {', '.join(f'P{k}' for k in sorted(int(k) for k in arbitres))}")
    print("─" * 95)
    print(f"  Total passes réussies : {stats['total_passes']}")
    if possession_equipes:
        parts = "  /  ".join(f"Équipe {k[-1]} : {v}%" for k, v in possession_equipes.items())
        print(f"  Possession — {parts}")
    sp_list = [f"P{p['player_id']}" for p in players.values() if p['sous_performance']]
    if sp_list:
        print(f"  Sous-performance (!) : {', '.join(sp_list)}")

    if subst_reco:
        print("─" * 95)
        print("  SUBSTITUTIONS RECOMMANDÉES :")
        for _, r in sorted(subst_reco.items(), key=lambda x: x[1]["score"]):
            eq    = f"E{r['equipe']}" if r['equipe'] else "  ?"
            poste = (r['poste'] or "?")[:11]
            print(f"    → P{r['player_id']:<5} {eq}  {poste:<12}  score {r['score']:>5.1f}")
            for line in r["interpretation"].split("\n")[1:]:
                print(f"       {line}")
    else:
        print("─" * 95)
        print("  Aucune substitution recommandée.")

    if analyse_adverse:
        print("─" * 95)
        print("  ANALYSE ADVERSE :")
        for adv in analyse_adverse.values():
            print(f"    → {adv['conseil']}")

    print("═" * 95)
    print(f"\nStats sauvegardées : {stats_path}")
