"""
tracking/tactical.py
====================
Semaine 3 — Tactical Decision Engine

Tâche 3.1 : Règles de décision (SI fatigue > 70 ET perf < 50 → remplacement)
Tâche 3.2 : Recommandation de remplacement (qui sort, qui entre, pourquoi)
Tâche 3.3 : Tactical Compatibility Score (TCS)
Tâche 3.4 : Analyse adverse (zones faibles, joueurs faibles)
Tâche 3.5 : Simulation tactique (4-3-3 / 4-4-2 / 3-5-2 / 3-4-3)
Tâche 3.6 : Interprétation des décisions
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# ═══════════════════════════════════════════════════════════════════════════════
#  Tâche 3.1 & 3.2 — Moteur de règles + recommandations substitution enrichies
# ═══════════════════════════════════════════════════════════════════════════════

# Seuils du sujet
FATIGUE_THRESH  = 70.0   # > 70 → candidat sortie
PERF_THRESH     = 50.0   # < 50 → sous-performance
INJURY_HIGH     = "Élevé"

# Postes qu'on ne sort jamais sur critère fatigue seule
PROTECTED_POSTES = {"Gardien"}


def _decision_rules(player: dict) -> list[str]:
    """
    Applique les règles SI/ALORS du sujet sur un joueur.
    Retourne la liste des raisons déclenchées (vide = pas de remplacement).
    """
    raisons = []
    fat  = player.get("fatigue_score", 0.0)
    perf = player.get("score", 100.0)
    inj  = player.get("injury_risk", "Faible")
    inj_score = player.get("injury_score", 0.0)

    # Règle principale du sujet
    if fat > FATIGUE_THRESH and perf < PERF_THRESH:
        raisons.append(
            f"Règle principale : fatigue ({fat:.1f}%) > 70 ET performance ({perf:.1f}/100) < 50"
        )

    # Règles individuelles
    if fat > FATIGUE_THRESH and perf >= PERF_THRESH:
        raisons.append(f"Fatigue critique ({fat:.1f}%) — label : {player.get('fatigue_label','?')}")

    if perf < PERF_THRESH and fat <= FATIGUE_THRESH:
        raisons.append(f"Sous-performance ({perf:.1f}/100) sans fatigue excessive")

    if inj == INJURY_HIGH:
        raisons.append(f"Risque blessure élevé (score : {inj_score:.1f}/100)")

    if player.get("sprint_pct", 0) > 25:
        raisons.append(f"Volume de sprints très élevé ({player['sprint_pct']:.1f}% des frames)")

    return raisons


def compute_substitution_recommendations(players: dict) -> dict:
    """
    Tâche 3.2 — Pour chaque joueur candidat à la sortie, produit une fiche complète :
      - raisons détaillées
      - profil du remplaçant idéal
      - interprétation textuelle (Tâche 3.6)
    """
    # Grouper par équipe pour trouver les remplaçants potentiels
    by_team: dict[int, list[dict]] = {}
    for p in players.values():
        t = p.get("equipe")
        if t:
            by_team.setdefault(t, []).append(p)

    recommendations = {}

    for pno_str, p in players.items():
        if p.get("poste") in PROTECTED_POSTES:
            continue

        raisons = _decision_rules(p)
        if not raisons:
            continue

        # Trouver le meilleur remplaçant (même équipe, même poste ou polyvalent,
        # meilleur score, pas lui-même)
        team = p.get("equipe")
        same_team = [q for q in by_team.get(team, [])
                     if q["player_id"] != p["player_id"]
                     and q.get("poste") == p.get("poste")
                     and q.get("fatigue_score", 100) < 50]

        # Fallback : n'importe quel joueur de l'équipe moins fatigué
        if not same_team:
            same_team = [q for q in by_team.get(team, [])
                         if q["player_id"] != p["player_id"]
                         and q.get("fatigue_score", 100) < p.get("fatigue_score", 0)]

        remplacant = None
        if same_team:
            remplacant = max(same_team, key=lambda q: q.get("score", 0))

        tcs = _tactical_compatibility_score(p, remplacant) if remplacant else None

        # Interprétation textuelle (Tâche 3.6)
        interpretation = _interpret_substitution(p, remplacant, raisons, tcs)

        recommendations[pno_str] = {
            "player_id":     p["player_id"],
            "equipe":        team,
            "poste":         p.get("poste"),
            "score":         p.get("score", 0),
            "fatigue_score": p.get("fatigue_score", 0),
            "injury_risk":   p.get("injury_risk", "Faible"),
            "raisons":       raisons,
            "remplacant_id": remplacant["player_id"] if remplacant else None,
            "remplacant_score": remplacant["score"] if remplacant else None,
            "tcs":           tcs,
            "interpretation": interpretation,
        }

    return recommendations


# ═══════════════════════════════════════════════════════════════════════════════
#  Tâche 3.3 — Tactical Compatibility Score (TCS)
# ═══════════════════════════════════════════════════════════════════════════════

def _tactical_compatibility_score(sortant: dict, entrant: dict) -> dict:
    """
    Calcule le TCS entre le joueur qui sort et son remplaçant.

    Composantes :
      - fraîcheur physique   (40%) : 100 - fatigue_entrant
      - performance relative (30%) : score_entrant / max(score_sortant, 1)
      - complémentarité poste(20%) : même poste = 1.0, sinon 0.5
      - risque blessure      (10%) : 0 si risque élevé, 1 si faible

    Score final : 0–100
    """
    risk_map = {"Faible": 1.0, "Modéré": 0.5, "Élevé": 0.0}

    fraicheur   = max(0, 100 - entrant.get("fatigue_score", 0)) / 100.0
    perf_rel    = min(entrant.get("score", 0) / max(sortant.get("score", 1), 1), 1.5) / 1.5
    compl_poste = 1.0 if entrant.get("poste") == sortant.get("poste") else 0.5
    risque      = risk_map.get(entrant.get("injury_risk", "Faible"), 0.5)

    tcs_score = (
        0.40 * fraicheur +
        0.30 * perf_rel  +
        0.20 * compl_poste +
        0.10 * risque
    ) * 100.0

    if tcs_score >= 75:
        label = "Excellent"
    elif tcs_score >= 55:
        label = "Bon"
    elif tcs_score >= 35:
        label = "Acceptable"
    else:
        label = "Risqué"

    return {
        "score":          round(tcs_score, 1),
        "label":          label,
        "fraicheur":      round(fraicheur * 100, 1),
        "perf_relative":  round(perf_rel * 100, 1),
        "compl_poste":    round(compl_poste * 100, 1),
        "risque_blessure":round(risque * 100, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Tâche 3.4 — Analyse adverse
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_opponent(players: dict, our_team: int) -> dict:
    """
    Identifie les faiblesses de l'équipe adverse :
      - joueurs les plus faibles (score bas)
      - joueurs les plus fatigués (candidats à exploiter)
      - zones défensives faibles (basé sur les positions moyennes)
      - opportunités tactiques
    """
    opponent_team = 2 if our_team == 1 else 1
    adv_players = [p for p in players.values() if p.get("equipe") == opponent_team]

    if not adv_players:
        return {"error": "Aucun joueur adverse détecté"}

    # Joueur le plus faible (score le plus bas)
    weakest = min(adv_players, key=lambda p: p.get("score", 100))

    # Joueur le plus fatigué
    most_tired = max(adv_players, key=lambda p: p.get("fatigue_score", 0))

    # Analyse par poste adverse
    poste_analysis = {}
    for p in adv_players:
        poste = p.get("poste") or "Inconnu"
        if poste not in poste_analysis:
            poste_analysis[poste] = {"scores": [], "fatigues": [], "players": []}
        poste_analysis[poste]["scores"].append(p.get("score", 0))
        poste_analysis[poste]["fatigues"].append(p.get("fatigue_score", 0))
        poste_analysis[poste]["players"].append(p["player_id"])

    weakest_poste = None
    weakest_poste_score = 999
    for poste, data in poste_analysis.items():
        if poste in ("Gardien", "Arbitre", "Inconnu"):
            continue
        avg_score = sum(data["scores"]) / len(data["scores"])
        if avg_score < weakest_poste_score:
            weakest_poste_score = avg_score
            weakest_poste = poste

    # Zones d'exploitation (basé sur possession et passes)
    low_possession = [p for p in adv_players if p.get("possession_pct", 100) < 5]
    low_pass_rate  = [p for p in adv_players if p.get("taux_reussite_passes", 100) < 40
                      and p.get("passes_tentees", 0) > 0]

    # Génération des opportunités textuelles
    opportunities = []

    if weakest_poste:
        opportunities.append(
            f"Zone faible identifiée : ligne des {weakest_poste}s adverses "
            f"(score moyen {weakest_poste_score:.1f}/100) — à cibler en priorité."
        )

    if most_tired.get("fatigue_score", 0) > 60:
        opportunities.append(
            f"P{most_tired['player_id']} ({most_tired.get('poste','?')}) adverse est épuisé "
            f"(fatigue {most_tired['fatigue_score']:.1f}%) — exploiter son couloir."
        )

    if low_pass_rate:
        names = ", ".join(f"P{p['player_id']}" for p in low_pass_rate[:3])
        opportunities.append(
            f"Pressing haut recommandé sur {names} — taux de passes réussies faible."
        )

    avg_adv_score = sum(p.get("score", 0) for p in adv_players) / max(len(adv_players), 1)
    if avg_adv_score < 40:
        opportunities.append(
            f"Score moyen adverse faible ({avg_adv_score:.1f}/100) — transition rapide recommandée."
        )

    return {
        "equipe_adverse":     opponent_team,
        "nb_joueurs_adverse": len(adv_players),
        "joueur_plus_faible": {
            "player_id": weakest["player_id"],
            "poste":     weakest.get("poste"),
            "score":     weakest.get("score"),
        },
        "joueur_plus_fatigue": {
            "player_id":     most_tired["player_id"],
            "poste":         most_tired.get("poste"),
            "fatigue_score": most_tired.get("fatigue_score"),
        },
        "zone_faible":         weakest_poste,
        "score_zone_faible":   round(weakest_poste_score, 1) if weakest_poste else None,
        "analyse_par_poste":   {
            k: {"score_moyen": round(sum(v["scores"]) / len(v["scores"]), 1),
                "fatigue_moy": round(sum(v["fatigues"]) / len(v["fatigues"]), 1),
                "joueurs":     v["players"]}
            for k, v in poste_analysis.items()
        },
        "opportunites":        opportunities,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Tâche 3.5 — Simulation tactique (formations)
# ═══════════════════════════════════════════════════════════════════════════════

# Profils de formation : (défenseurs, milieux, attaquants)
FORMATIONS = {
    "4-3-3": (4, 3, 3),
    "4-4-2": (4, 4, 2),
    "3-5-2": (3, 5, 2),
    "3-4-3": (3, 4, 3),
}


def _xg_simplified(attaquants: int, possession_pct: float, avg_score: float) -> float:
    """
    xG simplifié basé sur le nombre d'attaquants, la possession et le score moyen.
    Formule empirique normalisée entre 0 et 3.
    """
    base = attaquants * 0.4
    poss_bonus = (possession_pct / 100.0) * 0.8
    perf_bonus = (avg_score / 100.0) * 0.6
    return round(min(base + poss_bonus + perf_bonus, 3.0), 2)


def _pressing_index(milieux: int, avg_fatigue: float) -> float:
    """Indice de pressing 0-100 : milieux nombreux + joueurs frais = pressing haut."""
    base = milieux * 12.0
    fatigue_penalty = avg_fatigue * 0.4
    return round(max(0, min(100, base - fatigue_penalty)), 1)


def _defensive_risk(defenseurs: int, avg_fatigue: float, opp_attaquants: int) -> float:
    """Risque défensif 0-100 : peu de défenseurs + fatigués + adversaire offensif = risque élevé."""
    base = max(0, (opp_attaquants - defenseurs) * 15.0)
    fat_bonus = avg_fatigue * 0.3
    return round(min(100, base + fat_bonus), 1)


def simulate_formations(players: dict, our_team: int) -> dict:
    """
    Compare les 4 formations sur les métriques du sujet.
    Retourne le classement et la formation recommandée.
    """
    our_players = [p for p in players.values() if p.get("equipe") == our_team]
    opp_players = [p for p in players.values() if p.get("equipe") != our_team
                   and p.get("equipe") is not None]

    if not our_players:
        return {"error": "Aucun joueur pour l'équipe analysée"}

    # Stats globales équipe
    avg_score   = sum(p.get("score", 0)         for p in our_players) / len(our_players)
    avg_fatigue = sum(p.get("fatigue_score", 0)  for p in our_players) / len(our_players)
    poss_total  = sum(p.get("possession_pct", 0) for p in our_players)
    possession  = min(poss_total, 100.0)

    opp_att = sum(1 for p in opp_players if p.get("poste") == "Attaquant")

    results = {}
    for formation, (def_, mil, att) in FORMATIONS.items():
        xg       = _xg_simplified(att, possession, avg_score)
        pressing = _pressing_index(mil, avg_fatigue)
        risk_def = _defensive_risk(def_, avg_fatigue, opp_att)

        # Score global de la formation (pondéré)
        global_score = round(
            0.35 * xg / 3.0 * 100 +
            0.30 * pressing +
            0.35 * (100 - risk_def),
            1
        )

        results[formation] = {
            "formation":       formation,
            "defenseurs":      def_,
            "milieux":         mil,
            "attaquants":      att,
            "possession_pct":  round(possession, 1),
            "pressing":        pressing,
            "xg_simplifie":    xg,
            "risque_defensif": risk_def,
            "score_global":    global_score,
        }

    # Classement
    ranked = sorted(results.values(), key=lambda x: x["score_global"], reverse=True)
    for i, r in enumerate(ranked):
        r["rang"] = i + 1

    best = ranked[0]["formation"]
    reason = (
        f"Le {best} maximise le score global ({ranked[0]['score_global']:.1f}/100) "
        f"avec xG={ranked[0]['xg_simplifie']}, pressing={ranked[0]['pressing']:.0f}/100 "
        f"et risque défensif={ranked[0]['risque_defensif']:.0f}/100."
    )

    return {
        "formation_recommandee": best,
        "raison":                reason,
        "formations":            {f: results[f] for f in FORMATIONS},
        "classement":            [r["formation"] for r in ranked],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Tâche 3.6 — Interprétation textuelle des décisions
# ═══════════════════════════════════════════════════════════════════════════════

def _interpret_substitution(
    sortant: dict,
    entrant: dict | None,
    raisons: list[str],
    tcs: dict | None,
) -> str:
    """Génère l'explication complète de la décision de substitution."""
    lines = [
        f"→ SORTIE recommandée : P{sortant['player_id']} "
        f"({sortant.get('poste','?')}, E{sortant.get('equipe','?')})",
        f"   Performance : {sortant.get('score', 0):.1f}/100 | "
        f"Fatigue : {sortant.get('fatigue_score',0):.1f}% | "
        f"Risque blessure : {sortant.get('injury_risk','?')}",
        "",
        "   Déclencheurs :",
    ]
    for r in raisons:
        lines.append(f"   • {r}")

    if entrant:
        lines += [
            "",
            f"→ ENTRÉE suggérée : P{entrant['player_id']} "
            f"({entrant.get('poste','?')}, E{entrant.get('equipe','?')})",
            f"   Performance : {entrant.get('score',0):.1f}/100 | "
            f"Fatigue : {entrant.get('fatigue_score',0):.1f}%",
        ]
        if tcs:
            lines.append(
                f"   Compatibilité tactique (TCS) : {tcs['score']:.1f}/100 → {tcs['label']}"
            )
    else:
        lines.append("\n   ⚠️ Aucun remplaçant disponible identifié.")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  Point d'entrée principal — appelé depuis stats.py
# ═══════════════════════════════════════════════════════════════════════════════

def run_tactical_engine(players: dict, our_team: int = 1) -> dict:
    """
    Lance le moteur tactique S3 complet.

    Parameters
    ----------
    players   : dict de joueurs (format JSON stats)
    our_team  : équipe à analyser (1 ou 2)

    Returns
    -------
    dict avec substitutions, analyse adverse, simulation formations
    """
    print(f"[S3] Moteur tactique — équipe {our_team}...")

    substitutions  = compute_substitution_recommendations(players)
    opponent_analysis = analyze_opponent(players, our_team)
    formation_sim  = simulate_formations(players, our_team)

    print(f"[S3] {len(substitutions)} substitution(s) recommandée(s)")
    print(f"[S3] Formation recommandée : {formation_sim.get('formation_recommandee', '?')}")

    return {
        "substitutions_s3":       substitutions,
        "analyse_adverse":        opponent_analysis,
        "simulation_formations":  formation_sim,
    }