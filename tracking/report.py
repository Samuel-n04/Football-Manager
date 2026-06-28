"""
tracking/report.py
==================
Semaine 4 — Tâche 4.5 : Rapport automatique post-match

Génère un rapport textuel complet (Markdown) à partir du JSON de stats.
Couvre :
  - Résumé du match
  - Performances individuelles (top / flop)
  - Fatigue & risques blessure
  - Substitutions recommandées
  - Décisions IA (Q-Learning)
  - Formation recommandée
  - Analyse adverse
  - Interprétation des résultats
"""

from __future__ import annotations
from pathlib import Path
import json
from datetime import datetime


def _medal(rank: int) -> str:
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")


def _risk_emoji(risk: str) -> str:
    return {"Élevé": "🔴", "Modéré": "🟡", "Faible": "🟢"}.get(risk, "⚪")


def _fat_emoji(score: float) -> str:
    if score >= 70: return "🔴"
    if score >= 50: return "🟡"
    if score >= 30: return "🟠"
    return "🟢"


def generate_report(stats: dict, output_path: str | None = None) -> str:
    """
    Génère le rapport automatique post-match en Markdown.

    Parameters
    ----------
    stats       : dict chargé depuis le JSON de stats
    output_path : si fourni, sauvegarde le fichier .md à cet emplacement

    Returns
    -------
    str : contenu Markdown du rapport
    """
    now      = datetime.now().strftime("%d/%m/%Y à %H:%M")
    video    = Path(stats.get("video", "match")).name
    duree    = stats.get("duree_secondes", 0)
    frames   = stats.get("frames_traitees", 0)
    joueurs  = stats.get("joueurs", {})
    passes   = stats.get("total_passes", 0)
    poss_eq  = stats.get("possession_equipes", {})
    subst    = stats.get("substitutions_recommandees", {})
    tactical = stats.get("tactical_engine", {})
    ml       = stats.get("ml_pipeline", {})

    lines = []

    # ── En-tête ───────────────────────────────────────────────────────────────
    lines += [
        f"# 📋 Rapport de match automatique",
        f"",
        f"**Vidéo analysée :** `{video}`  ",
        f"**Durée analysée :** {duree}s ({frames} frames)  ",
        f"**Généré le :** {now}  ",
        f"**Joueurs détectés :** {len(joueurs)}",
        f"",
        "---",
        "",
    ]

    # ── Résumé global ─────────────────────────────────────────────────────────
    lines += ["## 1. Résumé du match", ""]

    if poss_eq:
        for eq, pct in poss_eq.items():
            num = eq.replace("equipe_", "")
            lines.append(f"- **Équipe {num}** — possession : **{pct}%**")
    lines.append(f"- **Passes réussies totales :** {passes}")

    form_e1 = stats.get("formation_E1", "—")
    form_e2 = stats.get("formation_E2", "—")
    lines += [
        f"- **Formation Équipe 1 observée :** {form_e1}",
        f"- **Formation Équipe 2 observée :** {form_e2}",
        "",
    ]

    # Simulation formations S3
    sim = tactical.get("simulation_formations", {})
    if sim and "formation_recommandee" in sim:
        lines += [
            f"**Formation recommandée par l'IA :** `{sim['formation_recommandee']}`  ",
            f"_{sim.get('raison', '')}_",
            "",
        ]

    lines += ["---", ""]

    # ── Performances individuelles ─────────────────────────────────────────────
    lines += ["## 2. Performances individuelles", ""]

    sorted_players = sorted(joueurs.values(), key=lambda p: p.get("classement", 99))

    lines += ["### 🏆 Classement général", ""]
    lines += ["| Rang | Joueur | Équipe | Poste | Score | Fatigue | Blessure | Distance | Passes |"]
    lines += ["|------|--------|--------|-------|-------|---------|----------|----------|--------|"]
    for p in sorted_players:
        rang = p.get("classement", "?")
        med  = _medal(rang) if isinstance(rang, int) and rang <= 3 else f"#{rang}"
        fat  = p.get("fatigue_score", 0)
        inj  = p.get("injury_risk", "Faible")
        lines.append(
            f"| {med} | P{p['player_id']} | E{p.get('equipe','?')} "
            f"| {p.get('poste') or '?'} "
            f"| **{p.get('score',0):.1f}** "
            f"| {_fat_emoji(fat)} {fat:.1f}% "
            f"| {_risk_emoji(inj)} {inj} "
            f"| {p.get('distance_m',0):.0f}m "
            f"| {p.get('passes_reussies',0)}/{p.get('passes_tentees',0)} |"
        )
    lines += [""]

    # Top 3 et Flop
    if len(sorted_players) >= 3:
        lines += ["### ⭐ Top 3 performeurs", ""]
        for p in sorted_players[:3]:
            lines.append(
                f"- **P{p['player_id']}** ({p.get('poste','?')}, E{p.get('equipe','?')}) — "
                f"Score {p.get('score',0):.1f}/100 · "
                f"Vitesse moy. {p.get('vitesse_moyenne_kmh',0):.1f} km/h · "
                f"{p.get('passes_reussies',0)}/{p.get('passes_tentees',0)} passes"
            )
        lines += [""]

    under = [p for p in sorted_players if p.get("sous_performance")]
    if under:
        lines += ["### ⚠️ Joueurs en sous-performance", ""]
        for p in under:
            lines.append(
                f"- **P{p['player_id']}** — Score {p.get('score',0):.1f}/100 · "
                f"Fatigue {p.get('fatigue_score',0):.1f}% · "
                f"Risque {p.get('injury_risk','?')}"
            )
        lines += [""]

    lines += ["---", ""]

    # ── Fatigue & Blessures ───────────────────────────────────────────────────
    lines += ["## 3. Fatigue & Risques blessure", ""]

    critical = [p for p in sorted_players if p.get("fatigue_score", 0) >= 70]
    moderate = [p for p in sorted_players if 50 <= p.get("fatigue_score", 0) < 70]
    high_inj = [p for p in sorted_players if p.get("injury_risk") == "Élevé"]

    if critical:
        lines += [f"**🔴 Fatigue critique (> 70%) :** " +
                  ", ".join(f"P{p['player_id']} ({p['fatigue_score']:.1f}%)" for p in critical)]
    if moderate:
        lines += [f"**🟡 Fatigue modérée (50-70%) :** " +
                  ", ".join(f"P{p['player_id']} ({p['fatigue_score']:.1f}%)" for p in moderate)]
    if high_inj:
        lines += ["", f"**🚨 Risque blessure élevé :** " +
                  ", ".join(f"P{p['player_id']} (score {p.get('injury_score',0):.1f}/100)"
                            for p in high_inj)]
    if not critical and not moderate and not high_inj:
        lines += ["✅ Aucun joueur en état physique critique."]

    lines += ["", "---", ""]

    # ── Substitutions ─────────────────────────────────────────────────────────
    lines += ["## 4. Substitutions recommandées par l'IA", ""]

    subst_s3 = tactical.get("substitutions_s3", subst)
    if not subst_s3:
        lines += ["✅ Aucune substitution nécessaire selon le moteur de décision.", ""]
    else:
        for k, r in sorted(subst_s3.items(), key=lambda x: x[1].get("score", 99)):
            rem_id = r.get("remplacant_id")
            tcs    = r.get("tcs", {})
            lines += [
                f"### 🔄 Sortir P{r['player_id']} "
                f"({r.get('poste','?')}, E{r.get('equipe','?')})",
                f"- Score : {r.get('score',0):.1f}/100 · "
                f"Fatigue : {r.get('fatigue_score',0):.1f}%",
            ]
            for raison in r.get("raisons", []):
                lines.append(f"- _{raison}_")
            if rem_id:
                tcs_str = f" — TCS {tcs['score']:.0f}/100 ({tcs['label']})" if tcs else ""
                lines.append(f"- **Remplaçant suggéré :** P{rem_id}{tcs_str}")
            lines += [""]

    lines += ["---", ""]

    # ── Analyse adverse ───────────────────────────────────────────────────────
    opp = tactical.get("analyse_adverse", {})
    if opp and "error" not in opp:
        lines += ["## 5. Analyse de l'équipe adverse", ""]
        wf = opp.get("joueur_plus_faible", {})
        wt = opp.get("joueur_plus_fatigue", {})
        lines += [
            f"- **Joueur adverse le plus faible :** P{wf.get('player_id','?')} "
            f"({wf.get('poste','?')}) — score {wf.get('score','?')}/100",
            f"- **Joueur adverse le plus fatigué :** P{wt.get('player_id','?')} "
            f"— fatigue {wt.get('fatigue_score','?')}%",
            f"- **Zone défensive faible :** {opp.get('zone_faible') or '—'}",
            "",
            "**Opportunités identifiées :**",
        ]
        for opp_txt in opp.get("opportunites", []):
            lines.append(f"- {opp_txt}")
        lines += ["", "---", ""]

    # ── Modèles ML ────────────────────────────────────────────────────────────
    if ml and "error" not in ml:
        lines += ["## 6. Résultats Machine Learning", ""]
        comp = ml.get("comparaison_modeles", [])
        for m_info in comp:
            lines.append(
                f"- **{m_info['modèle']}** ({m_info['tâche']}) — "
                f"{m_info['métrique']} | entraînement : {m_info['temps_train']}"
            )
        lines += ["", "---", ""]

    # ── Interprétation globale ─────────────────────────────────────────────────
    lines += ["## 7. Interprétation des résultats", ""]

    nb_players = len(joueurs)
    nb_subst   = len(subst_s3)
    avg_score  = (sum(p.get("score", 0) for p in joueurs.values()) / nb_players
                  if nb_players > 0 else 0)
    avg_fat    = (sum(p.get("fatigue_score", 0) for p in joueurs.values()) / nb_players
                  if nb_players > 0 else 0)

    lines += [
        f"Sur les {nb_players} joueurs analysés, le score de performance moyen est de "
        f"**{avg_score:.1f}/100** avec une fatigue moyenne de **{avg_fat:.1f}%**.",
        "",
    ]

    if nb_subst == 0:
        lines.append("L'ensemble des joueurs présente un profil physique et tactique satisfaisant "
                     "— aucun changement urgent n'est recommandé par le moteur de décision.")
    elif nb_subst == 1:
        lines.append("Un joueur nécessite une attention particulière et une substitution est recommandée "
                     "pour préserver l'intégrité physique de l'équipe.")
    else:
        lines.append(f"{nb_subst} joueurs présentent des indicateurs de fatigue ou de sous-performance "
                     f"préoccupants. Des substitutions rapides sont recommandées pour maintenir le niveau collectif.")

    if sim and "formation_recommandee" in sim:
        lines += [
            "",
            f"Sur le plan tactique, la formation **{sim['formation_recommandee']}** est identifiée "
            f"comme la plus adaptée au contexte actuel de l'équipe.",
        ]

    lines += [
        "",
        "---",
        "",
        "_Rapport généré automatiquement par Football Decision Intelligence AI — Semaine 4_",
    ]

    content = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

    return content


def generate_report_from_file(stats_path: str, output_path: str | None = None) -> str:
    """Charge le JSON et génère le rapport."""
    with open(stats_path, encoding="utf-8") as f:
        stats = json.load(f)
    return generate_report(stats, output_path)