import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Football Decision Intelligence AI",
    page_icon="⚽",
    layout="wide",
)

INPUT_DIR  = Path("input_videos")
OUTPUT_DIR = Path("output_videos")
OUTPUT_DIR.mkdir(exist_ok=True)

TEAM_COLORS = {1: "#3a86ff", 2: "#ff006e"}
POSTE_ORDER = ["Gardien", "Défenseur", "Milieu", "Attaquant"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def stats_path_for(video_path: Path) -> Path:
    return OUTPUT_DIR / (video_path.stem + "_stats.json")

def output_video_for(video_path: Path) -> Path:
    return OUTPUT_DIR / (video_path.stem + "_tracking.mp4")

def load_stats(path: Path) -> dict | None:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None

def players_df(stats: dict) -> pd.DataFrame:
    rows = []
    for p in stats["joueurs"].values():
        rows.append({
            "ID":              f"P{p['player_id']}",
            "Équipe":          f"E{p['equipe']}" if p["equipe"] else "?",
            "Poste":           p["poste"] or "?",
            "Score":           p["score"],
            "Rang":            p["classement"],
            "Fatigue (%)":     p.get("fatigue_score", 0.0),
            "Risque Blessure": p.get("injury_risk", "Faible"),
            "Vit. moy (km/h)": p["vitesse_moyenne_kmh"],
            "Distance (m)":    p["distance_m"],
            "Passes tent.":    p["passes_tentees"],
            "Passes réuss.":   p["passes_reussies"],
            "Taux pass. (%)":  p["taux_reussite_passes"],
            "Possession (%)":  p["possession_pct"],
            "Sous-perf.":      p["sous_performance"],
            "_team":           p["equipe"],
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Rang")
    return df

def color_row(row):
    team = row["_team"]
    base = TEAM_COLORS.get(team, "#ffffff")
    bg   = base + "22"
    if row["Sous-perf."]:
        bg = "#ff444422"
    return [f"background-color: {bg}"] * len(row)

def _stream_to_file(proc, log_path: Path) -> None:
    with open(log_path, "w", buffering=1, encoding="utf-8") as f:
        for line in proc.stdout:
            f.write(line)
    proc.wait()

def _read_log(log_path: Path) -> list[str]:
    try:
        return log_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []

@st.cache_resource
def _get_qtable():
    """Charge ou entraîne la Q-Table (une seule fois par session Streamlit)."""
    try:
        from tracking.reinforcement import load_or_train_qtable
        q_table, train_stats = load_or_train_qtable()
        return q_table, train_stats
    except Exception as e:
        return None, {"error": str(e)}


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("⚽ Football Decision Intelligence AI")
st.sidebar.markdown("---")

input_videos = sorted(INPUT_DIR.glob("*.mp4")) + sorted(INPUT_DIR.glob("*.avi"))
video_names  = [v.name for v in input_videos]

selected_name = st.sidebar.selectbox(
    "Vidéo d'entrée",
    options=video_names,
    index=0 if video_names else None,
    placeholder="Aucune vidéo trouvée",
)

uploaded = st.sidebar.file_uploader("… ou uploader une vidéo", type=["mp4", "avi", "mov"])
if uploaded is not None:
    dest = INPUT_DIR / uploaded.name
    with open(dest, "wb") as f:
        f.write(uploaded.read())
    st.sidebar.success(f"Vidéo sauvegardée : {uploaded.name}")
    selected_name = uploaded.name

selected_video = INPUT_DIR / selected_name if selected_name else None
tracking       = st.session_state.get("tracking", False)

run_btn  = st.sidebar.button("▶ Lancer le tracking", disabled=selected_video is None or tracking,
                              use_container_width=True, type="primary")
stop_btn = st.sidebar.button("⏹ Arrêter", disabled=not tracking, use_container_width=True)

st.sidebar.markdown("---")
if selected_video:
    sp = stats_path_for(selected_video)
    if sp.exists():
        mtime = time.strftime("%d/%m/%Y %H:%M", time.localtime(sp.stat().st_mtime))
        st.sidebar.caption(f"Stats existantes ({mtime})")
    else:
        st.sidebar.caption("Aucune stats pour cette vidéo")

# ── Q-Table chargée en arrière-plan ──────────────────────────────────────────
q_table, rl_train_stats = _get_qtable()

# ── Start / stop tracking ─────────────────────────────────────────────────────
if run_btn and selected_video:
    log_path = OUTPUT_DIR / f"{selected_video.stem}_tracking.log"
    log_path.unlink(missing_ok=True)
    proc = subprocess.Popen(
        [sys.executable, "-u", "tracking_foot.py", str(selected_video)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    threading.Thread(target=_stream_to_file, args=(proc, log_path), daemon=True).start()
    st.session_state.update(proc=proc, log_path=str(log_path),
                             tracking=True, tracking_video=selected_video.name)
    st.rerun()

if stop_btn:
    proc = st.session_state.get("proc")
    if proc:
        proc.terminate()
    st.session_state["tracking"] = False
    st.rerun()

# ── Live progress ─────────────────────────────────────────────────────────────
@st.fragment(run_every=0.5)
def _tracking_progress() -> None:
    if not st.session_state.get("tracking"):
        return
    proc     = st.session_state.get("proc")
    log_path = Path(st.session_state.get("log_path", ""))
    vid_name = st.session_state.get("tracking_video", "")
    running  = proc is not None and proc.poll() is None
    if running:
        lines = _read_log(log_path)
        pct_val, pct_text = 0.0, "Démarrage…"
        for line in reversed(lines):
            if "/" in line and "frames" in line:
                try:
                    pct_val  = int(float(line.split("(")[1].split("%")[0])) / 100
                    pct_text = line.strip()
                    break
                except Exception:
                    pass
        st.info(f"Tracking en cours — **{vid_name}**")
        st.progress(pct_val, text=pct_text)
    else:
        rc = proc.returncode if proc else -1
        st.session_state["tracking"] = False
        if rc == 0:
            st.success(f"Tracking terminé — **{vid_name}**")
        else:
            st.warning(f"Tracking interrompu — **{vid_name}** (stats partielles sauvegardées)")
        st.rerun()

_tracking_progress()

# ── Load stats ────────────────────────────────────────────────────────────────
stats = None
if selected_video:
    stats = load_stats(stats_path_for(selected_video))

if stats is None:
    st.title("⚽ Football Decision Intelligence AI")
    st.info("Sélectionnez une vidéo dans la barre latérale et lancez le tracking pour afficher les statistiques.")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.title(f"⚽ {Path(stats['video']).name}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Durée",            f"{stats['duree_secondes']}s")
col2.metric("Frames traitées",  stats["frames_traitees"])
col3.metric("Passes réussies",  stats["total_passes"])
col4.metric("Joueurs détectés", len(stats["joueurs"]))

# ══════════════════════════════════════════════════════════════════════════════
# ONGLETS PRINCIPAUX : Avant match / Pendant match / Après match  (Tâche 4.4)
# ══════════════════════════════════════════════════════════════════════════════
tab_avant, tab_pendant, tab_apres = st.tabs([
    "🔮 Avant le match",
    "⚡ Pendant le match",
    "📊 Après le match",
])

tactical = stats.get("tactical_engine", {})


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  ONGLET 1 — AVANT LE MATCH                                                ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab_avant:
    st.subheader("🗺️ Tactique recommandée")

    sim = tactical.get("simulation_formations", {})
    if sim and "error" not in sim:
        best_form = sim.get("formation_recommandee", "?")
        st.success(f"**Formation recommandée : {best_form}**")
        st.caption(sim.get("raison", ""))

        formations_data = sim.get("formations", {})
        if formations_data:
            sim_rows = []
            for fname, fdata in formations_data.items():
                sim_rows.append({
                    "Formation":       fname,
                    "Rang":            fdata.get("rang", "?"),
                    "Score global":    fdata.get("score_global", 0),
                    "xG simplifié":    fdata.get("xg_simplifie", 0),
                    "Pressing":        fdata.get("pressing", 0),
                    "Risque défensif": fdata.get("risque_defensif", 0),
                })
            sim_df = pd.DataFrame(sim_rows).sort_values("Rang")

            def _form_highlight(row):
                if row["Formation"] == best_form:
                    return ["background-color:#00cc4422;font-weight:bold"] * len(row)
                return [""] * len(row)

            st.dataframe(
                sim_df.style
                      .apply(_form_highlight, axis=1)
                      .format({"Score global": "{:.1f}", "xG simplifié": "{:.2f}",
                                "Pressing": "{:.1f}", "Risque défensif": "{:.1f}"})
                      .hide(axis="index"),
                use_container_width=True,
            )
            st.bar_chart(
                sim_df[["Formation","Score global","Pressing","xG simplifié"]].set_index("Formation"),
                color=["#3a86ff","#ff006e","#06d6a0"],
            )
    else:
        st.info("Simulation de formations non disponible.")

    st.markdown("---")
    st.subheader("🕵️ Analyse adverse — Joueurs & zones à cibler")

    opp = tactical.get("analyse_adverse", {})
    if opp and "error" not in opp:
        co1, co2, co3 = st.columns(3)
        wf = opp.get("joueur_plus_faible", {})
        wt = opp.get("joueur_plus_fatigue", {})
        co1.metric("Joueur adverse le + faible",
                   f"P{wf.get('player_id','?')}",
                   f"Score {wf.get('score','?')}/100")
        co2.metric("Joueur adverse le + fatigué",
                   f"P{wt.get('player_id','?')}",
                   f"Fatigue {wt.get('fatigue_score','?')}%")
        co3.metric("Zone défensive faible",
                   opp.get("zone_faible") or "—",
                   f"Score moy. {opp.get('score_zone_faible') or '—'}")

        st.markdown("**Opportunités tactiques :**")
        for opp_txt in opp.get("opportunites", []):
            st.markdown(f"⚡ {opp_txt}")

        analyse_poste = opp.get("analyse_par_poste", {})
        if analyse_poste:
            st.markdown("**Analyse par ligne adverse :**")
            ap_df = pd.DataFrame([
                {"Ligne": k, "Score moyen": v["score_moyen"],
                 "Fatigue moy.": v["fatigue_moy"], "Joueurs": str(v["joueurs"])}
                for k, v in analyse_poste.items()
            ])
            st.dataframe(ap_df.style.hide(axis="index"), use_container_width=True)
    else:
        st.info("Analyse adverse non disponible.")

    st.markdown("---")
    st.subheader("📋 Formations détectées")
    fc1, fc2 = st.columns(2)
    fc1.metric("Formation Équipe 1", stats.get("formation_E1", "—"))
    fc2.metric("Formation Équipe 2", stats.get("formation_E2", "—"))


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  ONGLET 2 — PENDANT LE MATCH                                              ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab_pendant:

    # ── Alertes fatigue live ───────────────────────────────────────────────────
    st.subheader("🚨 Alertes en temps réel")

    joueurs_data = list(stats["joueurs"].values())
    alertes_crit = [p for p in joueurs_data if p.get("fatigue_score", 0) >= 70]
    alertes_mod  = [p for p in joueurs_data if 50 <= p.get("fatigue_score", 0) < 70]

    if alertes_crit:
        for p in alertes_crit:
            st.error(
                f"🔴 **P{p['player_id']}** ({p.get('poste','?')}, E{p.get('equipe','?')}) — "
                f"Fatigue {p['fatigue_score']:.1f}% · Risque {p.get('injury_risk','?')} · "
                f"Score {p['score']:.1f}/100 → **Substitution recommandée**"
            )
    if alertes_mod:
        for p in alertes_mod:
            st.warning(
                f"🟡 **P{p['player_id']}** ({p.get('poste','?')}, E{p.get('equipe','?')}) — "
                f"Fatigue {p['fatigue_score']:.1f}% · À surveiller"
            )
    if not alertes_crit and not alertes_mod:
        st.success("✅ Aucune alerte fatigue — équipe en bonne condition physique.")

    st.markdown("---")

    # ── Tableau joueurs live ───────────────────────────────────────────────────
# ── Tableau joueurs live ───────────────────────────────────────────────────
    st.subheader("👥 État des joueurs")
    df = players_df(stats)
    
    if df.empty:
        st.info("⏳ En attente de données du tracking pour afficher le tableau des joueurs...")
    else:
        c1, c2, c3 = st.columns(3)
        team_filter    = c1.multiselect("Équipe", ["E1","E2"], default=["E1","E2"], key="pend_team")
        poste_filter   = c2.multiselect("Poste", POSTE_ORDER+["?"], default=POSTE_ORDER+["?"], key="pend_poste")
        show_underperf = c3.checkbox("Sous-performances uniquement", value=False, key="pend_under")

        mask = df["Équipe"].isin(team_filter) & df["Poste"].isin(poste_filter)
        if show_underperf:
            mask &= df["Sous-perf."]
        filtered = df[mask]

        if filtered.empty:
            st.warning("Aucun joueur ne correspond aux filtres sélectionnés.")
        else:
            cols_to_hide = [c for c in filtered.columns if c.startswith("_") or c == "Sous-perf."]
            styled = (
                filtered.style.apply(color_row, axis=1)
                .format({k: v for k, v in {
                    "Score":"{:.1f}","Fatigue (%)":"{:.1f}",
                    "Vit. moy (km/h)":"{:.2f}","Distance (m)":"{:.1f}",
                    "Taux pass. (%)":"{:.1f}","Possession (%)":"{:.1f}",
                }.items() if k in filtered.columns})
                .hide(axis="index")
            )
            if cols_to_hide:
                styled = styled.hide(cols_to_hide, axis="columns")
            st.dataframe(styled, use_container_width=True, height=420)
            st.caption("🔵 Équipe 1 · 🔴 Équipe 2 · fond rouge = sous-performance")

    st.markdown("---")

    # ── Substitutions recommandées ─────────────────────────────────────────────
    st.subheader("🔄 Substitutions recommandées (Moteur S3)")
    subst_s3 = tactical.get("substitutions_s3", stats.get("substitutions_recommandees", {}))

    if not subst_s3:
        st.success("Aucune substitution nécessaire.")
    else:
        st.warning(f"{len(subst_s3)} substitution(s) recommandée(s)")
        for _, r in sorted(subst_s3.items(), key=lambda x: x[1]["score"]):
            team_color = TEAM_COLORS.get(r.get("equipe"), "#888")
            with st.container(border=True):
                c1, c2, c3 = st.columns([1, 2, 2])
                c1.markdown(
                    f"<div style='font-size:1.5rem;font-weight:bold;color:{team_color}'>"
                    f"P{r['player_id']}</div>"
                    f"<div style='color:{team_color}'>E{r.get('equipe','?')} · {r.get('poste','?')}</div>"
                    f"<div>Score : <b>{r['score']:.1f}</b>/100</div>"
                    f"<div>Fatigue : <b>{r.get('fatigue_score',0):.1f}%</b></div>",
                    unsafe_allow_html=True,
                )
                c2.markdown("**Raisons :**")
                for raison in r.get("raisons", []):
                    c2.markdown(f"- {raison}")
                if r.get("remplacant_id"):
                    tcs = r.get("tcs", {})
                    tcs_col = "#00cc44" if tcs.get("score",0) >= 75 else (
                              "#ff9900" if tcs.get("score",0) >= 55 else "#ff4444")
                    c3.markdown(f"**Remplaçant : P{r['remplacant_id']}**")
                    if tcs:
                        c3.markdown(
                            f"<div style='padding:8px;border-radius:6px;"
                            f"background:{tcs_col}22;border:1px solid {tcs_col}'>"
                            f"<b>TCS : {tcs['score']:.1f}/100 — {tcs['label']}</b><br>"
                            f"Fraîcheur : {tcs['fraicheur']:.0f}% · "
                            f"Perf. : {tcs['perf_relative']:.0f}%"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

    st.markdown("---")

    # ── Assistant IA Live (Q-Learning) ────────────────────────────────────────
    st.subheader("🤖 Assistant Tactique IA — Q-Learning (Tâches 4.1 & 4.2)")

    if q_table is None:
        st.error(f"Q-Table indisponible : {rl_train_stats.get('error','?')}")
    else:
        if rl_train_stats.get("loaded_from_cache"):
            st.caption("Q-Table chargée depuis le cache.")
        else:
            st.caption(
                f"Q-Table entraînée ({rl_train_stats.get('n_episodes',0)} épisodes) — "
                f"Récompense moyenne : {rl_train_stats.get('reward_mean',0):.1f} · "
                f"500 derniers épisodes : {rl_train_stats.get('reward_last_500',0):.1f}"
            )

        with st.container(border=True):
            sc1, sc2, sc3, sc4 = st.columns(4)
            sim_minute  = sc1.slider("Minute", 0, 90, 70, step=5)
            sim_score   = sc2.number_input("Différence score", min_value=-3, max_value=3, value=-1)
            sim_fat_lbl = sc3.select_slider("Fatigue équipe", ["Faible","Modérée","Critique"], value="Modérée")
            sim_poss    = sc4.radio("Possession", ["Notre équipe","Adversaire"], index=0)

            fat_map  = {"Faible": 20.0, "Modérée": 55.0, "Critique": 85.0}
            poss_map = {"Notre équipe": 1, "Adversaire": 0}

            from tracking.reinforcement import get_ia_action
            ia_result = get_ia_action(
                q_table,
                minute=sim_minute,
                score_diff=int(sim_score),
                fatigue_pct=fat_map[sim_fat_lbl],
                possession=poss_map[sim_poss],
            )

            st.markdown(
                f"### 🎯 Décision IA : `{ia_result['action_label']}`"
                f"  *(confiance : {ia_result['confiance']:.1f}%)*"
            )

            qv_df = pd.DataFrame([
                {"Action": label, "Valeur Q": val}
                for label, val in ia_result["q_values"].items()
            ]).sort_values("Valeur Q", ascending=False)
            st.bar_chart(qv_df.set_index("Action"), color="#06d6a0")

    st.markdown("---")

    # ── Graphiques rapides ─────────────────────────────────────────────────────
    st.subheader("📈 Métriques live")
    df_full = players_df(stats)
    if not df_full.empty:
        met = st.selectbox("Métrique", ["Score","Fatigue (%)","Vit. moy (km/h)","Distance (m)"])
        chart_df = (
            df_full[["ID","Équipe",met]]
            .sort_values(met, ascending=False)
            .assign(couleur=lambda d: d["Équipe"].map({"E1":"#3a86ff","E2":"#ff006e"}))
            .set_index("ID")
        )
        st.bar_chart(chart_df[[met,"couleur"]], color="couleur")

        poss = stats.get("possession_equipes", {})
        if poss:
            poss_df = pd.DataFrame(
                {"Équipe": list(poss.keys()), "Possession (%)": list(poss.values())}
            )
            st.bar_chart(poss_df.set_index("Équipe"), color=["#3a86ff"])


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  ONGLET 3 — APRÈS LE MATCH                                                ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  ONGLET 3 — APRÈS LE MATCH                                                 ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
with tab_apres:

    # ── Performance S2 complète ────────────────────────────────────────────────
    st.subheader("🧠 Analyse Performance IA (Semaine 2)")

    fat_rows = []
    for p in stats["joueurs"].values():
        fat_rows.append({
            "Joueur":          f"P{p['player_id']}",
            "Équipe":          f"E{p['equipe']}" if p["equipe"] else "?",
            "Poste":           p.get("poste") or "?",
            "Score":           p.get("score", 0.0),
            "Fatigue (%)":     p.get("fatigue_score", 0.0),
            "Label fatigue":   p.get("fatigue_label", "—"),
            "Sprint (%)":      p.get("sprint_pct", 0.0),
            "Risque blessure": p.get("injury_risk", "Faible"),
            "Score blessure":  p.get("injury_score", 0.0),
            "Risque ML (RF)":  p.get("injury_risk_ml") or "—",
            "Perf. future MLP":p.get("future_performance_mlp") or "—",
        })
    
    fat_df = pd.DataFrame(fat_rows)

    if fat_df.empty:
        st.info("⏳ En attente de données du tracking pour générer l'analyse de performance complète...")
    else:
        # Le tri ne s'exécute que si des colonnes valides existent
        fat_df = fat_df.sort_values("Score", ascending=False)

        RISK_COLOR = {"Élevé": "#ff000033", "Modéré": "#ff990033", "Faible": "#00cc4422"}

        def _fat_color(val):
            if val >= 70: return "background-color:#ff444444;color:#ff0000;font-weight:bold"
            if val >= 50: return "background-color:#ff990033"
            if val >= 30: return "background-color:#ffcc0022"
            return ""

        st.dataframe(
            fat_df.style
                  .map(_fat_color, subset=["Fatigue (%)"])
                  .map(lambda v: f"background-color:{RISK_COLOR.get(v,'')}", subset=["Risque blessure","Risque ML (RF)"])
                  .format({"Score":"{:.1f}","Fatigue (%)":"{:.1f}","Sprint (%)":"{:.1f}","Score blessure":"{:.1f}"})
                  .hide(axis="index"),
            use_container_width=True,
        )

    # ML résultats
    ml = stats.get("ml_pipeline", {})
    if ml and "error" not in ml:
        st.markdown("---")
        st.markdown("**Modèles ML — Comparaison RF / XGBoost / MLP**")
        comp = ml.get("comparaison_modeles", [])
        if comp:
            st.dataframe(pd.DataFrame(comp).style.hide(axis="index"), use_container_width=True)

        col_rf, col_xgb = st.columns(2)
        rf = ml.get("random_forest", {})
        if rf and "error" not in rf:
            with col_rf:
                st.markdown(f"**🌲 Random Forest** — Accuracy `{rf['accuracy']*100:.1f}%`")
                fi = rf.get("feature_importance", {})
                if fi:
                    fi_df = pd.DataFrame({"Feature": list(fi.keys()), "Importance": list(fi.values())}).sort_values("Importance", ascending=False)
                    st.bar_chart(fi_df.set_index("Feature"), color="#3a86ff")
        xgb_r = ml.get("xgboost", {})
        if xgb_r and "error" not in xgb_r:
            with col_xgb:
                st.markdown(f"**⚡ XGBoost** — Accuracy `{xgb_r['accuracy']*100:.1f}%`")
        mlp = ml.get("mlp", {})
        if mlp and "error" not in mlp:
            st.markdown(f"**🧠 MLP** — MAE `{mlp['mae']:.1f}` pts | RMSE `{mlp['rmse']:.1f}` pts")
            lc = mlp.get("loss_curve", [])
            if lc:
                lc_df = pd.DataFrame({"Époque": [i*5 for i in range(len(lc))], "Loss": lc})
                st.line_chart(lc_df.set_index("Époque"), color="#ff006e")

    st.markdown("---")

    # ── Scatter vitesse vs distance ────────────────────────────────────────────
    st.subheader("📉 Vitesse vs Distance parcourue")
    df_full = players_df(stats)
    if not df_full.empty:
        scatter_df = df_full[["ID","Équipe","Vit. moy (km/h)","Distance (m)","Score","Sous-perf."]].copy()
        scatter_df["Couleur"] = scatter_df["Équipe"].map({"E1":"#3a86ff","E2":"#ff006e"})
        st.scatter_chart(scatter_df, x="Vit. moy (km/h)", y="Distance (m)",
                         color="Couleur", size="Score")

    st.markdown("---")

    # ── Comparaison Coach vs IA (Tâche 4.3) ──────────────────────────────────
    st.subheader("🆚 Comparaison Coach humain vs IA (Tâche 4.3)")

    if q_table is None:
        st.warning("Q-Table non disponible — comparaison impossible.")
    else:
        try:
            from tracking.reinforcement import compare_coach_vs_ia
            comparaison = compare_coach_vs_ia(q_table)
            taux = comparaison[0]["taux_accord_global"] if comparaison else 0

            st.metric("Taux d'accord Coach ↔ IA", f"{taux}%")

            for sc in comparaison:
                accord_icon = "✅" if sc["accord"] else "❌"
                with st.expander(f"{accord_icon} {sc['contexte']}"):
                    cc1, cc2 = st.columns(2)
                    cc1.markdown(f"**👤 Coach humain :** `{sc['coach']}`")
                    cc2.markdown(f"**🤖 Décision IA :** `{sc['ia']}` *(confiance {sc['confiance_ia']:.1f}%)*")
                    qv_df2 = pd.DataFrame([
                        {"Action": a, "Valeur Q": v}
                        for a, v in sc["q_values"].items()
                    ]).sort_values("Valeur Q", ascending=False)
                    st.bar_chart(qv_df2.set_index("Action"), color="#06d6a0")
        except Exception as e:
            st.error(f"Erreur comparaison : {e}")

    st.markdown("---")

    # ── Rapport automatique (Tâche 4.5) ──────────────────────────────────────
    st.subheader("📋 Rapport automatique post-match (Tâche 4.5)")

    try:
        from tracking.report import generate_report
        report_path = str(OUTPUT_DIR / (Path(stats["video"]).stem + "_rapport.md"))
        rapport_md  = generate_report(stats, output_path=report_path)

        st.download_button(
            label="⬇️ Télécharger le rapport (.md)",
            data=rapport_md,
            file_name=Path(report_path).name,
            mime="text/markdown",
        )
        with st.expander("📄 Aperçu du rapport", expanded=False):
            st.markdown(rapport_md)
    except Exception as e:
        st.error(f"Erreur génération rapport : {e}")

    st.markdown("---")

    # ── Vidéo annotée ─────────────────────────────────────────────────────────
    st.subheader("🎬 Vidéo annotée")
    out_video = output_video_for(selected_video)
    if out_video.exists():
        st.video(str(out_video))
    else:
        st.info("Aucune vidéo annotée trouvée. Lancez le tracking pour en générer une.")