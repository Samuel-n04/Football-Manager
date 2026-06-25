import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ── football_analysis modules ──────────────────────────────────────────────────
_FA = Path(__file__).parent.parent / "football_analysis"
sys.path.insert(0, str(_FA))

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY = True
except ImportError:
    PLOTLY = False

from data.generate_dataset import generate_player_dataset, generate_opponent_data
from performance.performance_score import PerformanceScoreCalculator
from performance.fatigue_score import FatigueScoreCalculator
from performance.injury_risk import InjuryRiskCalculator
from ml_models.injury_predictor import InjuryPredictor
from ml_models.performance_predictor import PerformancePredictor
from decision_engine.rules_engine import RulesEngine
from decision_engine.substitution_recommender import SubstitutionRecommender
from decision_engine.tactical_analyzer import TacticalAnalyzer
from decision_engine.tactical_simulator import TacticalSimulator
from report.report_generator import ReportGenerator

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Football Decision Intelligence AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .kpi-value { font-size: 2rem; font-weight: bold; color: #e94560; }
    .kpi-label { font-size: 0.85rem; color: #aaa; }
    .alert-critique {
        background: #5a0000;
        border-left: 4px solid #e94560;
        padding: 8px; border-radius: 4px; margin: 4px 0;
    }
    .alert-haute {
        background: #4a3000;
        border-left: 4px solid #ffa500;
        padding: 8px; border-radius: 4px; margin: 4px 0;
    }
    .alert-basse {
        background: #003020;
        border-left: 4px solid #00c9a7;
        padding: 8px; border-radius: 4px; margin: 4px 0;
    }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1117 0%, #161b22 100%); }
</style>
""", unsafe_allow_html=True)

# ── Constantes ─────────────────────────────────────────────────────────────────
INPUT_DIR  = Path("input_videos")
OUTPUT_DIR = Path("output_videos")
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

POSTE_TO_POSITION = {
    "Gardien":   "GK",
    "Défenseur": "CB",
    "Milieu":    "CM",
    "Attaquant": "ST",
}

# ── Session state ──────────────────────────────────────────────────────────────
for _k, _v in {
    "tracking":    False,
    "sim_actions": [],
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Helpers tracking ───────────────────────────────────────────────────────────

def stats_path_for(p: Path) -> Path:
    return OUTPUT_DIR / (p.stem + "_stats.json")

def output_video_for(p: Path) -> Path:
    return OUTPUT_DIR / (p.stem + "_tracking.mp4")

def find_output_video(p: Path) -> Path | None:
    """Cherche la vidéo annotée dans l'ordre de priorité."""
    stem = p.stem
    for candidate in [
        OUTPUT_DIR / (stem + "_tracking_web.mp4"),
        OUTPUT_DIR / (stem + "_tracking.mp4"),
        OUTPUT_DIR / (stem + "_tracking.avi"),
        OUTPUT_DIR / "output_video.avi",
    ]:
        if candidate.exists() and candidate.stat().st_size > 10_000:
            return candidate
    return None

def is_video_valid(path: Path) -> tuple[bool, int]:
    """Retourne (valide, nb_frames). Valide = le fichier peut être lu par OpenCV."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return False, 0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frames > 0, frames

def convert_to_web_mp4(src: Path) -> Path | None:
    """
    Convertit la vidéo en MP4 H264 compatible navigateur.
    Essaie ffmpeg d'abord, puis ré-encodage OpenCV, puis retourne None.
    """
    dst = OUTPUT_DIR / (src.stem + "_web.mp4")
    if dst.exists() and dst.stat().st_size > 10_000:
        return dst

    # Tentative ffmpeg
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-c:v", "libx264", "-preset", "fast", "-crf", "23",
             "-movflags", "+faststart", "-an", str(dst)],
            capture_output=True, timeout=600,
        )
        if r.returncode == 0 and dst.exists() and dst.stat().st_size > 10_000:
            return dst
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Tentative ré-encodage OpenCV (avc1 / H264)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return None
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w_v   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_v   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    for _cc in ("avc1", "H264"):
        _w = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*_cc), fps_v, (w_v, h_v))
        if _w.isOpened():
            writer = _w
            break

    if writer is None:
        cap.release()
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
    cap.release()
    writer.release()

    if dst.exists() and dst.stat().st_size > 10_000:
        return dst
    return None

def load_stats(path: Path) -> dict | None:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None

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


# ── Conversion tracking JSON → DataFrame ML ────────────────────────────────────

def tracking_to_df(stats: dict) -> pd.DataFrame:
    """Convertit les stats du tracking vidéo vers le format attendu par les modules ML."""
    joueurs  = stats["joueurs"]
    dur_min  = stats["duree_secondes"] / 60.0

    rows = []
    for p in joueurs.values():
        pos = POSTE_TO_POSITION.get(p.get("poste"), "CM")
        spd = float(p["vitesse_moyenne_kmh"])
        dist = float(p["distance_m"])
        pass_acc = float(p["taux_reussite_passes"]) if p["taux_reussite_passes"] else 70.0
        sprints  = max(5, int(spd * 1.5))
        accels   = sprints * 2
        recov    = int(p.get("passes_recues", 0))
        pressing = max(0, int(p.get("passes_tentees", 0)) // 3)

        rows.append({
            "PlayerID":        p["player_id"],
            "Team":            p["equipe"] or 0,
            "Position":        pos,
            "MinutesPlayed":   round(dur_min, 1),
            "SpeedAvg":        round(spd, 2),
            "SpeedMax":        round(spd * 1.4, 2),
            "SprintCount":     sprints,
            "Distance":        round(dist, 1),
            "Accelerations":   accels,
            "PassAccuracy":    round(pass_acc, 2),
            "Shots":           0,
            "Recoveries":      recov,
            "PressingActions": pressing,
            "Fouls":           0,
            "YellowCards":     0,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    fat_calc  = FatigueScoreCalculator()
    perf_calc = PerformanceScoreCalculator()
    inj_calc  = InjuryRiskCalculator()

    df = fat_calc.calculate(df)
    df = perf_calc.calculate(df)
    df = inj_calc.calculate(df)
    df = fat_calc.classify(df)
    df = perf_calc.classify(df)
    return df


# ── Chargement des données ML (cachées) ──────────────────────────────────────

@st.cache_data
def load_synthetic_data():
    df     = generate_player_dataset(n_players=22, n_matches=15)
    opp_df = generate_opponent_data(n_players=11)
    return df, opp_df

@st.cache_resource
def build_ml_models(df):
    inj_pred  = InjuryPredictor()
    perf_pred = PerformancePredictor()
    inj_pred.train(df)
    perf_pred.train(df)
    return inj_pred, perf_pred

with st.spinner("Chargement des données et entraînement des modèles…"):
    _synth_df, opp_df = load_synthetic_data()
    inj_pred, perf_pred = build_ml_models(_synth_df)

synth_last = _synth_df[_synth_df["MatchID"] == _synth_df["MatchID"].max()].copy()
synth_bench = _synth_df[_synth_df["MatchID"] == _synth_df["MatchID"].max() - 1].copy()
synth_last["Team"]  = (synth_last["PlayerID"] <= 11).map({True: 1, False: 2})
synth_bench["Team"] = (synth_bench["PlayerID"] <= 11).map({True: 1, False: 2})


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚽ Football AI Coach")
    st.markdown("---")

    input_videos = sorted(INPUT_DIR.glob("*.mp4")) + sorted(INPUT_DIR.glob("*.avi"))
    video_names  = [v.name for v in input_videos]

    selected_name = st.selectbox(
        "Vidéo d'entrée",
        options=video_names,
        index=0 if video_names else None,
        placeholder="Aucune vidéo trouvée",
    )

    uploaded = st.file_uploader("ou uploader une vidéo", type=["mp4", "avi", "mov"])
    if uploaded is not None:
        dest = INPUT_DIR / uploaded.name
        with open(dest, "wb") as f:
            f.write(uploaded.read())
        st.success(f"Sauvegardée : {uploaded.name}")
        selected_name = uploaded.name

    selected_video = INPUT_DIR / selected_name if selected_name else None
    tracking = st.session_state.get("tracking", False)

    st.markdown("---")
    run_btn  = st.button("▶ Lancer le tracking",
                         disabled=selected_video is None or tracking,
                         use_container_width=True, type="primary")
    stop_btn = st.button("⏹ Arrêter",
                         disabled=not tracking,
                         use_container_width=True)

    if selected_video:
        sp = stats_path_for(selected_video)
        if sp.exists():
            mtime = time.strftime("%d/%m/%Y %H:%M", time.localtime(sp.stat().st_mtime))
            st.caption(f"Stats disponibles ({mtime})")
        else:
            st.caption("Aucune stats pour cette vidéo")

    st.markdown("---")
    team_choice = st.selectbox("Équipe analysée", ["Équipe A (1)", "Équipe B (2)"])
    minute      = st.slider("Minute du match", 0, 90, 60, 5)
    score_diff  = st.number_input("Différence de score (+ = on gagne)", -3, 3, 0, 1)
    st.markdown("---")
    st.markdown("**Légende alertes**")
    st.markdown("🔴 Critique &nbsp; 🟠 Haute &nbsp; 🟢 Basse")


# ── Tracking start / stop ──────────────────────────────────────────────────────
if run_btn and selected_video:
    log_path = OUTPUT_DIR / f"{selected_video.stem}_tracking.log"
    log_path.unlink(missing_ok=True)
    proc = subprocess.Popen(
        [sys.executable, "-u", "tracking_foot.py", str(selected_video)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    threading.Thread(target=_stream_to_file, args=(proc, log_path), daemon=True).start()
    st.session_state.update(
        proc=proc, log_path=str(log_path),
        tracking=True, tracking_video=selected_video.name,
    )
    st.rerun()

if stop_btn:
    proc = st.session_state.get("proc")
    if proc:
        proc.terminate()
    st.session_state["tracking"] = False
    st.rerun()

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
            st.warning("Tracking interrompu — stats partielles sauvegardées")
        st.rerun()

_tracking_progress()

# ── Chargement stats vidéo + sélection équipe ─────────────────────────────────
video_stats = None
if selected_video:
    video_stats = load_stats(stats_path_for(selected_video))

team_id = 1 if "1" in team_choice else 2

if video_stats:
    full_df    = tracking_to_df(video_stats)
    bench_df   = synth_bench[synth_bench["Team"] == team_id].copy()
    data_label = f"Tracking vidéo : {Path(video_stats['video']).name}"
else:
    full_df    = synth_last.copy()
    bench_df   = synth_bench.copy()
    data_label = "Données simulées (aucune vidéo analysée)"

if full_df.empty:
    st.warning("Aucune donnée disponible. Lancez le tracking ou vérifiez vos fichiers.")
    st.stop()

active_team = full_df[full_df["Team"] == team_id].copy()
bench_team  = bench_df[bench_df["Team"] == team_id].copy() if "Team" in bench_df.columns else bench_df.copy()

# ── Onglets ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab_vid = st.tabs([
    "⬅ Avant le match",
    "🎮 Pendant le match",
    "📊 Après le match",
    "🎥 Vidéo annotée",
])


# =============================================================================
# ONGLET 1 — AVANT LE MATCH
# =============================================================================
with tab1:
    st.header("Analyse pré-match")
    st.caption(f"Source de données : {data_label}")
    col1, col2 = st.columns(2)

    # ── Tactique recommandée ──────────────────────────────────────────────────
    with col1:
        st.subheader("Tactique recommandée")
        sim  = TacticalSimulator()
        tac  = sim.recommend_formation(active_team, {"minute": 0, "score_diff": 0})
        all_results = pd.DataFrame(tac["all_results"])

        st.metric("Formation conseillée", tac["recommended_formation"])
        st.caption(tac["description"])

        if PLOTLY:
            fig = px.bar(
                all_results, x="Formation", y="GlobalScore",
                color="GlobalScore", color_continuous_scale="RdYlGn",
                title="Score global par formation",
            )
            fig.update_layout(
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                font_color="#e6edf3",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(all_results.set_index("Formation")[["GlobalScore"]])

        with st.expander("Détails formations"):
            st.dataframe(
                all_results[["Formation", "Possession", "Pressing", "xG", "DefensiveRisk", "GlobalScore"]],
                use_container_width=True,
            )

    # ── Analyse adverse ───────────────────────────────────────────────────────
    with col2:
        st.subheader("Analyse adverse")
        analyzer = TacticalAnalyzer()
        opp      = analyzer.analyze_opponent(opp_df)

        st.info(opp["summary"])

        st.markdown("**Joueur clé adverse à surveiller**")
        top_opp = opp_df.nlargest(1, "PerformanceScore").iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Joueur #",    int(top_opp["PlayerID"]))
        c2.metric("Poste",       top_opp["Position"])
        c3.metric("Performance", f"{top_opp['PerformanceScore']:.1f}/100")

        weak_zones = opp["weak_zones"]["zones"]
        if PLOTLY:
            fig_zones = go.Figure(go.Bar(
                x=list(weak_zones.keys()),
                y=list(weak_zones.values()),
                marker_color=[
                    "#e94560" if z == opp["weak_zones"]["weakest_zone"] else "#0f3460"
                    for z in weak_zones
                ],
            ))
            fig_zones.update_layout(
                title="Zones vulnérables adverses (%)",
                yaxis_title="%",
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                font_color="#e6edf3",
            )
            st.plotly_chart(fig_zones, use_container_width=True)
        else:
            st.bar_chart(pd.DataFrame.from_dict(weak_zones, orient="index", columns=["%"]))

        st.markdown("**Recommandations tactiques**")
        for rec in opp["tactical_recommendations"]:
            st.markdown(f"- {rec}")

    st.markdown("---")

    # ── Risques pré-match ─────────────────────────────────────────────────────
    st.subheader("Risques pré-match")
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown("**Joueurs adverses en sous-performance**")
        if opp["weak_players"]:
            st.dataframe(pd.DataFrame(opp["weak_players"]), use_container_width=True)
        else:
            st.success("Aucun joueur adverse en sous-performance.")

    with col_r2:
        st.markdown("**Prédiction de performance (prochain match)**")
        preds = []
        for pid in active_team["PlayerID"].unique():
            hist = _synth_df[_synth_df["PlayerID"] == pid].sort_values("MatchID")
            if len(hist) >= 5:
                try:
                    pred = perf_pred.predict_next(hist)
                    preds.append({"Joueur #": pid, "Performance prédite": round(pred, 1)})
                except Exception:
                    pass
        if preds and PLOTLY:
            pred_df = pd.DataFrame(preds)
            fig_pred = px.bar(
                pred_df, x="Joueur #", y="Performance prédite",
                color="Performance prédite", color_continuous_scale="RdYlGn",
                range_color=[0, 100], title="Performance prédite",
            )
            fig_pred.update_layout(
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                font_color="#e6edf3",
            )
            st.plotly_chart(fig_pred, use_container_width=True)
        elif preds:
            st.dataframe(pd.DataFrame(preds), use_container_width=True)
        else:
            st.info("Données insuffisantes pour les prédictions individuelles.")


# =============================================================================
# ONGLET 2 — PENDANT LE MATCH
# =============================================================================
with tab2:
    st.header(f"Suivi en temps réel — Minute {minute}")

    # ── KPIs ──────────────────────────────────────────────────────────────────
    avg_fat  = active_team["FatigueScore"].mean()
    avg_perf = active_team["PerformanceScore"].mean()
    n_inj    = int((active_team["InjuryRisk"] == "Eleve").sum())

    rules        = RulesEngine()
    rule_results = rules.evaluate(active_team)
    n_subs       = int((rule_results["TopAction"] == "SUBSTITUTION").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fatigue moyenne",               f"{avg_fat:.1f} / 100",  delta_color="inverse")
    c2.metric("Performance moyenne",           f"{avg_perf:.1f} / 100")
    c3.metric("Joueurs risque blessure élevé", n_inj,                   delta_color="inverse")
    c4.metric("Remplacements conseillés",      n_subs,                  delta_color="inverse")

    st.markdown("---")
    col1, col2 = st.columns([1.4, 1])

    # ── Scatter Fatigue / Performance ─────────────────────────────────────────
    with col1:
        st.subheader("Fatigue et Performance par joueur")
        if PLOTLY:
            size_col = "Distance" if "Distance" in active_team.columns else None
            fig = px.scatter(
                active_team,
                x="FatigueScore", y="PerformanceScore",
                color="InjuryRisk",
                size=size_col,
                hover_data=["PlayerID", "Position"],
                color_discrete_map={
                    "Faible": "#00c9a7",
                    "Moyen":  "#ffa500",
                    "Eleve":  "#e94560",
                },
                title="Vue joueurs (taille = distance parcourue)",
            )
            fig.add_vline(x=70, line_dash="dash", line_color="orange",
                          annotation_text="Seuil fatigue")
            fig.add_hline(y=50, line_dash="dash", line_color="red",
                          annotation_text="Seuil performance")
            fig.update_layout(
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                font_color="#e6edf3",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.scatter_chart(
                active_team, x="FatigueScore", y="PerformanceScore",
                color="InjuryRisk", size="Distance",
            )

    # ── Alertes actives ───────────────────────────────────────────────────────
    with col2:
        st.subheader("Alertes actives")
        alerts = (
            rule_results[rule_results["AlertCount"] > 0]
            .sort_values("AlertCount", ascending=False)
        )
        if alerts.empty:
            st.success("Aucune alerte active.")
        else:
            for _, row in alerts.iterrows():
                for alert in row["Alerts"]:
                    sev  = alert["severity"]
                    css  = ("alert-critique" if sev == "CRITIQUE"
                            else "alert-haute" if sev == "HAUTE"
                            else "alert-basse")
                    icon = "🔴" if sev == "CRITIQUE" else ("🟠" if sev == "HAUTE" else "🟢")
                    st.markdown(
                        f'<div class="{css}">{icon} <b>Joueur #{int(row["PlayerID"])}</b> — '
                        f'{alert["label"]}<br>'
                        f'Fatigue : {row["FatigueScore"]:.0f} | '
                        f'Performance : {row["PerformanceScore"]:.0f}</div>',
                        unsafe_allow_html=True,
                    )

    st.markdown("---")

    # ── Remplacements proposés ────────────────────────────────────────────────
    st.subheader("Remplacements proposés")
    recommender = SubstitutionRecommender()
    subs = recommender.recommend(active_team, bench_team, max_subs=3)

    if not subs:
        st.success("Aucun remplacement nécessaire pour le moment.")
    else:
        for s in subs:
            color = ("#e94560" if s["urgency"] == "IMMEDIATE"
                     else "#ffa500" if s["urgency"] == "HAUTE"
                     else "#00c9a7")
            st.markdown(
                f"""
                <div style="border:1px solid {color}; border-radius:8px;
                            padding:12px; margin:6px 0; background:#161b22;">
                    <b>Urgence : {s['urgency']}</b><br>
                    <b>Sortant :</b> Joueur #{s['out_player_id']} ({s['out_position']}) —
                    Fatigue {s['out_fatigue']} | Performance {s['out_performance']} |
                    Risque {s['out_injury_risk']}<br>
                    <b>Entrant :</b> Joueur #{s['in_player_id']} ({s['in_position']}) —
                    Fraîcheur {100 - s['in_fatigue']:.0f}%<br>
                    <b>Score compatibilité :</b> {s['tactical_score']:.1f}/100<br>
                    <i>{s['reason']}</i>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Décisions enregistrées ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Enregistrer une décision")
    col_act1, col_act2, col_act3 = st.columns(3)

    with col_act1:
        if st.button("⚡ Mode offensif (4-3-3)", use_container_width=True):
            st.session_state.sim_actions.append(f"Min {minute}' — Tactique offensive (4-3-3)")
            st.success("Décision enregistrée")

    with col_act2:
        if st.button("🛡️ Mode défensif (5-3-2)", use_container_width=True):
            st.session_state.sim_actions.append(f"Min {minute}' — Tactique défensive (5-3-2)")
            st.success("Décision enregistrée")

    with col_act3:
        if subs:
            sel_sub = st.selectbox("Remplacement",
                                   [f"#{s['out_player_id']} → #{s['in_player_id']}" for s in subs])
            if st.button("Confirmer le remplacement", use_container_width=True):
                st.session_state.sim_actions.append(f"Min {minute}' — Remplacement {sel_sub}")
                st.success("Décision enregistrée")

    if st.session_state.sim_actions:
        with st.expander(f"Historique ({len(st.session_state.sim_actions)} décision(s))"):
            for a in reversed(st.session_state.sim_actions):
                st.markdown(f"- {a}")
            if st.button("Effacer l'historique"):
                st.session_state.sim_actions = []
                st.rerun()


# =============================================================================
# ONGLET 3 — APRÈS LE MATCH
# =============================================================================
with tab3:
    st.header("Rapport post-match")

    sim2  = TacticalSimulator()
    tac2  = sim2.recommend_formation(active_team, {"minute": 90, "score_diff": score_diff})
    opp2  = TacticalAnalyzer().analyze_opponent(opp_df)
    subs2 = SubstitutionRecommender().recommend(active_team, bench_team, max_subs=3)

    gen    = ReportGenerator()
    report = gen.generate(
        player_stats=active_team,
        substitutions=subs2,
        tactical_recommendation=tac2,
        opponent_analysis=opp2,
        rl_summary={
            "win_rate":          "N/A",
            "most_chosen_action": "Remplacement préventif",
        },
        match_context={
            "team":     "Équipe A" if team_id == 1 else "Équipe B",
            "opponent": "Équipe B" if team_id == 1 else "Équipe A",
            "score":    f"+{score_diff}" if score_diff >= 0 else str(score_diff),
        },
    )

    # ── Résumé ────────────────────────────────────────────────────────────────
    st.info(report["summary"])

    # ── Performances ──────────────────────────────────────────────────────────
    st.subheader("Performances")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Meilleures performances**")
        if report["top_performers"]:
            st.dataframe(pd.DataFrame(report["top_performers"]), use_container_width=True)

    with col2:
        st.markdown("**Sous-performances**")
        if report["underperformers"]:
            st.dataframe(pd.DataFrame(report["underperformers"]), use_container_width=True)
        else:
            st.success("Aucune sous-performance critique.")

    with col3:
        st.markdown("**Bilan fatigue**")
        fat = report["fatigue_report"]
        st.metric("Fatigue moyenne", f"{fat.get('avg_fatigue', 0):.1f}")
        st.metric("Fatigue max",     f"{fat.get('max_fatigue', 0):.1f}")
        st.metric("Joueurs > 70",    fat.get("players_above_70", 0))

    st.markdown("---")

    if PLOTLY:
        col_a, col_b = st.columns(2)
        with col_a:
            fig_perf = px.histogram(
                active_team, x="PerformanceScore", nbins=10,
                color_discrete_sequence=["#0f3460"],
                title="Distribution des performances",
            )
            fig_perf.add_vline(x=50, line_dash="dash", line_color="red")
            fig_perf.update_layout(
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                font_color="#e6edf3",
            )
            st.plotly_chart(fig_perf, use_container_width=True)

        with col_b:
            fig_fat = px.histogram(
                active_team, x="FatigueScore", nbins=10,
                color_discrete_sequence=["#e94560"],
                title="Distribution de la fatigue",
            )
            fig_fat.add_vline(x=70, line_dash="dash", line_color="orange")
            fig_fat.update_layout(
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                font_color="#e6edf3",
            )
            st.plotly_chart(fig_fat, use_container_width=True)

    st.markdown("---")

    # ── Décisions ─────────────────────────────────────────────────────────────
    st.subheader("Décisions de remplacement recommandées")
    decisions = report.get("substitution_decisions", [])
    if decisions:
        for d in decisions:
            st.markdown(
                f"- **{d['urgence']}** — {d['sortant']} → {d['entrant']}  \n"
                f"  *{d['explication']}*  \n"
                f"  Score compatibilité : {d['score_compatibilite']:.1f}/100"
            )
    else:
        st.success("Aucun remplacement recommandé.")

    if st.session_state.sim_actions:
        st.markdown("**Décisions enregistrées durant le match**")
        for a in st.session_state.sim_actions:
            st.markdown(f"- {a}")

    st.markdown("---")

    # ── Injury Risk ───────────────────────────────────────────────────────────
    st.subheader("Risque de blessure")
    inj  = report["injury_risk_report"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Risque Faible", inj.get("Faible", 0))
    c2.metric("Risque Moyen",  inj.get("Moyen",  0))
    c3.metric("Risque Élevé",  inj.get("Eleve",  0), delta_color="inverse")

    if inj.get("high_risk_players"):
        st.warning(f"Joueurs à haut risque : {inj['high_risk_players']}")

    st.markdown("---")

    # ── Erreurs tactiques ─────────────────────────────────────────────────────
    st.subheader("Erreurs tactiques identifiées")
    for err in report.get("tactical_errors", []):
        st.error(err)

    st.markdown("---")

    # ── Conclusion ────────────────────────────────────────────────────────────
    st.subheader("Conclusion")
    st.success(report.get("conclusion", ""))

    # ── Comparaison ML ────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("Comparaison ML — Injury Risk (Random Forest vs XGBoost)"):
        col1, col2 = st.columns(2)
        col1.metric("Random Forest", f"{inj_pred.results.get('rf_accuracy', 0)*100:.1f}%")
        col2.metric("XGBoost",       f"{inj_pred.results.get('xgb_accuracy', 0)*100:.1f}%")
        st.caption(f"Meilleur modèle : {inj_pred.results.get('best_model', 'N/A')}")

        feat_imp = inj_pred.results.get("feature_importance", {})
        if feat_imp and PLOTLY:
            fi_df = pd.DataFrame(
                list(feat_imp.items()), columns=["Variable", "Importance"]
            ).sort_values("Importance")
            fig_fi = px.bar(
                fi_df, x="Importance", y="Variable", orientation="h",
                color="Importance", color_continuous_scale="RdYlGn",
                title="Importance des variables — Random Forest",
            )
            fig_fi.update_layout(
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                font_color="#e6edf3",
            )
            st.plotly_chart(fig_fi, use_container_width=True)

    with st.expander("Comparaison Deep Learning — Performance (MLP vs LSTM)"):
        col1, col2 = st.columns(2)
        col1.metric("MLP MAE",  perf_pred.results.get("mlp_mae",  "N/A"))
        col2.metric("LSTM MAE", perf_pred.results.get("lstm_mae", "N/A"))
        st.info(f"Meilleur modèle : {perf_pred.results.get('best_model', 'N/A')}")

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.download_button(
        label="Télécharger le rapport (JSON)",
        data=json.dumps(report, ensure_ascii=False, indent=2, default=str),
        file_name="match_report.json",
        mime="application/json",
    )


# =============================================================================
# ONGLET 4 — VIDÉO ANNOTÉE
# =============================================================================
with tab_vid:
    if not selected_video:
        st.info("Sélectionnez une vidéo dans la barre latérale.")
    else:
        raw_video = find_output_video(selected_video)

        if raw_video is None:
            st.info("Aucune vidéo annotée trouvée. Lancez le tracking pour en générer une.")
        else:
            valid, n_frames = is_video_valid(raw_video)

            if not valid:
                st.error(
                    f"Le fichier `{raw_video.name}` est corrompu ou vide.  \n"
                    "Le tracking a probablement été interrompu trop tôt.  \n"
                    "**Relancez le tracking et laissez-le terminer complètement.**"
                )
            else:
                st.caption(f"Source : `{raw_video.name}` — {n_frames} frames")

                # Conversion vers MP4 H264 compatible navigateur
                with st.spinner("Conversion H264 pour le navigateur…"):
                    web_video = convert_to_web_mp4(raw_video)

                if web_video:
                    try:
                        st.video(web_video.read_bytes(), format="video/mp4")
                    except Exception as e:
                        st.warning(f"Lecture dans le navigateur échouée : {e}")
                else:
                    # Dernière tentative : afficher directement avec st.video
                    st.warning(
                        "Conversion H264 impossible (ffmpeg non installé et codec avc1 non disponible).  \n"
                        "La vidéo est disponible en téléchargement ci-dessous."
                    )

                # Bouton téléchargement toujours disponible
                video_to_dl = web_video or raw_video
                st.download_button(
                    label=f"Télécharger `{video_to_dl.name}`",
                    data=video_to_dl.read_bytes(),
                    file_name=video_to_dl.name,
                    mime="video/mp4" if video_to_dl.suffix == ".mp4" else "video/x-msvideo",
                )
