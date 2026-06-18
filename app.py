import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Football Manager",
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
            "ID":        f"P{p['player_id']}",
            "Équipe":    f"E{p['equipe']}" if p["equipe"] else "?",
            "Poste":     p["poste"] or "?",
            "Score":     p["score"],
            "Rang":      p["classement"],
            "Vit. moy (km/h)": p["vitesse_moyenne_kmh"],
            "Distance (m)":    p["distance_m"],
            "Passes tent.":    p["passes_tentees"],
            "Passes réuss.":   p["passes_reussies"],
            "Taux pass. (%)":  p["taux_reussite_passes"],
            "Possession (%)":  p["possession_pct"],
            "Sous-perf.":      p["sous_performance"],
            "_team":           p["equipe"],
        })
    df = pd.DataFrame(rows).sort_values("Rang")
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


# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.title("⚽ Football Manager")
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

run_btn  = st.sidebar.button(
    "▶ Lancer le tracking",
    disabled=selected_video is None or tracking,
    width="stretch",
    type="primary",
)
stop_btn = st.sidebar.button(
    "⏹ Arrêter",
    disabled=not tracking,
    width="stretch",
)

st.sidebar.markdown("---")
if selected_video:
    sp = stats_path_for(selected_video)
    if sp.exists():
        mtime = time.strftime("%d/%m/%Y %H:%M", time.localtime(sp.stat().st_mtime))
        st.sidebar.caption(f"Stats existantes ({mtime})")
    else:
        st.sidebar.caption("Aucune stats pour cette vidéo")

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

# ── Live progress (fragment = only this section reruns, not the whole page) ───

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
    st.title("Football Manager — Dashboard")
    st.info(
        "Sélectionnez une vidéo dans la barre latérale et lancez le tracking "
        "pour afficher les statistiques."
    )
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────

st.title(f"⚽ {Path(stats['video']).name}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Durée",            f"{stats['duree_secondes']}s")
col2.metric("Frames traitées",  stats["frames_traitees"])
col3.metric("Passes réussies",  stats["total_passes"])
col4.metric("Joueurs détectés", len(stats["joueurs"]))

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_players, tab_charts, tab_subst, tab_video = st.tabs(
    ["Joueurs", "Graphiques", "Substitutions", "Vidéo annotée"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Players table
# ═══════════════════════════════════════════════════════════════════════════════
with tab_players:
    df = players_df(stats)

    c1, c2, c3 = st.columns(3)
    team_filter = c1.multiselect(
        "Équipe", options=["E1", "E2"], default=["E1", "E2"]
    )
    poste_filter = c2.multiselect(
        "Poste",
        options=POSTE_ORDER + ["?"],
        default=POSTE_ORDER + ["?"],
    )
    show_underperf = c3.checkbox("Sous-performances uniquement", value=False)

    mask = df["Équipe"].isin(team_filter) & df["Poste"].isin(poste_filter)
    if show_underperf:
        mask &= df["Sous-perf."]
    filtered = df[mask]

    hidden_cols = [c for c in filtered.columns if c.startswith("_")]
    styled = (
        filtered
        .style
        .apply(color_row, axis=1)
        .format({
            "Score":           "{:.1f}",
            "Vit. moy (km/h)": "{:.2f}",
            "Distance (m)":    "{:.1f}",
            "Taux pass. (%)":  "{:.1f}",
            "Possession (%)":  "{:.1f}",
        })
        .hide(axis="index")
        .hide(hidden_cols + ["Sous-perf."], axis="columns")
    )
    st.dataframe(styled, width="stretch", height=520)

    st.caption(
        "🔵 Équipe 1 · 🔴 Équipe 2 · fond rouge = sous-performance détectée"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Charts
# ═══════════════════════════════════════════════════════════════════════════════
with tab_charts:
    df_full = players_df(stats)

    # ── Possession par équipe ──────────────────────────────────────────────────
    poss = stats.get("possession_equipes", {})
    if poss:
        st.subheader("Possession par équipe")
        poss_df = pd.DataFrame(
            {"Équipe": list(poss.keys()), "Possession (%)": list(poss.values())}
        )
        st.bar_chart(poss_df.set_index("Équipe"), color=["#3a86ff"])

    st.markdown("---")

    # ── Score, Vitesse, Distance ───────────────────────────────────────────────
    chart_metric = st.selectbox(
        "Métrique à afficher",
        ["Score", "Vit. moy (km/h)", "Distance (m)", "Possession (%)"],
    )

    chart_df = (
        df_full[["ID", "Équipe", chart_metric]]
        .sort_values(chart_metric, ascending=False)
        .assign(couleur=lambda d: d["Équipe"].map({"E1": "#3a86ff", "E2": "#ff006e"}))
        .set_index("ID")
    )

    st.bar_chart(chart_df[[chart_metric, "couleur"]], color="couleur")

    st.markdown("---")

    # ── Scatter: vitesse vs distance ──────────────────────────────────────────
    st.subheader("Vitesse moyenne vs Distance parcourue")
    scatter_df = df_full[["ID", "Équipe", "Vit. moy (km/h)", "Distance (m)", "Score", "Sous-perf."]].copy()
    scatter_df["Couleur"] = scatter_df["Équipe"].map({"E1": "#3a86ff", "E2": "#ff006e"})
    st.scatter_chart(
        scatter_df,
        x="Vit. moy (km/h)",
        y="Distance (m)",
        color="Couleur",
        size="Score",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Substitution recommendations
# ═══════════════════════════════════════════════════════════════════════════════
with tab_subst:
    subst = stats.get("substitutions_recommandees", {})
    if not subst:
        st.success("Aucune substitution recommandée (tous les joueurs qualifiés ou données insuffisantes).")
    else:
        st.warning(f"{len(subst)} substitution(s) recommandée(s)")
        for _, r in sorted(subst.items(), key=lambda x: x[1]["score"]):
            team_color = TEAM_COLORS.get(r["equipe"], "#888")
            with st.container(border=True):
                c1, c2 = st.columns([1, 3])
                c1.markdown(
                    f"<div style='font-size:2rem;font-weight:bold;color:{team_color};'>"
                    f"P{r['player_id']}</div>"
                    f"<div style='color:{team_color};'>E{r['equipe']} · {r['poste'] or '?'}</div>",
                    unsafe_allow_html=True,
                )
                c2.markdown(f"**Score : {r['score']:.1f} / 100**")
                for raison in r["raisons"]:
                    c2.markdown(f"- {raison}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Annotated video
# ═══════════════════════════════════════════════════════════════════════════════
with tab_video:
    out_video = output_video_for(selected_video)
    if out_video.exists():
        st.video(str(out_video))
    else:
        st.info("Aucune vidéo annotée trouvée. Lancez le tracking pour en générer une.")
