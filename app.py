import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import altair as alt
import cv2
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Football Manager",
    page_icon="⚽",
    layout="wide",
)

INPUT_DIR  = Path("input_videos")
OUTPUT_DIR = Path("output_videos")
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

TEAM_COLORS = {1: "#3a86ff", 2: "#ff006e"}
POSTE_ORDER = ["Gardien", "Défenseur", "Milieu", "Attaquant"]
RISK_COLORS = {"Faible": "#2ecc71", "Moyen": "#f39c12", "Élevé": "#e74c3c"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def stats_path_for(video_path: Path) -> Path:
    return OUTPUT_DIR / (video_path.stem + "_stats.json")


def output_video_for(video_path: Path) -> Path:
    return OUTPUT_DIR / (video_path.stem + "_tracking.mp4")


def find_output_video(p: Path) -> Path | None:
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
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return False, 0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frames > 0, frames


def convert_to_web_mp4(src: Path) -> Path | None:
    dst = OUTPUT_DIR / (src.stem + "_web.mp4")
    if dst.exists() and dst.stat().st_size > 10_000:
        return dst
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


def players_df(stats: dict) -> pd.DataFrame:
    rows = []
    for p in stats["joueurs"].values():
        rows.append({
            "ID":              f"P{p['player_id']}",
            "Équipe":          f"E{p['equipe']}" if p["equipe"] else "?",
            "Poste":           p["poste"] or "?",
            "Score":           p["score"],
            "Rang":            p["classement"],
            "Fatigue":         p.get("fatigue_score", 0.0),
            "Risque":          p.get("injury_risk_label", "Faible"),
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
                    pct_val  = float(line.split("(")[1].split("%")[0]) / 100
                    pct_text = line.strip()
                    break
                except Exception:
                    pass
        st.info(f"Tracking en cours — **{vid_name}**")
        st.progress(pct_val, text=pct_text)
        preview_path = OUTPUT_DIR / f"{Path(vid_name).stem}_preview.jpg"
        if preview_path.exists():
            st.image(str(preview_path), caption="Aperçu en direct", use_container_width=True)
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

tab_players, tab_charts, tab_subst, tab_adverse, tab_rapport, tab_video = st.tabs(
    ["Joueurs", "Graphiques", "Substitutions", "Analyse adverse", "Rapport", "Vidéo annotée"]
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

    if df.empty:
        filtered = df
    else:
        mask = df["Équipe"].isin(team_filter) & df["Poste"].isin(poste_filter)
        if show_underperf:
            mask &= df["Sous-perf."]
        filtered = df[mask]

    cols_to_hide = [c for c in filtered.columns if c.startswith("_") or c == "Sous-perf."]
    styled = (
        filtered
        .style
        .apply(color_row, axis=1)
        .format({k: v for k, v in {
            "Score":           "{:.1f}",
            "Fatigue":         "{:.0f}",
            "Vit. moy (km/h)": "{:.2f}",
            "Distance (m)":    "{:.1f}",
            "Taux pass. (%)":  "{:.1f}",
            "Possession (%)":  "{:.1f}",
        }.items() if k in filtered.columns})
        .hide(axis="index")
    )
    if cols_to_hide:
        styled = styled.hide(cols_to_hide, axis="columns")
    st.dataframe(styled, width="stretch", height=520)

    st.caption(
        "🔵 Équipe 1 · 🔴 Équipe 2 · fond rouge = sous-performance · "
        "Fatigue : 0-100 (drop de vitesse 1ère→2ème mi-temps) · Risque : Faible / Moyen / Élevé"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Charts
# ═══════════════════════════════════════════════════════════════════════════════
with tab_charts:
    df_full = players_df(stats)

    TEAM_DOMAIN  = ["E1", "E2", "?"]
    TEAM_RANGE   = ["#3a86ff", "#ff006e", "#888888"]
    team_scale   = alt.Scale(domain=TEAM_DOMAIN, range=TEAM_RANGE)

    if df_full.empty:
        st.info("Aucune donnée joueur disponible.")
    else:
        # ── Possession par équipe ──────────────────────────────────────────────
        poss = stats.get("possession_equipes", {})
        if poss:
            st.subheader("Possession par équipe")
            poss_df = pd.DataFrame({
                "Équipe":         [k.replace("equipe_", "Équipe ") for k in poss],
                "Possession (%)": list(poss.values()),
            })
            poss_chart = (
                alt.Chart(poss_df)
                .mark_bar()
                .encode(
                    x=alt.X("Équipe:N", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Possession (%):Q", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color(
                        "Équipe:N",
                        scale=alt.Scale(
                            domain=["Équipe 1", "Équipe 2"],
                            range=["#3a86ff", "#ff006e"],
                        ),
                        legend=None,
                    ),
                    tooltip=["Équipe:N", alt.Tooltip("Possession (%):Q", format=".1f")],
                )
                .properties(height=300)
            )
            st.altair_chart(poss_chart, use_container_width=True)

        st.markdown("---")

        # ── Métrique par joueur ────────────────────────────────────────────────
        chart_metric = st.selectbox(
            "Métrique à afficher",
            ["Score", "Fatigue", "Vit. moy (km/h)", "Distance (m)", "Possession (%)"],
        )
        metric_df = df_full[["ID", "Équipe", chart_metric]].sort_values(chart_metric, ascending=False)
        metric_chart = (
            alt.Chart(metric_df)
            .mark_bar()
            .encode(
                x=alt.X("ID:N", sort=None, axis=alt.Axis(labelAngle=-45)),
                y=alt.Y(f"{chart_metric}:Q"),
                color=alt.Color("Équipe:N", scale=team_scale),
                tooltip=["ID:N", "Équipe:N", alt.Tooltip(f"{chart_metric}:Q", format=".1f")],
            )
            .properties(height=350)
        )
        st.altair_chart(metric_chart, use_container_width=True)

        st.markdown("---")

        # ── Scatter : vitesse vs distance ──────────────────────────────────────
        st.subheader("Vitesse moyenne vs Distance parcourue")
        scatter_df = (
            df_full[["ID", "Équipe", "Vit. moy (km/h)", "Distance (m)", "Score"]]
            .copy()
            .rename(columns={"Vit. moy (km/h)": "Vitesse", "Distance (m)": "Distance"})
        )
        scatter_chart = (
            alt.Chart(scatter_df)
            .mark_circle()
            .encode(
                x=alt.X("Vitesse:Q",  title="Vitesse moyenne (km/h)"),
                y=alt.Y("Distance:Q", title="Distance parcourue (m)"),
                color=alt.Color("Équipe:N", scale=team_scale),
                size=alt.Size("Score:Q", scale=alt.Scale(range=[50, 400])),
                tooltip=[
                    "ID:N", "Équipe:N",
                    alt.Tooltip("Vitesse:Q",  format=".1f", title="Vitesse (km/h)"),
                    alt.Tooltip("Distance:Q", format=".0f", title="Distance (m)"),
                    alt.Tooltip("Score:Q",    format=".1f"),
                ],
            )
            .properties(height=350)
            .interactive()
        )
        st.altair_chart(scatter_chart, use_container_width=True)

        st.markdown("---")

        # ── Fatigue par joueur ─────────────────────────────────────────────────
        st.subheader("Fatigue par joueur")
        fat_df = df_full[["ID", "Équipe", "Fatigue", "Risque"]].sort_values("Fatigue", ascending=False)
        risk_scale = alt.Scale(
            domain=["Faible", "Moyen", "Élevé"],
            range=["#2ecc71", "#f39c12", "#e74c3c"],
        )
        fat_chart = (
            alt.Chart(fat_df)
            .mark_bar()
            .encode(
                x=alt.X("ID:N", sort=None, axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("Fatigue:Q", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("Risque:N", scale=risk_scale),
                tooltip=["ID:N", "Équipe:N",
                         alt.Tooltip("Fatigue:Q", format=".0f"),
                         "Risque:N"],
            )
            .properties(height=350)
        )
        st.altair_chart(fat_chart, use_container_width=True)

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
            risk_color = RISK_COLORS.get(r.get("interpretation", "").split("Risque blessure : ")[-1].strip(), "#888")
            with st.container(border=True):
                interp = r.get("interpretation", "")
                if interp:
                    lines = interp.split("\n")
                    st.markdown(
                        f"<span style='font-size:1.1rem;font-weight:bold;color:{team_color};'>"
                        f"{lines[0]}</span>",
                        unsafe_allow_html=True,
                    )
                    for line in lines[1:]:
                        if "Risque blessure" in line:
                            level = line.split(": ")[-1].strip()
                            rc    = RISK_COLORS.get(level, "#888")
                            st.markdown(
                                f"<span style='color:{rc};font-weight:bold;'>{line}</span>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(line)
                else:
                    st.markdown(f"**P{r['player_id']}** — Score : {r['score']:.1f}/100")
                    for raison in r["raisons"]:
                        st.markdown(f"- {raison}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Analyse adverse
# ═══════════════════════════════════════════════════════════════════════════════
with tab_adverse:
    analyse_adv = stats.get("analyse_adverse", {})
    if not analyse_adv:
        st.info("Analyse adverse non disponible (données insuffisantes ou une seule équipe détectée).")
    else:
        st.caption(
            "Le système détecte deux équipes (E1 / E2) par couleur de maillot. "
            "Sélectionnez votre équipe pour afficher uniquement le joueur adverse à cibler."
        )
        mon_equipe = st.selectbox(
            "Votre équipe",
            options=["— Afficher les deux —", "E1", "E2"],
            index=0,
        )

        for team_key, adv in sorted(analyse_adv.items()):
            eq_num = team_key.replace("equipe_", "")
            # equipe_1 = conseil pour l'équipe 1 (joueur cible dans l'équipe 2)
            if mon_equipe != "— Afficher les deux —" and mon_equipe != f"E{eq_num}":
                continue
            tc = TEAM_COLORS.get(int(eq_num), "#888")
            with st.container(border=True):
                st.markdown(
                    f"<span style='color:{tc};font-size:1.1rem;font-weight:bold;'>"
                    f"Conseil pour Équipe {eq_num}</span>",
                    unsafe_allow_html=True,
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Joueur adverse à cibler", adv["joueur_cible"])
                c2.metric("Poste",                   adv["poste"])
                c3.metric("Score adverse",            f"{adv['score']:.0f}/100")
                c4.metric("Zone",                     adv["zone"].capitalize())
                st.info(adv["conseil"])
                st.caption(
                    f"Vitesse : {adv['vitesse_kmh']:.1f} km/h · "
                    f"Taux passes : {adv['taux_passes_pct']:.0f}%"
                )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Rapport automatique
# ═══════════════════════════════════════════════════════════════════════════════
with tab_rapport:
    rapport = stats.get("rapport_automatique", {})

    if not rapport:
        st.info(
            "Rapport non disponible. Ce fichier de stats a été généré avec une ancienne version — "
            "relancez le tracking pour obtenir un rapport complet."
        )
    else:
        # KPI
        best  = rapport.get("meilleur_joueur")
        worst = rapport.get("pire_joueur")
        dom   = rapport.get("equipe_dominante")

        c1, c2, c3 = st.columns(3)
        if best:
            c1.metric(
                "Meilleur joueur",
                f"{best['id']} ({best['poste']})",
                f"Score {best['score']:.0f}/100",
            )
        if worst:
            c2.metric(
                "Joueur en difficulté",
                f"{worst['id']} ({worst['poste']})",
                f"Score {worst['score']:.0f}/100",
                delta_color="inverse",
            )
        if dom:
            poss_pct = rapport.get("possession", {}).get(dom, 0)
            c3.metric(
                "Équipe dominante",
                dom.replace("equipe_", "Équipe "),
                f"{poss_pct}% possession",
            )

        # Substitutions
        subst_items = rapport.get("substitutions", [])
        if subst_items:
            st.markdown("---")
            st.subheader(f"Substitutions recommandées ({rapport.get('nb_substitutions', 0)})")
            for s in subst_items:
                team_num = int(s["equipe"].replace("E", "")) if s["equipe"].startswith("E") else None
                tc = TEAM_COLORS.get(team_num, "#888")
                with st.container(border=True):
                    lines = s["interpretation"].split("\n")
                    st.markdown(
                        f"<span style='color:{tc};font-weight:bold;'>{lines[0]}</span>",
                        unsafe_allow_html=True,
                    )
                    for line in lines[1:]:
                        if "Risque blessure" in line:
                            level = line.split(": ")[-1].strip()
                            rc    = RISK_COLORS.get(level, "#888")
                            st.markdown(
                                f"<span style='color:{rc};font-weight:bold;'>{line}</span>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(line)

        # Analyse adverse (recap)
        adv_data = rapport.get("analyse_adverse", {})
        if adv_data:
            st.markdown("---")
            st.subheader("Analyse adverse")
            for team_key, adv in sorted(adv_data.items()):
                eq_num = team_key.replace("equipe_", "")
                tc     = TEAM_COLORS.get(int(eq_num), "#888")
                st.markdown(
                    f"<span style='color:{tc};font-weight:bold;'>Équipe {eq_num}</span> — {adv['conseil']}",
                    unsafe_allow_html=True,
                )

        # Export
        st.markdown("---")
        st.subheader("Exporter le rapport")
        stem = Path(stats["video"]).stem
        ca, cb = st.columns(2)
        ca.download_button(
            "⬇ Télécharger JSON",
            data=json.dumps(rapport, indent=2, ensure_ascii=False),
            file_name=f"{stem}_rapport.json",
            mime="application/json",
        )
        cb.download_button(
            "⬇ Télécharger TXT",
            data=rapport.get("texte", ""),
            file_name=f"{stem}_rapport.txt",
            mime="text/plain",
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Annotated video
# ═══════════════════════════════════════════════════════════════════════════════
with tab_video:
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

            with st.spinner("Conversion H264 pour le navigateur…"):
                web_video = convert_to_web_mp4(raw_video)

            if web_video:
                try:
                    st.video(web_video.read_bytes(), format="video/mp4")
                except Exception as e:
                    st.warning(f"Lecture dans le navigateur échouée : {e}")
            else:
                st.warning(
                    "Conversion H264 impossible (ffmpeg non installé et codec avc1 non disponible).  \n"
                    "La vidéo est disponible en téléchargement ci-dessous."
                )

            video_to_dl = web_video or raw_video
            st.download_button(
                label=f"Télécharger `{video_to_dl.name}`",
                data=video_to_dl.read_bytes(),
                file_name=video_to_dl.name,
                mime="video/mp4" if video_to_dl.suffix == ".mp4" else "video/x-msvideo",
            )
