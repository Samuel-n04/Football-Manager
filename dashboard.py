"""
Football Decision Intelligence AI — Dashboard Coach
Lancement :  streamlit run dashboard.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

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

# ── Configuration de la page ──────────────────────────────────────────────────
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
    .alert-critique { background:#5a0000; border-left:4px solid #e94560; padding:8px; border-radius:4px; margin:4px 0; }
    .alert-haute    { background:#4a3000; border-left:4px solid #ffa500; padding:8px; border-radius:4px; margin:4px 0; }
    .alert-basse    { background:#003020; border-left:4px solid #00c9a7; padding:8px; border-radius:4px; margin:4px 0; }
</style>
""", unsafe_allow_html=True)

# ── Chargement des données ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = generate_player_dataset(n_players=22, n_matches=15)
    opp_df = generate_opponent_data(n_players=11)
    return df, opp_df

@st.cache_resource
def build_models(df):
    perf_calc = PerformanceScoreCalculator()
    fat_calc  = FatigueScoreCalculator()
    inj_calc  = InjuryRiskCalculator()

    df = perf_calc.calculate(df)
    df = fat_calc.calculate(df)
    df = inj_calc.calculate(df)
    df = perf_calc.classify(df)
    df = fat_calc.classify(df)

    inj_pred = InjuryPredictor()
    inj_pred.train(df)

    perf_pred = PerformancePredictor()
    perf_pred.train(df)

    return df, inj_pred, perf_pred

with st.spinner("Chargement des données et entraînement des modèles…"):
    raw_df, opp_df = load_data()
    full_df, inj_pred, perf_pred = build_models(raw_df)

# Données du match en cours (dernier match)
last_match = full_df[full_df['MatchID'] == full_df['MatchID'].max()].copy()
last_match['Team'] = (last_match['PlayerID'] <= 11).map({True: 1, False: 2})

bench = full_df[full_df['MatchID'] == full_df['MatchID'].max() - 1].copy()
bench['Team'] = (bench['PlayerID'] <= 11).map({True: 1, False: 2})

# ── Barre latérale ────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚽ Football AI Coach")
    st.markdown("---")
    team_choice = st.selectbox("Équipe analysée", ["Équipe A (1)", "Équipe B (2)"])
    minute = st.slider("Minute du match", 0, 90, 60, 5)
    score_diff = st.number_input("Différence de score (+ = on gagne)", -3, 3, 0, 1)
    st.markdown("---")
    st.markdown("**Légende alertes**")
    st.markdown("🔴 Critique &nbsp; 🟠 Haute &nbsp; 🟢 Basse")

team_id = 1 if "1" in team_choice else 2
active_team = last_match[last_match['Team'] == team_id].copy()
bench_team  = bench[bench['Team'] == team_id].copy()

# ── Onglets ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["⬅ Avant le match", "🎮 Pendant le match", "📊 Après le match"])


# =============================================================================
# ONGLET 1 — AVANT LE MATCH
# =============================================================================
with tab1:
    st.header("Analyse pré-match")
    col1, col2 = st.columns(2)

    # ── Tactique recommandée ─────────────────────────────────────────────────
    with col1:
        st.subheader("Tactique recommandée")
        sim = TacticalSimulator()
        tac = sim.recommend_formation(active_team, {'minute': 0, 'score_diff': 0})
        all_results = pd.DataFrame(tac['all_results'])

        st.metric("Formation conseillée", tac['recommended_formation'])
        st.caption(tac['description'])

        fig = px.bar(
            all_results, x='Formation', y='GlobalScore',
            color='GlobalScore', color_continuous_scale='RdYlGn',
            title='Score global par formation',
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Détails formations"):
            st.dataframe(
                all_results[['Formation', 'Possession', 'Pressing', 'xG', 'DefensiveRisk', 'GlobalScore']],
                use_container_width=True,
            )

    # ── Analyse adverse ──────────────────────────────────────────────────────
    with col2:
        st.subheader("Analyse adverse")
        analyzer = TacticalAnalyzer()
        opp = analyzer.analyze_opponent(opp_df)

        st.info(opp['summary'])

        # Joueur clé adverse (le plus performant = le plus dangereux)
        st.markdown("**Joueur clé adverse à surveiller**")
        top_opp = opp_df.nlargest(1, 'PerformanceScore').iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Joueur #", int(top_opp['PlayerID']))
        c2.metric("Poste", top_opp['Position'])
        c3.metric("Performance", f"{top_opp['PerformanceScore']:.1f}/100")

        # Zones vulnérables adverses
        weak_zones = opp['weak_zones']['zones']
        fig_zones = go.Figure(go.Bar(
            x=list(weak_zones.keys()),
            y=list(weak_zones.values()),
            marker_color=[
                '#e94560' if z == opp['weak_zones']['weakest_zone'] else '#0f3460'
                for z in weak_zones
            ],
        ))
        fig_zones.update_layout(title="Zones vulnérables adverses (%)", yaxis_title="%")
        st.plotly_chart(fig_zones, use_container_width=True)

        st.markdown("**Recommandations tactiques**")
        for rec in opp['tactical_recommendations']:
            st.markdown(f"- {rec}")

    st.markdown("---")

    # ── Risques pré-match ────────────────────────────────────────────────────
    st.subheader("Risques pré-match")
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown("**Joueurs adverses en sous-performance**")
        if opp['weak_players']:
            st.dataframe(pd.DataFrame(opp['weak_players']), use_container_width=True)
        else:
            st.success("Aucun joueur adverse en sous-performance.")

    with col_r2:
        st.markdown("**Prédiction de performance (prochain match)**")
        preds = []
        for pid in active_team['PlayerID'].unique():
            hist = full_df[full_df['PlayerID'] == pid].sort_values('MatchID')
            if len(hist) >= 5:
                pred = perf_pred.predict_next(hist)
                preds.append({'Joueur #': pid, 'Performance prédite': pred})
        if preds:
            pred_df = pd.DataFrame(preds)
            fig_pred = px.bar(
                pred_df, x='Joueur #', y='Performance prédite',
                color='Performance prédite', color_continuous_scale='RdYlGn',
                range_color=[0, 100], title='Performance prédite',
            )
            st.plotly_chart(fig_pred, use_container_width=True)


# =============================================================================
# ONGLET 2 — PENDANT LE MATCH
# =============================================================================
with tab2:
    st.header(f"Suivi en temps réel — Minute {minute}")

    # ── KPIs ─────────────────────────────────────────────────────────────────
    avg_fat  = active_team['FatigueScore'].mean()
    avg_perf = active_team['PerformanceScore'].mean()
    n_inj    = (active_team['InjuryRisk'] == 'Eleve').sum()

    rules = RulesEngine()
    rule_results = rules.evaluate(active_team)
    n_subs = int((rule_results['TopAction'] == 'SUBSTITUTION').sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fatigue moyenne",          f"{avg_fat:.1f} / 100",  delta_color="inverse")
    c2.metric("Performance moyenne",      f"{avg_perf:.1f} / 100")
    c3.metric("Joueurs risque blessure élevé", int(n_inj),         delta_color="inverse")
    c4.metric("Remplacements conseillés", int(n_subs),             delta_color="inverse")

    st.markdown("---")
    col1, col2 = st.columns([1.4, 1])

    # ── Graphique Fatigue / Performance ─────────────────────────────────────
    with col1:
        st.subheader("Fatigue et Performance par joueur")
        fig = px.scatter(
            active_team,
            x='FatigueScore', y='PerformanceScore',
            color='InjuryRisk', size='Distance',
            hover_data=['PlayerID', 'Position', 'SprintCount'],
            color_discrete_map={'Faible': '#00c9a7', 'Moyen': '#ffa500', 'Eleve': '#e94560'},
            title='Vue joueurs (taille = distance parcourue)',
        )
        fig.add_vline(x=70, line_dash='dash', line_color='orange',
                      annotation_text='Seuil fatigue')
        fig.add_hline(y=50, line_dash='dash', line_color='red',
                      annotation_text='Seuil performance')
        st.plotly_chart(fig, use_container_width=True)

    # ── Alertes actives ──────────────────────────────────────────────────────
    with col2:
        st.subheader("Alertes actives")
        alerts = rule_results[rule_results['AlertCount'] > 0].sort_values('AlertCount', ascending=False)
        if alerts.empty:
            st.success("Aucune alerte active.")
        else:
            for _, row in alerts.iterrows():
                for alert in row['Alerts']:
                    sev = alert['severity']
                    css  = 'alert-critique' if sev == 'CRITIQUE' else ('alert-haute' if sev == 'HAUTE' else 'alert-basse')
                    icon = '🔴' if sev == 'CRITIQUE' else ('🟠' if sev == 'HAUTE' else '🟢')
                    st.markdown(
                        f'<div class="{css}">{icon} <b>Joueur #{int(row["PlayerID"])}</b> — '
                        f'{alert["label"]}<br>'
                        f'Fatigue : {row["FatigueScore"]:.0f} | Performance : {row["PerformanceScore"]:.0f}</div>',
                        unsafe_allow_html=True,
                    )

    st.markdown("---")

    # ── Remplacements proposés ───────────────────────────────────────────────
    st.subheader("Remplacements proposés")
    recommender = SubstitutionRecommender()
    subs = recommender.recommend(active_team, bench_team, max_subs=3)

    if not subs:
        st.success("Aucun remplacement nécessaire pour le moment.")
    else:
        for s in subs:
            color = '#e94560' if s['urgency'] == 'IMMEDIATE' else ('#ffa500' if s['urgency'] == 'HAUTE' else '#00c9a7')
            st.markdown(
                f"""
                <div style="border:1px solid {color}; border-radius:8px; padding:12px; margin:6px 0">
                    <b>Urgence : {s['urgency']}</b><br>
                    <b>Sortant :</b> Joueur #{s['out_player_id']} ({s['out_position']}) —
                    Fatigue {s['out_fatigue']} | Performance {s['out_performance']} | Risque {s['out_injury_risk']}<br>
                    <b>Entrant :</b> Joueur #{s['in_player_id']} ({s['in_position']}) —
                    Fraîcheur {100 - s['in_fatigue']:.0f}%<br>
                    <b>Score compatibilité :</b> {s['tactical_score']:.1f}/100<br>
                    <i>{s['reason']}</i>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# ONGLET 3 — APRÈS LE MATCH
# =============================================================================
with tab3:
    st.header("Rapport post-match")

    sim2 = TacticalSimulator()
    tac2 = sim2.recommend_formation(active_team, {'minute': 90, 'score_diff': score_diff})
    opp2 = TacticalAnalyzer().analyze_opponent(opp_df)
    subs2 = SubstitutionRecommender().recommend(active_team, bench_team, max_subs=3)

    gen = ReportGenerator()
    report = gen.generate(
        player_stats=active_team,
        substitutions=subs2,
        tactical_recommendation=tac2,
        opponent_analysis=opp2,
        rl_summary={},
        match_context={
            'team': 'Équipe A',
            'opponent': 'Équipe B',
            'score': f"+{score_diff}" if score_diff >= 0 else str(score_diff),
        },
    )

    # ── Résumé ───────────────────────────────────────────────────────────────
    st.info(report['summary'])

    # ── Performances ─────────────────────────────────────────────────────────
    st.subheader("Performances")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Meilleures performances**")
        if report['top_performers']:
            st.dataframe(pd.DataFrame(report['top_performers']), use_container_width=True)

    with col2:
        st.markdown("**Sous-performances**")
        if report['underperformers']:
            st.dataframe(pd.DataFrame(report['underperformers']), use_container_width=True)
        else:
            st.success("Aucune sous-performance critique.")

    with col3:
        st.markdown("**Bilan fatigue**")
        fat = report['fatigue_report']
        st.metric("Fatigue moyenne", f"{fat.get('avg_fatigue', 0):.1f}")
        st.metric("Fatigue max",     f"{fat.get('max_fatigue', 0):.1f}")
        st.metric("Joueurs > 70",    fat.get('players_above_70', 0))

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        fig_perf = px.histogram(
            active_team, x='PerformanceScore', nbins=10,
            color_discrete_sequence=['#0f3460'],
            title='Distribution des performances',
        )
        fig_perf.add_vline(x=50, line_dash='dash', line_color='red')
        st.plotly_chart(fig_perf, use_container_width=True)

    with col_b:
        fig_fat = px.histogram(
            active_team, x='FatigueScore', nbins=10,
            color_discrete_sequence=['#e94560'],
            title='Distribution de la fatigue',
        )
        fig_fat.add_vline(x=70, line_dash='dash', line_color='orange')
        st.plotly_chart(fig_fat, use_container_width=True)

    st.markdown("---")

    # ── Décisions ─────────────────────────────────────────────────────────────
    st.subheader("Décisions de remplacement recommandées")
    decisions = report.get('substitution_decisions', [])
    if decisions:
        for d in decisions:
            st.markdown(
                f"- **{d['urgence']}** — {d['sortant']} → {d['entrant']}  \n"
                f"  *{d['explication']}*  \n"
                f"  Score compatibilité : {d['score_compatibilite']:.1f}/100"
            )
    else:
        st.success("Aucun remplacement recommandé.")

    st.markdown("---")

    # ── Injury Risk ──────────────────────────────────────────────────────────
    st.subheader("Risque de blessure")
    inj = report['injury_risk_report']
    c1, c2, c3 = st.columns(3)
    c1.metric("Risque Faible", inj.get('Faible', 0))
    c2.metric("Risque Moyen",  inj.get('Moyen',  0))
    c3.metric("Risque Élevé",  inj.get('Eleve',  0), delta_color="inverse")

    if inj.get('high_risk_players'):
        st.warning(f"Joueurs à haut risque : {inj['high_risk_players']}")

    st.markdown("---")

    # ── Erreurs tactiques ────────────────────────────────────────────────────
    st.subheader("Erreurs tactiques identifiées")
    for err in report.get('tactical_errors', []):
        st.error(err)

    st.markdown("---")

    # ── Conclusion ───────────────────────────────────────────────────────────
    st.subheader("Conclusion")
    st.success(report.get('conclusion', ''))

    # ── Comparaison ML ───────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("Comparaison ML — Injury Risk (Random Forest vs XGBoost)"):
        col1, col2 = st.columns(2)
        col1.metric("Random Forest", f"{inj_pred.results.get('rf_accuracy', 0)*100:.1f}%")
        col2.metric("XGBoost",       f"{inj_pred.results.get('xgb_accuracy', 0)*100:.1f}%")
        st.caption(f"Meilleur modèle : {inj_pred.results.get('best_model', 'N/A')}")

        feat_imp = inj_pred.results.get('feature_importance', {})
        if feat_imp:
            fi_df = pd.DataFrame(list(feat_imp.items()), columns=['Variable', 'Importance'])
            fi_df = fi_df.sort_values('Importance')
            fig_fi = px.bar(fi_df, x='Importance', y='Variable', orientation='h',
                            color='Importance', color_continuous_scale='RdYlGn',
                            title='Importance des variables — Random Forest')
            st.plotly_chart(fig_fi, use_container_width=True)

    with st.expander("Comparaison Deep Learning — Performance (MLP vs LSTM)"):
        col1, col2 = st.columns(2)
        col1.metric("MLP MAE",  perf_pred.results.get('mlp_mae',  'N/A'))
        col2.metric("LSTM MAE", perf_pred.results.get('lstm_mae', 'N/A'))
        st.info(f"Meilleur modèle : {perf_pred.results.get('best_model', 'N/A')}")

    # ── Export ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.download_button(
        label="Télécharger le rapport (JSON)",
        data=json.dumps(report, ensure_ascii=False, indent=2, default=str),
        file_name="match_report.json",
        mime="application/json",
    )
