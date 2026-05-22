import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Rétention Client — Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# STYLES GLOBAUX
# ─────────────────────────────────────────────
st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stSidebar"], .stMarkdown,
    p, label, h1, h2, h3, h4, div, span {
        font-family: 'Inter', 'Segoe UI', 'Helvetica Neue', sans-serif !important;
        color: #1A1A2E !important;
    }
    .stApp { background-color: #F4F6F9 !important; }
    [data-testid="stSidebar"] { background-color: #1A1A2E !important; }
    [data-testid="stHeader"]  { background-color: #F4F6F9 !important; }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div { color: #FFFFFF !important; }

    .kpi-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 18px 14px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-top: 4px solid #4361EE;
        margin-bottom: 12px;
    }
    .kpi-card.red    { border-top-color: #E63946; }
    .kpi-card.orange { border-top-color: #F4A261; }
    .kpi-card.green  { border-top-color: #2A9D8F; }
    .kpi-card.blue   { border-top-color: #4361EE; }
    .kpi-card.purple { border-top-color: #7209B7; }

    .kpi-label {
        font-size: 10px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.8px;
        color: #8A8FA8 !important; margin-bottom: 6px;
    }
    .kpi-value { font-size: 26px; font-weight: 800; color: #1A1A2E !important; line-height: 1.1; }
    .kpi-sub   { font-size: 11px; color: #8A8FA8 !important; margin-top: 3px; }

    .section-title {
        font-size: 15px; font-weight: 700; color: #1A1A2E !important;
        border-left: 4px solid #4361EE;
        padding-left: 10px; margin: 18px 0 12px 0;
    }
    [data-testid="stDataFrame"] div { background-color: #FFFFFF !important; color: #1A1A2E !important; }
    hr { border-color: #E8E8F0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PALETTE PLOTLY
# ─────────────────────────────────────────────
COLORS = {
    "blue": "#4361EE", "red": "#E63946", "orange": "#F4A261",
    "green": "#2A9D8F", "purple": "#7209B7", "gray": "#8A8FA8",
    "risk": {"Faible": "#2A9D8F", "Modéré": "#F4A261", "Critique": "#E63946"},
}
PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Segoe UI, sans-serif", color="#1A1A2E", size=12),
    margin=dict(l=10, r=10, t=40, b=10),
    hoverlabel=dict(bgcolor="#1A1A2E", font_color="#FFFFFF", font_size=12),
)
def ax(): return dict(showgrid=True, gridcolor="#F0F0F5", showline=True,
                    linecolor="#E8E8F0", tickfont=dict(color="#8A8FA8", size=11), zeroline=False)

# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    d = "models" if os.path.exists("models") else "../models"
    return (joblib.load(os.path.join(d, "preprocessor.joblib")),
            joblib.load(os.path.join(d, "XGBoost.joblib")))

@st.cache_data
def load_data():
    p = ("data/raw/customer_churn_business_dataset.csv"
        if os.path.exists("data/raw/customer_churn_business_dataset.csv")
        else "../data/raw/customer_churn_business_dataset.csv")
    df = pd.read_csv(p)
    df["engagement_score"] = (
        (df["monthly_logins"] / 30 * 40) +
        (df["weekly_active_days"] / 7 * 40) +
        (df["csat_score"].fillna(3) / 5 * 20)
    ).clip(0, 100)
    return df

@st.cache_data
def predict(_pre, _mdl, df):
    X = df.drop(columns=["churn", "customer_id"], errors="ignore")
    p = _mdl.predict_proba(_pre.transform(X))[:, 1]
    out = df.copy()
    out["proba_churn"]           = p
    out["expected_loss_mensuel"] = out["monthly_fee"]  * p
    out["expected_loss_total"]   = out["total_revenue"] * p
    out["risk_level"] = np.select(
        [p < 0.4, p < 0.75], ["Faible", "Modéré"], default="Critique"
    )
    return out

try:
    preprocessor, model = load_pipeline()
    df_raw = load_data()
    df     = predict(preprocessor, model, df_raw)
except Exception as e:
    st.error(f"❌ Erreur : {e}"); st.stop()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style='text-align:center;padding:20px 0 10px'>
            <div style='font-size:26px'>📊</div>
            <div style='font-size:14px;font-weight:700;color:#FFF;margin-top:6px'>Rétention Client</div>
            <div style='font-size:10px;color:#8A9BB5;margin-top:2px'>Plateforme Décisionnelle</div>
        </div>
        <hr style='border-color:#2D3561;margin:10px 0 20px'>
    """, unsafe_allow_html=True)
    page = st.radio("Navigation", ["  Vue Exécutive", " Analyse Churn",
                            " Impact Financier", " Simulateur CRM"],
                label_visibility="collapsed")
    st.markdown("<hr style='border-color:#2D3561;margin:20px 0 10px'>"
                "<div style='font-size:10px;color:#4A5568;text-align:center'>XGBoost · 10 000 clients</div>",
                unsafe_allow_html=True)

def kpi(label, value, sub="", color="blue"):
    return (f'<div class="kpi-card {color}">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div>')

# ═════════════════════════════════════════════
# PAGE 1 — VUE EXÉCUTIVE
# ═════════════════════════════════════════════
if "Exécutive" in page:
    st.markdown("## 📈 Vue Exécutive — Santé du Portefeuille")
    st.markdown("<p style='color:#8A8FA8;margin-top:-10px'>Vision consolidée des 10 000 clients en temps réel.</p>", unsafe_allow_html=True)
    st.markdown("---")

    nb   = len(df)
    nrisk = len(df[df["proba_churn"] > 0.5])
    ncrit = len(df[df["risk_level"] == "Critique"])
    taux  = nrisk / nb
    mrr_e = df[df["proba_churn"] >= 0.4]["monthly_fee"].sum()
    mrr_t = df["monthly_fee"].sum()
    loss  = df["expected_loss_total"].sum()
    eng   = df["engagement_score"].mean()

    # Ligne 1 — KPIs principaux (déjà présents, légèrement enrichis)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.markdown(kpi("Clients à Risque",    f"{nrisk:,}",     f"{taux*100:.1f}% du portefeuille","red"),    unsafe_allow_html=True)
    c2.markdown(kpi("Comptes Critiques",   f"{ncrit:,}",     "Proba > 75%","red"),                          unsafe_allow_html=True)
    c3.markdown(kpi("MRR Exposé",          f"{mrr_e:,.0f}€", f"/ {mrr_t:,.0f}€ total","orange"),           unsafe_allow_html=True)
    c4.markdown(kpi("Expected Loss Total", f"{loss:,.0f}€",  "Σ(rev × proba)","purple"),                   unsafe_allow_html=True)
    c5.markdown(kpi("Engagement Moyen",    f"{eng:.1f}/100", "Score comportemental","green" if eng>60 else "red"), unsafe_allow_html=True)

    # Ligne 2 — Nouveaux KPIs
    nps_moy       = df["nps_score"].mean()
    taux_retention = (1 - taux) * 100
    tenure_crit   = df[df["risk_level"]=="Critique"]["tenure_months"].mean()
    tenure_secu   = df[df["risk_level"]=="Faible"]["tenure_months"].mean()
    rev_recuperable = df[df["risk_level"]=="Critique"]["expected_loss_mensuel"].sum() * 0.20

    st.markdown("")
    k1,k2,k3,k4 = st.columns(4)
    k1.markdown(kpi("Taux de Rétention",
                    f"{taux_retention:.1f}%",
                    "clients hors risque >50%",
                    "green" if taux_retention > 85 else "orange"),
                unsafe_allow_html=True)
    k2.markdown(kpi("NPS Moyen Portefeuille",
                    f"{nps_moy:.0f}",
                    "Net Promoter Score",
                    "green" if nps_moy > 30 else ("orange" if nps_moy > 0 else "red")),
                unsafe_allow_html=True)
    k3.markdown(kpi("Ancienneté Moy. Critiques",
                    f"{tenure_crit:.0f} mois",
                    f"vs {tenure_secu:.0f} mois (sécurisés)",
                    "orange"),
                unsafe_allow_html=True)
    k4.markdown(kpi("Revenu Récupérable (20%)",
                    f"{rev_recuperable:,.0f}€",
                    "si 20% des critiques retenus",
                    "green"),
                unsafe_allow_html=True)


    st.markdown("")
    # ── Ligne 1 : Risque + Répartition
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Risque Global</div>", unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=taux*100,
            number={"suffix":"%","font":{"size":34}},
            delta={"reference":10,"increasing":{"color":"#E63946"},
                "decreasing":{"color":"#2A9D8F"},"suffix":" pts vs seuil cible"},
            title={"text":"Clients identifiés à risque de départ","font":{"size":12,"color":"#8A8FA8"}},
            gauge={"axis":{"range":[0,30],"tickvals":[0,10,20,30],"ticktext":["0%","10% ⚠️","20% 🚨","30%"]},
                "bar":{"color":"#E63946","thickness":0.25},
                "bgcolor":"#F4F6F9","borderwidth":0,
                "steps":[{"range":[0,10],"color":"#E8F8F5"},
                            {"range":[10,20],"color":"#FFF3E0"},
                            {"range":[20,30],"color":"#FFE5E7"}],
                "threshold":{"line":{"color":"#E63946","width":3},"thickness":0.75,"value":10}}
        ))
        fig.update_layout(**PLOTLY_BASE, height=260)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"""
        <div style='background:#FFE5E7;border-radius:8px;padding:10px 14px;
                    border-left:4px solid #E63946;font-size:12px;color:#1A1A2E'>
            <b>📌 En clair :</b> {nrisk:,} clients sur {nb:,} sont susceptibles de partir,
            représentant <b>{mrr_e:,.0f}€</b> de MRR mensuel exposé.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-title'>Répartition du Risque</div>", unsafe_allow_html=True)
        rc = df["risk_level"].value_counts().reset_index()
        rc.columns = ["risk_level","count"]
        fig = px.pie(rc, values="count", names="risk_level", hole=0.62,
                    color="risk_level", color_discrete_map=COLORS["risk"])
        fig.update_traces(textposition="outside", textinfo="percent+label", textfont_size=11)
        fig.add_annotation(text=f"<b>{nb:,}</b><br>clients", x=0.5, y=0.5,
                        showarrow=False, font=dict(size=14,color="#1A1A2E"))
        fig.update_layout(**PLOTLY_BASE, height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Ligne 2 : MRR par Segment + Engagement moyen par segment
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("<div class='section-title'>MRR par Segment & Niveau de Risque</div>", unsafe_allow_html=True)
        seg = df.groupby("customer_segment").apply(lambda x: pd.Series({
            "Sécurisé": x[x["proba_churn"]<0.40]["monthly_fee"].sum(),
            "Modéré":   x[(x["proba_churn"]>=0.40)&(x["proba_churn"]<0.75)]["monthly_fee"].sum(),
            "Critique": x[x["proba_churn"]>=0.75]["monthly_fee"].sum(),
        }), include_groups=False).reset_index()
        fig = px.bar(seg, x="customer_segment", y=["Sécurisé","Modéré","Critique"],
                    barmode="stack",
                    color_discrete_sequence=["#2A9D8F","#F4A261","#E63946"])
        fig.update_layout(**PLOTLY_BASE, height=500,
                        xaxis={**ax(),"title":"Segment"},
                        yaxis={**ax(),"tickprefix":"€","title":"MRR (€)"},
                        legend=dict(orientation="h",y=-0.25,font_size=11))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("<div class='section-title'>Revenu à Risque par Segment</div>", unsafe_allow_html=True)
        risk_seg = (df.groupby("customer_segment")
                    .agg(expected_loss=("expected_loss_mensuel","sum"),
                        nb_clients=("customer_id","count"))
                    .reset_index()
                    .sort_values("expected_loss", ascending=True))
        risk_seg["label"] = risk_seg["expected_loss"].apply(lambda x: f"{x:,.0f}€")
        fig = px.bar(risk_seg, x="expected_loss", y="customer_segment",
                    orientation="h", text="label",
                    color="expected_loss",
                    color_continuous_scale=["#F4A261","#E63946"])
        fig.update_traces(textposition="inside", textfont=dict(color="white", size=12))
        fig.update_layout(**PLOTLY_BASE, height=500, showlegend=False,
                        coloraxis_showscale=False,
                        xaxis={**ax(), "title": "Expected Loss Mensuel (€)"},
                        yaxis={**ax(), "title": ""})
        st.plotly_chart(fig, use_container_width=True)



# ═════════════════════════════════════════════
# PAGE 2 — ANALYSE CHURN
# ═════════════════════════════════════════════
elif "Churn" in page:
    st.markdown("## 🎯 Analyse Churn — Profils & Comportements")
    st.markdown("<p style='color:#8A8FA8;margin-top:-10px'>Explorez les facteurs de risque avec les filtres interactifs.</p>", unsafe_allow_html=True)
    st.markdown("---")

    with st.sidebar:
        st.markdown("<hr style='border-color:#2D3561'>", unsafe_allow_html=True)
        st.markdown("<div style='color:#FFF;font-weight:700;font-size:13px;margin-bottom:10px'>🔽 Filtres</div>", unsafe_allow_html=True)
        f_seg   = st.multiselect("Segment",  df["customer_segment"].unique(), default=list(df["customer_segment"].unique()))
        f_con   = st.multiselect("Contrat",  df["contract_type"].unique(),    default=list(df["contract_type"].unique()))
        f_seuil = st.slider("Proba minimum", 0.0, 1.0, 0.0, 0.05)

    df_f = df[
        df["customer_segment"].isin(f_seg) & 
        df["contract_type"].isin(f_con) & 
        (df["proba_churn"] >= f_seuil)
        ].copy()
    
    # ── KPIs
    c1,c2,c3,c4 = st.columns(4)
    top_r = df_f.loc[df_f["proba_churn"].idxmax()] if not df_f.empty else None
    c1.markdown(kpi("Clients Filtrés",   f"{len(df_f):,}",                       "dans la sélection","blue"),  unsafe_allow_html=True)
    c2.markdown(kpi("Proba Churn Moy.",  f"{df_f['proba_churn'].mean()*100:.1f}%","sur le segment","orange"),  unsafe_allow_html=True)
    c3.markdown(kpi("CSAT Moyen",        f"{df_f['csat_score'].mean():.2f}/5",    "satisfaction","green"),     unsafe_allow_html=True)
    c4.markdown(kpi("Client + à Risque",
                    top_r["customer_id"] if top_r is not None else "N/A",
                    f"Proba: {top_r['proba_churn']*100:.1f}%" if top_r is not None else "","red"),
                unsafe_allow_html=True)
    st.markdown("")

    # ── Ligne 1 : Ancienneté + Tickets & Échecs par niveau de risque
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Taux de Churn par Tranche d'Ancienneté</div>", unsafe_allow_html=True)
        df_f2 = df_f.copy()
        df_f2["tranche_tenure"] = pd.cut(
            df_f2["tenure_months"],
            bins=[0, 12, 24, 36, 200],
            labels=["0–12 mois", "12–24 mois", "24–36 mois", "36+ mois"]
        )
        churn_tenure = (df_f2.groupby("tranche_tenure", observed=True)["churn"]
                            .mean().reset_index()
                            .sort_values("tranche_tenure"))
        churn_tenure["couleur"] = churn_tenure["churn"].apply(
            lambda x: "#E63946" if x >= 0.15 else ("#F4A261" if x >= 0.08 else "#2A9D8F")
        )
        churn_tenure["label"] = churn_tenure["churn"].apply(lambda x: f"{x*100:.1f}%")
        fig = px.bar(churn_tenure, x="tranche_tenure", y="churn",
                    text="label", color="couleur", color_discrete_map="identity")
        fig.update_traces(textposition="outside")
        fig.update_layout(**PLOTLY_BASE, height=350, showlegend=False,
                        xaxis={**ax(), "title": "Ancienneté du Client"},
                        yaxis={**ax(), "title": "Taux de Churn Réel", "tickformat": ".0%"})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div style='background:#F8F9FA;border-radius:8px;padding:10px 14px;
                    border-left:4px solid #4361EE;font-size:12px;color:#1A1A2E'>
            💡 <b>Action :</b> Intensifier l'onboarding sur les 12 premiers mois —
            c'est la période la plus critique pour la rétention.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-title'>Signaux d'Alerte Moyens par Niveau de Risque</div>", unsafe_allow_html=True)
        signal = df_f.groupby("risk_level").agg(
            tickets=("support_tickets", "mean"),
            echecs=("payment_failures", "mean")
        ).reset_index()
        signal_m = signal.melt(id_vars="risk_level", value_vars=["tickets","echecs"],
                               var_name="Signal", value_name="Moyenne")
        signal_m["Signal"] = signal_m["Signal"].map({
            "tickets": "Tickets Support",
            "echecs":  "Échecs Paiement"
        })
        signal_m["risk_level"] = pd.Categorical(
            signal_m["risk_level"], categories=["Faible","Modéré","Critique"], ordered=True
        )
        signal_m = signal_m.sort_values("risk_level")
        fig = px.bar(signal_m, x="risk_level", y="Moyenne", color="Signal",
                    barmode="group", text=signal_m["Moyenne"].apply(lambda x: f"{x:.1f}"),
                    color_discrete_map={"Tickets Support": "#4361EE", "Échecs Paiement": "#E63946"})
        fig.update_traces(textposition="outside")
        fig.update_layout(**PLOTLY_BASE, height=350,
                        xaxis={**ax(), "title": "Niveau de Risque"},
                        yaxis={**ax(), "title": "Moyenne par Client"},
                        legend=dict(orientation="h", y=-0.25, font_size=11))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div style='background:#F8F9FA;border-radius:8px;padding:10px 14px;
                    border-left:4px solid #4361EE;font-size:12px;color:#1A1A2E'>
            💡 <b>Action :</b> Déclencher une alerte automatique dès 3 tickets
            ouverts ou 2 échecs de paiement consécutifs.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Ligne 2 : Échecs de paiement + CSAT par tranches
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("<div class='section-title'>Probabilité de Churn selon les Échecs de Paiement</div>", unsafe_allow_html=True)
        df_f3 = df_f.copy()
        df_f3["tranche_failures"] = pd.cut(
            df_f3["payment_failures"],
            bins=[-1, 0, 1, 2, 10],
            labels=["0 échec", "1 échec", "2 échecs", "3+ échecs"]
        )
        churn_fail = (df_f3.groupby("tranche_failures", observed=True)["proba_churn"]
                          .mean().reset_index()
                          .sort_values("tranche_failures"))
        churn_fail["couleur"] = churn_fail["proba_churn"].apply(
            lambda x: "#E63946" if x >= 0.40 else ("#F4A261" if x >= 0.20 else "#2A9D8F")
        )
        churn_fail["label"] = churn_fail["proba_churn"].apply(lambda x: f"{x*100:.1f}%")
        fig = px.bar(churn_fail, x="tranche_failures", y="proba_churn",
                    text="label", color="couleur", color_discrete_map="identity")
        fig.update_traces(textposition="outside")
        fig.update_layout(**PLOTLY_BASE, height=350, showlegend=False,
                        xaxis={**ax(), "title": "Nombre d'Échecs de Paiement"},
                        yaxis={**ax(), "title": "Probabilité de Churn Moyenne", "tickformat": ".0%"})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div style='background:#F8F9FA;border-radius:8px;padding:10px 14px;
                    border-left:4px solid #4361EE;font-size:12px;color:#1A1A2E'>
            💡 <b>Action :</b> Contacter immédiatement tout client avec 2+ échecs
            de paiement — risque de churn fortement amplifié.
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("<div class='section-title'>Probabilité de Churn par Tranche de Satisfaction (CSAT)</div>", unsafe_allow_html=True)
        df_f4 = df_f.copy()
        df_f4["tranche_csat"] = pd.cut(
            df_f4["csat_score"],
            bins=[0, 2, 3, 4, 5],
            labels=["1–2 (Très insatisfait)", "2–3 (Insatisfait)", "3–4 (Neutre)", "4–5 (Satisfait)"]
        )
        churn_csat = (df_f4.groupby("tranche_csat", observed=True)["proba_churn"]
                          .mean().reset_index()
                          .sort_values("tranche_csat"))
        churn_csat["couleur"] = churn_csat["proba_churn"].apply(
            lambda x: "#E63946" if x >= 0.40 else ("#F4A261" if x >= 0.20 else "#2A9D8F")
        )
        churn_csat["label"] = churn_csat["proba_churn"].apply(lambda x: f"{x*100:.1f}%")
        fig = px.bar(churn_csat, x="proba_churn", y="tranche_csat",
                    orientation="h", text="label",
                    color="couleur", color_discrete_map="identity")
        fig.update_traces(textposition="outside")
        fig.update_layout(**PLOTLY_BASE, height=350, showlegend=False,
                        xaxis={**ax(), "title": "Probabilité de Churn Moyenne", "tickformat": ".0%"},
                        yaxis={**ax(), "title": ""})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div style='background:#F8F9FA;border-radius:8px;padding:10px 14px;
                    border-left:4px solid #4361EE;font-size:12px;color:#1A1A2E'>
            💡 <b>Action :</b> Toute note CSAT inférieure à 3 doit déclencher
            un appel de satisfaction dans les 48h.
        </div>
        """, unsafe_allow_html=True)

    # ── Ligne 3 : Feature Importance Globale
    st.markdown("")
    st.markdown("<div class='section-title'>🎯 Variables les Plus Influentes sur le Churn</div>", unsafe_allow_html=True)

    # Récupérer la feature importance native de XGBoost
    importance = model.feature_importances_

    # Récupérer les noms de features
    try:
        num_names = preprocessor.named_transformers_["num"].feature_names_in_
        cat_input_features = preprocessor.transformers_[1][2]
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = cat_encoder.get_feature_names_out(cat_input_features)
        feat_names = list(num_names) + list(cat_names)
    except Exception:
        try:
            feat_names = list(preprocessor.get_feature_names_out())
        except Exception:
            feat_names = [f"feature_{i}" for i in range(len(importance))]

    # Mapping noms techniques → langage métier
    labels_fi = {
        "payment_failures":    "Échecs de paiement",
        "support_tickets":     "Tickets support",
        "csat_score":          "Score CSAT",
        "tenure_months":       "Ancienneté client",
        "last_login_days_ago": "Jours depuis dernière connexion",
        "monthly_logins":      "Connexions mensuelles",
        "nps_score":           "Score NPS",
        "contract_type":       "Type de contrat",
        "weekly_active_days":  "Jours actifs / semaine",
        "monthly_fee":         "MRR mensuel",
        "avg_resolution_time": "Temps résolution support",
        "escalations":         "Escalades support",
        "email_open_rate":     "Taux ouverture emails",
        "features_used":       "Fonctionnalités utilisées",
        "usage_growth_rate":   "Taux de croissance d'usage",
        "age":                 "Âge du client",
        "total_revenue":       "Revenu total historique",
    }

    def nom_fi(f):
        for k, v in labels_fi.items():
            if k in f:
                return v
        return f

    fi_df = pd.DataFrame({
        "feature":    [nom_fi(f) for f in feat_names],
        "importance": importance
    }).groupby("feature")["importance"].sum().reset_index()  # groupby car OHE crée plusieurs colonnes par variable
    fi_df = fi_df.sort_values("importance", ascending=True).tail(12)

    fi_df["couleur"] = fi_df["importance"].apply(
        lambda x: "#E63946" if x >= fi_df["importance"].quantile(0.75)
        else ("#F4A261" if x >= fi_df["importance"].quantile(0.50)
        else "#4361EE")
    )

    fig = px.bar(fi_df, x="importance", y="feature",
                orientation="h",
                text=fi_df["importance"].apply(lambda x: f"{x:.3f}"),
                color="couleur", color_discrete_map="identity")
    fig.update_traces(textposition="outside")
    fig.update_layout(**PLOTLY_BASE, height=500, showlegend=False,
                    xaxis={**ax(), "title": "Importance (contribution à la réduction d'impureté)"},
                    yaxis={**ax(), "title": ""})
    st.plotly_chart(fig, use_container_width=True)

    st.info("💡 **Comment lire ce graphique ?** Plus la barre est longue, plus la variable influence les décisions du modèle. Les variables en rouge sont les plus déterminantes pour prédire le churn — ce sont les leviers prioritaires pour vos campagnes de rétention.")

# ═════════════════════════════════════════════
# PAGE 3 — IMPACT FINANCIER
# ═════════════════════════════════════════════
elif "Financier" in page:
    st.markdown("## 💰 Impact Financier — Évaluation des Pertes")
    st.markdown("<p style='color:#8A8FA8;margin-top:-10px'>Quantifiez le risque financier et identifiez les comptes prioritaires.</p>", unsafe_allow_html=True)
    st.markdown("---")

    loss_m   = df["expected_loss_mensuel"].sum()
    loss_t   = df["expected_loss_total"].sum()
    mrr_s    = df[df["proba_churn"]<0.40]["monthly_fee"].sum()
    mrr_m    = df[(df["proba_churn"]>=0.40)&(df["proba_churn"]<0.75)]["monthly_fee"].sum()
    mrr_c    = df[df["proba_churn"]>=0.75]["monthly_fee"].sum()
    mrr_tot  = df["monthly_fee"].sum()
    pire     = df.loc[df["expected_loss_mensuel"].idxmax()]

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.markdown(kpi("Expected Loss/Mois",  f"{loss_m:,.0f}€",  "Σ(fee × proba)","red"),    unsafe_allow_html=True)
    c2.markdown(kpi("Expected Loss Total", f"{loss_t:,.0f}€",  "Σ(rev × proba)","red"),    unsafe_allow_html=True)
    c3.markdown(kpi("MRR Sécurisé",        f"{mrr_s:,.0f}€",  "Proba < 40%","green"),      unsafe_allow_html=True)
    c4.markdown(kpi("MRR Exposé",          f"{mrr_m+mrr_c:,.0f}€","Proba ≥ 40%","orange"), unsafe_allow_html=True)
    c5.markdown(kpi("Max Perte Potentielle",f"{pire['expected_loss_mensuel']:,.0f}€",
                    pire["customer_id"],"purple"), unsafe_allow_html=True)
    st.markdown("")

    col1, col2 = st.columns([1.2,1.8])
    with col1:
        st.markdown("<div class='section-title'>Décomposition du MRR</div>", unsafe_allow_html=True)
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute","relative","relative","relative","total"],
            x=["MRR Total","✅ Sécurisé","⚠️ Modéré","🚨 Critique","Exposé Total"],
            y=[mrr_tot,-mrr_s,-mrr_m,-mrr_c,0],
            text=[f"{mrr_tot:,.0f}€",f"-{mrr_s:,.0f}€",f"-{mrr_m:,.0f}€",f"-{mrr_c:,.0f}€",f"{mrr_m+mrr_c:,.0f}€"],
            textposition="outside",
            connector={"line":{"color":"#E8E8F0","width":1}},
            increasing={"marker":{"color":"#E63946"}},
            decreasing={"marker":{"color":"#2A9D8F"}},
            totals={"marker":{"color":"#F4A261"}},
        ))
        fig.update_layout(**PLOTLY_BASE, height=360, xaxis=ax(), yaxis={**ax(),"tickprefix":"€"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Top 15 Clients — Plus Fort Revenu à Risque</div>", unsafe_allow_html=True)
        top15 = df.nlargest(15,"expected_loss_mensuel").sort_values("expected_loss_mensuel",ascending=True)
        fig = px.bar(top15, x="expected_loss_mensuel", y="customer_id", orientation="h",
                    color="risk_level", color_discrete_map=COLORS["risk"],
                    text=top15["expected_loss_mensuel"].apply(lambda x:f"{x:.0f}€"),
                    hover_data=["monthly_fee","proba_churn","contract_type"])
        fig.update_traces(textposition="outside")
        fig.update_layout(**PLOTLY_BASE, height=360,
                        xaxis={**ax(),"title":"Expected Loss Mensuel (€)"},
                        yaxis={**ax(),"title":""},
                        legend=dict(title="Risque",orientation="h",y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns([1.8, 1.2])

    with col3:
        st.markdown("<div class='section-title'>Expected Loss par Type de Contrat</div>", unsafe_allow_html=True)
        lc = (df.groupby("contract_type")["expected_loss_mensuel"]
                .sum().reset_index()
                .sort_values("expected_loss_mensuel", ascending=False))
        lc["label"] = lc["expected_loss_mensuel"].apply(lambda x: f"{x:,.0f}€")
        fig = px.bar(lc, x="contract_type", y="expected_loss_mensuel",
                    color="contract_type",
                    color_discrete_sequence=[COLORS["red"], COLORS["orange"], COLORS["green"]],
                    text="label")
        fig.update_traces(textposition="outside")
        fig.update_layout(**PLOTLY_BASE, height=320, showlegend=False,
                        xaxis={**ax(), "title": "Type de Contrat"},
                        yaxis={**ax(), "title": "Expected Loss Mensuel (€)"})
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("<div class='section-title'>Simulateur ROI de Rétention</div>", unsafe_allow_html=True)
        
        nb_critiques   = len(df[df["risk_level"] == "Critique"])
        mrr_crit_total = df[df["risk_level"] == "Critique"]["monthly_fee"].sum()
        
        pct_retention = st.slider(
            "% de clients critiques retenus", 5, 50, 20, 5,
            help="Estimation du taux de succès de la campagne de rétention"
        )
        cout_action = st.slider(
            "Coût par action de rétention (€)", 10, 200, 50, 10,
            help="Coût estimé d'un appel, email personnalisé ou remise"
        )
        
        nb_retenus     = int(nb_critiques * pct_retention / 100)
        gain_brut      = mrr_crit_total * pct_retention / 100
        cout_total     = nb_retenus * cout_action
        roi_net        = gain_brut - cout_total
        roi_couleur    = "#2A9D8F" if roi_net > 0 else "#E63946"
        roi_icon       = "📈" if roi_net > 0 else "📉"

        st.markdown(f"""
        <div style='background:#FFFFFF;border-radius:12px;padding:18px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.07);margin-top:10px'>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px'>
                <div style='text-align:center'>
                    <div style='font-size:10px;font-weight:700;color:#8A8FA8;
                                text-transform:uppercase;letter-spacing:0.8px'>
                        Clients Retenus
                    </div>
                    <div style='font-size:22px;font-weight:800;color:#4361EE'>
                        {nb_retenus}
                    </div>
                    <div style='font-size:10px;color:#8A8FA8'>
                        sur {nb_critiques} critiques
                    </div>
                </div>
                <div style='text-align:center'>
                    <div style='font-size:10px;font-weight:700;color:#8A8FA8;
                                text-transform:uppercase;letter-spacing:0.8px'>
                        Coût Campagne
                    </div>
                    <div style='font-size:22px;font-weight:800;color:#F4A261'>
                        {cout_total:,.0f}€
                    </div>
                    <div style='font-size:10px;color:#8A8FA8'>
                        {cout_action}€ × {nb_retenus} clients
                    </div>
                </div>
                <div style='text-align:center'>
                    <div style='font-size:10px;font-weight:700;color:#8A8FA8;
                                text-transform:uppercase;letter-spacing:0.8px'>
                        Gain MRR Récupéré
                    </div>
                    <div style='font-size:22px;font-weight:800;color:#2A9D8F'>
                        {gain_brut:,.0f}€
                    </div>
                    <div style='font-size:10px;color:#8A8FA8'>
                        MRR mensuel sauvegardé
                    </div>
                </div>
                <div style='text-align:center'>
                    <div style='font-size:10px;font-weight:700;color:#8A8FA8;
                                text-transform:uppercase;letter-spacing:0.8px'>
                        ROI Net {roi_icon}
                    </div>
                    <div style='font-size:22px;font-weight:800;color:{roi_couleur}'>
                        {roi_net:+,.0f}€
                    </div>
                    <div style='font-size:10px;color:#8A8FA8'>
                        gain - coût campagne
                    </div>
                </div>
            </div>
            <hr style='border-color:#E8E8F0;margin:14px 0'>
            <div style='font-size:11px;color:#8A8FA8;text-align:center'>
                Ajustez les curseurs pour simuler différents scénarios
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════
# PAGE 4 — SIMULATEUR CRM
# ═════════════════════════════════════════════
elif "Simulateur" in page:
    st.markdown("## ⚙️ Simulateur Client — Plan d'Action CRM")
    st.markdown("<p style='color:#8A8FA8;margin-top:-10px'>Modifiez les paramètres et observez l'impact sur le risque en temps réel.</p>", unsafe_allow_html=True)
    st.markdown("---")

    c_id = st.selectbox("🔍 Sélectionnez un client :", df["customer_id"].unique())
    cust = df[df["customer_id"]==c_id].iloc[0].to_dict()

    st.markdown("<div class='section-title'>Profil Actuel du Client</div>", unsafe_allow_html=True)
    p1,p2,p3,p4,p5,p6 = st.columns(6)
    p1.markdown(kpi("Segment",      cust["customer_segment"],"","blue"),  unsafe_allow_html=True)
    p2.markdown(kpi("Contrat",      cust["contract_type"],   "","blue"),  unsafe_allow_html=True)
    p3.markdown(kpi("MRR",          f"{cust['monthly_fee']}€","","green"),unsafe_allow_html=True)
    p4.markdown(kpi("CSAT",         f"{cust['csat_score']:.1f}/5","","green" if cust["csat_score"]>=3.5 else "red"),unsafe_allow_html=True)
    p5.markdown(kpi("Tickets",      f"{cust['support_tickets']}","","orange" if cust["support_tickets"]>3 else "blue"),unsafe_allow_html=True)
    p6.markdown(kpi("Risque Actuel",f"{cust['proba_churn']*100:.1f}%",cust["risk_level"],
                    "red" if cust["proba_churn"]>0.75 else ("orange" if cust["proba_churn"]>0.40 else "green")),
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='section-title'>Paramètres de Simulation</div>", unsafe_allow_html=True)

    col1,col2,col3 = st.columns(3)
    with col1:
        st.markdown("**Usage**")
        sim_logins = st.slider("Connexions Mensuelles", 0, 50, int(cust["monthly_logins"]))
        sim_days   = st.slider("Jours depuis connexion",0, 30, int(cust["last_login_days_ago"]))
        sim_active = st.slider("Jours Actifs/Semaine",  0, 7,  int(cust["weekly_active_days"]))
    with col2:
        st.markdown("**Satisfaction**")
        sim_csat    = st.slider("CSAT",    1.0, 5.0,   float(cust["csat_score"]) if pd.notna(cust["csat_score"]) else 3.0, 0.5)
        sim_tickets = st.slider("Tickets", 0,   10,    int(cust["support_tickets"]))
        sim_nps     = st.slider("NPS",    -100, 100,   int(cust["nps_score"]))
    with col3:
        st.markdown("**Financier**")
        sim_failures = st.slider("Échecs Paiement", 0, 5, int(cust["payment_failures"]))
        opts = ["Monthly","Quarterly","Yearly"]
        sim_contract = st.selectbox("Contrat Simulé", opts,
                                    index=opts.index(cust["contract_type"].capitalize())
                                    if cust["contract_type"].capitalize() in opts else 0)

    input_d = {k:v for k,v in cust.items()}
    input_d.update({"monthly_logins":sim_logins,"last_login_days_ago":sim_days,
                    "weekly_active_days":sim_active,"csat_score":sim_csat,
                    "support_tickets":sim_tickets,"nps_score":sim_nps,
                    "payment_failures":sim_failures,"contract_type":sim_contract})
    drop_c = ["churn","customer_id","proba_churn","risk_level",
            "expected_loss_mensuel","expected_loss_total","engagement_score"]
    inp_df    = pd.DataFrame([input_d]).drop(columns=drop_c, errors="ignore")
    sim_proba = model.predict_proba(preprocessor.transform(inp_df))[0][1]
    act_proba = cust["proba_churn"]
    delta     = sim_proba - act_proba

    st.markdown("---")
    st.markdown("<div class='section-title'>Résultat de la Simulation</div>", unsafe_allow_html=True)
    rc1,rc2,rc3 = st.columns([1,1,2])

    def gauge_fig(val, title):
        c = "#E63946" if val>0.75 else ("#F4A261" if val>0.4 else "#2A9D8F")
        f = go.Figure(go.Indicator(
            mode="gauge+number", value=val*100,
            number={"suffix":"%","font":{"size":30}},
            title={"text":title,"font":{"size":12,"color":"#8A8FA8"}},
            gauge={"axis":{"range":[0,100]},"bar":{"color":c},
                "steps":[{"range":[0,40],"color":"#E8F8F5"},
                            {"range":[40,75],"color":"#FFF3E0"},
                            {"range":[75,100],"color":"#FFE5E7"}]}
        ))
        f.update_layout(**PLOTLY_BASE, height=240)
        return f

    with rc1:
        st.plotly_chart(gauge_fig(act_proba,"Risque ACTUEL"), use_container_width=True)
    with rc2:
        fig_s = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=sim_proba*100,
            number={"suffix":"%","font":{"size":30}},
            delta={"reference":act_proba*100,"increasing":{"color":"#E63946"},"decreasing":{"color":"#2A9D8F"},"suffix":"%"},
            title={"text":"Risque SIMULÉ","font":{"size":12,"color":"#8A8FA8"}},
            gauge={"axis":{"range":[0,100]},
                "bar":{"color":"#E63946" if sim_proba>0.75 else ("#F4A261" if sim_proba>0.4 else "#2A9D8F")},
                "steps":[{"range":[0,40],"color":"#E8F8F5"},
                            {"range":[40,75],"color":"#FFF3E0"},
                            {"range":[75,100],"color":"#FFE5E7"}]}
        ))
        fig_s.update_layout(**PLOTLY_BASE, height=240)
        st.plotly_chart(fig_s, use_container_width=True)

    with rc3:
        if sim_proba>0.75:
            bg,icon,msg = "#FFE5E7","🚨","**Appel commercial urgent.** Proposer une remise ou upgrade immédiat. Escalader si segment Enterprise."
        elif sim_proba>0.40:
            bg,icon,msg = "#FFF3E0","⚠️","**Fidélisation à planifier.** Campagne email + audit satisfaction. Proposer contrat annuel avec remise."
        else:
            bg,icon,msg = "#E8F8F5","✅","**Client stabilisé.** Maintenir dans le flux standard. Surveiller l'engagement mensuel."
        st.markdown(f"""
        <div style='background:{bg};padding:20px;border-radius:12px;border:1px solid #E8E8F0;margin-top:10px'>
            <div style='font-size:22px;margin-bottom:8px'>{icon}</div>
            <div style='font-size:13px;color:#1A1A2E;line-height:1.6'>{msg}</div>
            <hr style='border-color:#E8E8F0;margin:12px 0'>
            <div style='font-size:13px;font-weight:700'>
                {'📉' if delta<0 else '📈'} Variation : {delta*100:+.1f}%
            </div>
            <div style='font-size:11px;color:#8A8FA8;margin-top:4px'>
                Perte mensuelle simulée : <b>{sim_proba*cust['monthly_fee']:,.0f}€</b>
                (vs {act_proba*cust['monthly_fee']:,.0f}€ actuel)
            </div>
        </div>
        """, unsafe_allow_html=True)


    # ─────────────────────────────────────────────
    # SHAP — EXPLICABILITÉ LOCALE
    # ─────────────────────────────────────────────
    st.markdown("<div class='section-title'>🔍 Explication SHAP — Facteurs du Risque Simulé</div>", unsafe_allow_html=True)

    with st.spinner("Calcul des contributions SHAP..."):
        try:
            import shap

            explainer       = shap.TreeExplainer(model)
            proc_sim        = preprocessor.transform(inp_df)

            # ── Noms de features
            try:
                num_names   = preprocessor.named_transformers_["num"].feature_names_in_
                cat_input_features = preprocessor.transformers_[1][2]
                cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
                cat_names   = cat_encoder.get_feature_names_out(cat_input_features)
                feat_names  = list(num_names) + list(cat_names)
            except Exception:
                try:
                    feat_names = list(preprocessor.get_feature_names_out())
                except Exception:
                    feat_names = [f"feature_{i}" for i in range(proc_sim.shape[1])]

            # ── Calcul SHAP
            shap_values_raw = explainer.shap_values(proc_sim)

            if isinstance(shap_values_raw, list):
                vals     = shap_values_raw[1][0]
            elif shap_values_raw.ndim == 3:
                vals     = shap_values_raw[0, :, 1]
            else:
                vals     = shap_values_raw[0]

            # ── Mapping noms techniques → langage métier
            labels = {
                "payment_failures":    "échecs de paiement",
                "support_tickets":     "tickets support ouverts",
                "csat_score":          "score de satisfaction (CSAT)",
                "tenure_months":       "ancienneté du client",
                "last_login_days_ago": "jours depuis la dernière connexion",
                "monthly_logins":      "fréquence de connexion mensuelle",
                "nps_score":           "score NPS",
                "contract_type":       "type de contrat",
                "weekly_active_days":  "jours actifs par semaine",
                "monthly_fee":         "montant mensuel facturé",
                "avg_resolution_time": "temps moyen de résolution support",
                "escalations":         "escalades support",
                "email_open_rate":     "taux d'ouverture des emails",
                "features_used":       "fonctionnalités utilisées",
                "usage_growth_rate":   "taux de croissance d'usage",
            }

            def nom(f):
                for k, v in labels.items():
                    if k in f:
                        return v
                return f

            # ── Top 5 features par impact absolu
            shap_df = pd.DataFrame({
                "feature": feat_names,
                "shap":    vals
            })
            shap_df["abs"] = shap_df["shap"].abs()
            shap_df = shap_df.sort_values("abs", ascending=False).head(5)

            risques     = shap_df[shap_df["shap"] > 0].head(3)
            protecteurs = shap_df[shap_df["shap"] < 0].head(2)

            # ── Génération des phrases
            lignes_risque = "".join([
                f"""<li style='margin-bottom:6px'>
                    ⚠️ <b>{nom(row['feature'])}</b>
                    <span style='color:#8A8FA8;font-size:11px'>
                        — impact : +{row['shap']:.3f} sur la probabilité de churn
                    </span>
                </li>"""
                for _, row in risques.iterrows()
            ]) if len(risques) > 0 else "<li style='color:#8A8FA8'>Aucun facteur aggravant détecté</li>"

            lignes_protec = "".join([
                f"""<li style='margin-bottom:6px'>
                    ✅ <b>{nom(row['feature'])}</b>
                    <span style='color:#8A8FA8;font-size:11px'>
                        — impact : {row['shap']:.3f} sur la probabilité de churn
                    </span>
                </li>"""
                for _, row in protecteurs.iterrows()
            ]) if len(protecteurs) > 0 else "<li style='color:#8A8FA8'>Aucun facteur protecteur détecté</li>"

            # ── Affichage avec composants natifs Streamlit
            st.markdown("**🔍 Pourquoi ce client est-il classé à ce niveau de risque ?**")

            st.error("🚨 Facteurs qui augmentent le risque de départ")
            if len(risques) > 0:
                for _, row in risques.iterrows():
                    st.markdown(f"⚠️ **{nom(row['feature'])}** — impact : `+{row['shap']:.3f}` sur la probabilité de churn")
            else:
                st.markdown("Aucun facteur aggravant détecté")

            st.success("✅ Facteurs qui jouent en faveur de la rétention")
            if len(protecteurs) > 0:
                for _, row in protecteurs.iterrows():
                    st.markdown(f"✅ **{nom(row['feature'])}** — impact : `{row['shap']:.3f}` sur la probabilité de churn")
            else:
                st.markdown("Aucun facteur protecteur détecté")

            st.info("💡 **Comment utiliser cette analyse ?** Agissez en priorité sur les facteurs rouges. Modifiez les curseurs pour simuler l'impact de vos actions.")

        except Exception as e:
            st.error(f"Erreur SHAP : {e}")