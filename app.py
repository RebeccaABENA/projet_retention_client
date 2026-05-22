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

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.markdown(kpi("Clients à Risque",    f"{nrisk:,}",     f"{taux*100:.1f}% du portefeuille","red"),    unsafe_allow_html=True)
    c2.markdown(kpi("Comptes Critiques",   f"{ncrit:,}",     "Proba > 75%","red"),                          unsafe_allow_html=True)
    c3.markdown(kpi("MRR Exposé",          f"{mrr_e:,.0f}€", f"/ {mrr_t:,.0f}€ total","orange"),           unsafe_allow_html=True)
    c4.markdown(kpi("Expected Loss Total", f"{loss:,.0f}€",  "Σ(rev × proba)","purple"),                   unsafe_allow_html=True)
    c5.markdown(kpi("Engagement Moyen",    f"{eng:.1f}/100", "Score comportemental","green" if eng>60 else "red"), unsafe_allow_html=True)

    st.markdown("")
    col1, col2, col3 = st.columns([1.1,1.1,1.8])

    with col1:
        st.markdown("<div class='section-title'>Risque Global</div>", unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=taux*100,
            number={"suffix":"%","font":{"size":34}},
            delta={"reference":10,"increasing":{"color":"#E63946"},"decreasing":{"color":"#2A9D8F"}},
            title={"text":"Taux de Churn Prédit","font":{"size":12,"color":"#8A8FA8"}},
            gauge={"axis":{"range":[0,30]},"bar":{"color":"#E63946","thickness":0.25},
                "bgcolor":"#F4F6F9","borderwidth":0,
                "steps":[{"range":[0,10],"color":"#E8F8F5"},
                            {"range":[10,20],"color":"#FFF3E0"},
                            {"range":[20,30],"color":"#FFE5E7"}]}
        ))
        fig.update_layout(**PLOTLY_BASE, height=250)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Répartition du Risque</div>", unsafe_allow_html=True)
        rc = df["risk_level"].value_counts().reset_index()
        rc.columns = ["risk_level","count"]
        fig = px.pie(rc, values="count", names="risk_level", hole=0.62,
                    color="risk_level", color_discrete_map=COLORS["risk"])
        fig.update_traces(textposition="outside", textinfo="percent+label", textfont_size=11)
        fig.add_annotation(text=f"<b>{nb:,}</b><br>clients", x=0.5, y=0.5,
                        showarrow=False, font=dict(size=14,color="#1A1A2E"))
        fig.update_layout(**PLOTLY_BASE, height=250, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.markdown("<div class='section-title'>MRR par Segment & Niveau de Risque</div>", unsafe_allow_html=True)
        seg = df.groupby("customer_segment").apply(lambda x: pd.Series({
    "Sécurisé":  x[x["proba_churn"]<0.40]["monthly_fee"].sum(),
    "Modéré":    x[(x["proba_churn"]>=0.40)&(x["proba_churn"]<0.75)]["monthly_fee"].sum(),
    "Critique":  x[x["proba_churn"]>=0.75]["monthly_fee"].sum(),
}), include_groups=False).reset_index()
        fig = px.bar(seg, x="customer_segment", y=["Sécurisé","Modéré","Critique"],
                    barmode="stack", color_discrete_sequence=["#2A9D8F","#F4A261","#E63946"])
        fig.update_layout(**PLOTLY_BASE, height=250,
                        xaxis=ax(), yaxis={**ax(),"tickprefix":"€"},
                        legend=dict(orientation="h",y=-0.3,font_size=11))
        st.plotly_chart(fig, use_container_width=True)

    col4, col5 = st.columns(2)
    with col4:
        st.markdown("<div class='section-title'>Engagement par Niveau de Risque</div>", unsafe_allow_html=True)
        dp = df.copy()
        dp["Churn Réel"] = dp["churn"].map({0:"No Churn",1:"Churn"})
        fig = px.box(df, x="risk_level", y="engagement_score", color="risk_level",
                    color_discrete_map=COLORS["risk"],
                    category_orders={"risk_level":["Faible","Modéré","Critique"]}, points="outliers")
        fig.update_layout(**PLOTLY_BASE, height=300, showlegend=False,
                        xaxis={**ax(),"title":""},yaxis={**ax(),"title":"Score Engagement"})
        st.plotly_chart(fig, use_container_width=True)

    with col5:
        st.markdown("<div class='section-title'>Distribution des Probabilités de Churn</div>", unsafe_allow_html=True)
        fig = px.histogram(df, x="proba_churn", nbins=40, color_discrete_sequence=["#4361EE"])
        fig.add_vline(x=0.40, line_dash="dash", line_color="#F4A261",
                    annotation_text="Seuil Modéré", annotation_font_color="#F4A261")
        fig.add_vline(x=0.75, line_dash="dash", line_color="#E63946",
                    annotation_text="Seuil Critique", annotation_font_color="#E63946")
        fig.update_layout(**PLOTLY_BASE, height=300,
                        xaxis={**ax(),"title":"Probabilité de Churn"},
                        yaxis={**ax(),"title":"Nombre de Clients"})
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

    df_f = df[df["customer_segment"].isin(f_seg) & df["contract_type"].isin(f_con) & (df["proba_churn"] >= f_seuil)]

    c1,c2,c3,c4 = st.columns(4)
    top_r = df_f.loc[df_f["proba_churn"].idxmax()] if not df_f.empty else None
    c1.markdown(kpi("Clients Filtrés",   f"{len(df_f):,}",                      "dans la sélection","blue"),   unsafe_allow_html=True)
    c2.markdown(kpi("Proba Churn Moy.",  f"{df_f['proba_churn'].mean()*100:.1f}%","sur le segment","orange"),  unsafe_allow_html=True)
    c3.markdown(kpi("CSAT Moyen",        f"{df_f['csat_score'].mean():.2f}/5",   "satisfaction","green"),      unsafe_allow_html=True)
    c4.markdown(kpi("Client + à Risque",
                    top_r["customer_id"] if top_r is not None else "N/A",
                    f"Proba: {top_r['proba_churn']*100:.1f}%" if top_r is not None else "","red"),
                unsafe_allow_html=True)
    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>Risque Moyen par Variable Catégorielle</div>", unsafe_allow_html=True)
        var_cat = st.selectbox("Variable :", ["contract_type","customer_segment","payment_method",
                                            "survey_response","signup_channel","discount_applied",
                                            "price_increase_last_3m","complaint_type"])
        cd = df_f.groupby(var_cat)["proba_churn"].mean().reset_index().sort_values("proba_churn", ascending=True)
        fig = px.bar(cd, x="proba_churn", y=var_cat, orientation="h",
                    color="proba_churn", color_continuous_scale=["#2A9D8F","#F4A261","#E63946"],
                    text=cd["proba_churn"].apply(lambda x:f"{x*100:.1f}%"))
        fig.update_traces(textposition="outside")
        fig.update_layout(**PLOTLY_BASE, height=320, coloraxis_showscale=False,
                        xaxis={**ax(),"title":"Proba Churn Moyenne","tickformat":".0%"},
                        yaxis={**ax(),"title":""})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Scatter : Engagement vs Risque de Churn</div>", unsafe_allow_html=True)
        fig = px.scatter(df_f, x="engagement_score", y="proba_churn",
                        color="risk_level", color_discrete_map=COLORS["risk"],
                        size="monthly_fee", size_max=14, opacity=0.6,
                        hover_data=["customer_id","contract_type","monthly_fee"])
        fig.add_hline(y=0.5, line_dash="dot", line_color="#8A8FA8",
                    annotation_text="Seuil 50%", annotation_font_color="#8A8FA8")
        fig.update_layout(**PLOTLY_BASE, height=320,
                        xaxis={**ax(),"title":"Score d'Engagement (0–100)"},
                        yaxis={**ax(),"title":"Probabilité de Churn"},
                        legend=dict(title="Risque",orientation="h",y=-0.25))
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<div class='section-title'>Heatmap : Risque Moyen Contrat × Segment</div>", unsafe_allow_html=True)
        hmap = pd.crosstab(df_f["contract_type"], df_f["customer_segment"],
                        values=df_f["proba_churn"], aggfunc="mean").fillna(0)
        fig = px.imshow(hmap, text_auto=".2f",
                        color_continuous_scale=["#E8F8F5","#F4A261","#E63946"], aspect="auto")
        fig.update_layout(**PLOTLY_BASE, height=280, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("<div class='section-title'>Analyse Univariée — Variable Numérique vs Churn</div>", unsafe_allow_html=True)
        num_var = st.selectbox("Variable numérique :", [
            "tenure_months","monthly_fee","csat_score","nps_score",
            "payment_failures","support_tickets","last_login_days_ago",
            "monthly_logins","engagement_score","avg_resolution_time"])
        dp = df_f.copy(); dp["Churn"] = dp["churn"].map({0:"No Churn",1:"Churn"})
        fig = px.violin(dp, x="Churn", y=num_var, color="Churn",
                        color_discrete_map={"No Churn":"#2A9D8F","Churn":"#E63946"},
                        box=True, points="outliers")
        fig.update_layout(**PLOTLY_BASE, height=280, showlegend=False,
                        xaxis={**ax(),"title":""},yaxis={**ax(),"title":num_var})
        st.plotly_chart(fig, use_container_width=True)


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

    col3, col4 = st.columns([1.8,1.2])
    with col3:
        st.markdown("<div class='section-title'>Quadrant Valeur × Risque — Urgences Absolues</div>", unsafe_allow_html=True)
        fig = px.scatter(df, x="total_revenue", y="proba_churn",
                        color="risk_level", color_discrete_map=COLORS["risk"],
                        size="monthly_fee", size_max=16, opacity=0.6,
                        hover_data=["customer_id","contract_type","customer_segment"])
        fig.add_hline(y=0.75, line_dash="dash", line_color="#E63946",
                    annotation_text="Seuil Critique", annotation_font_color="#E63946")
        fig.add_vline(x=df["total_revenue"].quantile(0.80), line_dash="dash", line_color="#F4A261",
                    annotation_text="Top 20% Valeur", annotation_font_color="#F4A261")
        fig.update_layout(**PLOTLY_BASE, height=360,
                        xaxis={**ax(),"title":"Revenu Total Historique (€)"},
                        yaxis={**ax(),"title":"Probabilité de Churn"},
                        legend=dict(title="Risque",orientation="h",y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("<div class='section-title'>Expected Loss par Contrat</div>", unsafe_allow_html=True)
        lc = df.groupby("contract_type")["expected_loss_mensuel"].sum().reset_index().sort_values("expected_loss_mensuel",ascending=False)
        fig = px.bar(lc, x="contract_type", y="expected_loss_mensuel",
                    color="contract_type",
                    color_discrete_sequence=[COLORS["red"],COLORS["orange"],COLORS["green"]],
                    text=lc["expected_loss_mensuel"].apply(lambda x:f"{x:,.0f}€"))
        fig.update_traces(textposition="outside")
        fig.update_layout(**PLOTLY_BASE, height=360, showlegend=False,
                        xaxis={**ax(),"title":""},yaxis={**ax(),"title":"Expected Loss (€)"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>📋 Comptes Critiques (Proba > 75% & MRR > Moyenne)</div>", unsafe_allow_html=True)
    crit = (df[(df["proba_churn"]>0.75)&(df["monthly_fee"]>df["monthly_fee"].mean())]
            .sort_values("expected_loss_mensuel",ascending=False)
            [["customer_id","customer_segment","contract_type","monthly_fee",
                "total_revenue","proba_churn","csat_score","payment_failures"]]
            .rename(columns={"customer_id":"ID Client","customer_segment":"Segment",
                            "contract_type":"Contrat","monthly_fee":"MRR (€)",
                            "total_revenue":"LTV (€)","proba_churn":"Proba Churn",
                            "csat_score":"CSAT","payment_failures":"Échecs Paiement"}))
    crit["Proba Churn"] = crit["Proba Churn"].apply(lambda x:f"{x*100:.1f}%")
    st.dataframe(crit, use_container_width=True, hide_index=True)


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

    # Radar comparatif
    st.markdown("<div class='section-title'>Radar Comparatif — Profil Actuel vs Simulé</div>", unsafe_allow_html=True)
    def norm(v,mn,mx): return (v-mn)/(mx-mn+1e-9)
    cats = ["Connexions","Jours Actifs","CSAT","NPS","Sans Tickets","Sans Échecs"]
    av = [norm(cust["monthly_logins"],0,50), norm(cust["weekly_active_days"],0,7),
        norm(cust["csat_score"],1,5),       norm(cust["nps_score"],-100,100),
        1-norm(cust["support_tickets"],0,10),1-norm(cust["payment_failures"],0,5)]
    sv = [norm(sim_logins,0,50), norm(sim_active,0,7),
        norm(sim_csat,1,5),     norm(sim_nps,-100,100),
        1-norm(sim_tickets,0,10),1-norm(sim_failures,0,5)]
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatterpolar(r=av+[av[0]], theta=cats+[cats[0]], fill="toself",
                                    name="Actuel", line_color="#E63946", fillcolor="rgba(230,57,70,0.15)"))
    fig_r.add_trace(go.Scatterpolar(r=sv+[sv[0]], theta=cats+[cats[0]], fill="toself",
                                    name="Simulé", line_color="#2A9D8F", fillcolor="rgba(42,157,143,0.15)"))
    fig_r.update_layout(**PLOTLY_BASE, height=380,
                        polar=dict(radialaxis=dict(visible=True,range=[0,1],tickfont_color="#8A8FA8"),
                                angularaxis=dict(tickfont_color="#1A1A2E"),bgcolor="rgba(0,0,0,0)"),
                        legend=dict(orientation="h",y=-0.1))
    st.plotly_chart(fig_r, use_container_width=True)

    # ─────────────────────────────────────────────
    # SHAP — EXPLICABILITÉ LOCALE
    # ─────────────────────────────────────────────
    if st.button("🔍 Générer l'explication SHAP", type="primary"):
        with st.spinner("Calcul des contributions SHAP en cours..."):
            try:
                import shap
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                # Explainer optimisé pour XGBoost
                explainer = shap.TreeExplainer(model)

                # Profil simulé transformé
                proc_sim = preprocessor.transform(inp_df)

                # Noms de features
                try:
                    cat_names  = preprocessor.named_transformers_["cat"]["onehot"] \
                                     .get_feature_names_out(CATEGORICAL_FEATURES)
                    feat_names = list(NUMERIC_FEATURES) + list(cat_names)
                except Exception:
                    feat_names = [f"feature_{i}" for i in range(proc_sim.shape[1])]

                # ── Calcul SHAP values (format ancien : liste de 2 arrays)
                shap_values_raw = explainer.shap_values(proc_sim)

                # Gestion des deux formats possibles
                if isinstance(shap_values_raw, list):
                    # Format ancien : [shap_class0, shap_class1]
                    vals     = shap_values_raw[1][0]
                    base_val = explainer.expected_value[1]
                elif shap_values_raw.ndim == 3:
                    # Format nouveau : (n_samples, n_features, n_classes)
                    vals     = shap_values_raw[0, :, 1]
                    base_val = explainer.expected_value[1]
                else:
                    # Format simple : (n_samples, n_features)
                    vals     = shap_values_raw[0]
                    base_val = (explainer.expected_value[1]
                                if isinstance(explainer.expected_value, (list, np.ndarray))
                                else explainer.expected_value)

                # Construction de l'objet Explanation
                explanation = shap.Explanation(
                    values        = vals,
                    base_values   = float(base_val),
                    data          = proc_sim[0],
                    feature_names = feat_names,
                )

                # Waterfall plot
                fig_shap, ax = plt.subplots(figsize=(10, 6))
                fig_shap.patch.set_facecolor("white")
                shap.plots.waterfall(explanation, max_display=15, show=False)
                plt.title("Contributions SHAP — Profil  simulé", fontsize=12, pad=10)
                plt.tight_layout()
                st.pyplot(fig_shap)
                plt.close(fig_shap)

                # Légende
                st.markdown("""
                <div style='background:#F8F9FA;border-radius:10px;padding:16px;
                            border-left:4px solid #4361EE;margin-top:12px'>
                    <div style='font-weight:700;font-size:13px;margin-bottom:8px'>
                        💡 Comment lire ce graphique ?
                    </div>
                    <ul style='font-size:12px;color:#555;line-height:1.8;margin:0;padding-left:16px'>
                        <li><b>E[f(x)]</b> = risque moyen du portefeuille (point de départ)</li>
                        <li><b>f(x)</b> = probabilité de churn du client simulé</li>
                        <li><span style='color:#E63946;font-weight:700'>Barres rouges</span>
                            = variables qui augmentent le risque</li>
                        <li><span style='color:#2A9D8F;font-weight:700'>Barres bleues</span>
                            = variables qui réduisent le risque</li>
                        <li>La longueur = intensité de l'impact</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Erreur lors du calcul SHAP : {e}")
                st.info("Vérifiez que SHAP est installé : pip install shap")