import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURATION DE LA PAGE & STYLES ---
st.set_page_config(page_title="Dashboard Rétention & Risque", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        /* Force la police Comic Sans MS et le fond blanc absolu */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], .stMarkdown, p, label, h1, h2, h3, div {
            font-family: 'Comic Sans MS', 'Chalkboard SE', 'Comic Neue', sans-serif !important;
            color: #111111 !important;
        }
        .stApp, [data-testid="stSidebar"], [data-testid="stHeader"] {
            background-color: #FFFFFF !important;
        }
        
        /* Conteneurs KPI */
        .kpi-container {
            background-color: #F8F9FA;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #E5E5E5;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .kpi-title { font-size: 14px; font-weight: bold; color: #555555 !important; margin-bottom: 5px; }
        .kpi-val { font-size: 24px; font-weight: bold; }
        .red-text { color: #D9534F !important; }
        .orange-text { color: #F0AD4E !important; }
        .green-text { color: #5CB85C !important; }
        
        /* Tableaux */
        [data-testid="stDataFrame"] div { background-color: #FFFFFF !important; color: #111111 !important; }
    </style>
""", unsafe_allow_html=True)

# --- CHARGEMENT DES DONNÉES ET MODÈLES ---
@st.cache_resource
def load_pipeline():
    models_dir = "models" if os.path.exists("models") else "../models"
    preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
    model = joblib.load(os.path.join(models_dir, "XGBoost.joblib"))
    return preprocessor, model

@st.cache_data
def load_and_prepare_data():
    data_path = "data/raw/customer_churn_business_dataset.csv" if os.path.exists("data/raw/customer_churn_business_dataset.csv") else "../data/raw/customer_churn_business_dataset.csv"
    df = pd.read_csv(data_path)
    
    # Création d'un proxy pour l'Engagement Score (0 à 100)
    # Basé sur l'activité (logins, jours actifs) et minoré par les jours depuis la dernière connexion
    df['engagement_score'] = ((df['monthly_logins'] / 30 * 40) + 
                              (df['weekly_active_days'] / 7 * 40) + 
                              (df['csat_score'].fillna(3) / 5 * 20))
    df['engagement_score'] = df['engagement_score'].clip(0, 100)
    return df

try:
    preprocessor, model = load_pipeline()
    df_raw = load_and_prepare_data()
except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()

# Inférer les probabilités sur l'ensemble du dataset
@st.cache_data
def compute_predictions(_preprocessor, _model, df):
    X = df.drop(columns=['churn', 'customer_id'], errors='ignore')
    X_processed = _preprocessor.transform(X)
    probs = _model.predict_proba(X_processed)[:, 1]
    
    res_df = df.copy()
    res_df['proba_churn'] = probs
    res_df['expected_loss_mensuel'] = res_df['monthly_fee'] * probs
    res_df['expected_loss_total'] = res_df['total_revenue'] * probs
    
    # Segmentation du risque
    conditions = [res_df['proba_churn'] < 0.4, res_df['proba_churn'] < 0.75]
    choices = ['Faible', 'Modéré']
    res_df['risk_level'] = np.select(conditions, choices, default='Critique')
    
    return res_df

df = compute_predictions(preprocessor, model, df_raw)

# Style commun des graphiques Plotly pour fond blanc
layout_transparent = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Comic Sans MS', color='#111111')
)

# --- NAVIGATION SIDEBAR ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("", [
    "📊 Page 1 : Vue Exécutive (Direction)", 
    "🎯 Page 2 : Analyse Churn (Marketing)", 
    "💰 Page 3 : Impact Financier (Finance)", 
    "⚙️ Page 4 : Simulateur (CRM)"
])

# =====================================================================
# PAGE 1 : VUE EXÉCUTIVE
# =====================================================================
if "Page 1" in page:
    st.title("📊 Vue Exécutive : Santé du Portefeuille")
    
    # KPIs
    clients_risque = df[df['proba_churn'] > 0.5]
    nb_risque = len(clients_risque)
    taux_churn = nb_risque / len(df)
    rev_mensuel_expose = clients_risque['monthly_fee'].sum()
    rev_total_risque = df['expected_loss_total'].sum()
    avg_engagement = df['engagement_score'].mean()
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(f'<div class="kpi-container"><div class="kpi-title">Clients à Risque (>50%)</div><div class="kpi-val red-text">{nb_risque}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-container"><div class="kpi-title">Taux Churn Prédit</div><div class="kpi-val red-text">{taux_churn*100:.1f}%</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-container"><div class="kpi-title">Revenu Mensuel Exposé</div><div class="kpi-val orange-text">{rev_mensuel_expose:,.0f} €</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-container"><div class="kpi-title">Revenu Total à Risque</div><div class="kpi-val red-text">{rev_total_risque:,.0f} €</div></div>', unsafe_allow_html=True)
    
    eng_color = "green-text" if avg_engagement > 60 else "red-text"
    c5.markdown(f'<div class="kpi-container"><div class="kpi-title">Score Engagement Moyen</div><div class="kpi-val {eng_color}">{avg_engagement:.1f}/100</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Graphiques
    g1, g2, g3 = st.columns([1, 1, 1.5])
    
    with g1:
        # Jauge de risque
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = taux_churn*100, title = {'text': "Risque Global (%)"},
            gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "#D9534F"},
                     'steps': [{'range': [0, 15], 'color': "#5CB85C"}, {'range': [15, 30], 'color': "#F0AD4E"}]}
        ))
        fig_gauge.update_layout(**layout_transparent, height=300, margin=dict(t=50, b=0, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    with g2:
        # Donut de répartition
        risk_counts = df['risk_level'].value_counts().reset_index()
        fig_donut = px.pie(risk_counts, values='count', names='risk_level', hole=0.6,
                           color='risk_level', color_discrete_map={'Faible':'#5CB85C', 'Modéré':'#F0AD4E', 'Critique':'#D9534F'},
                           title="Répartition des Niveaux de Risque")
        fig_donut.update_layout(**layout_transparent, height=300, margin=dict(t=50, b=0, l=0, r=0))
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with g3:
        # Barplot segment
        seg_data = df.groupby('customer_segment').apply(lambda x: pd.Series({
            'Sécurisé': x[x['proba_churn'] < 0.4]['monthly_fee'].sum(),
            'Exposé': x[x['proba_churn'] >= 0.4]['monthly_fee'].sum()
        })).reset_index()
        fig_bar = px.bar(seg_data, x='customer_segment', y=['Sécurisé', 'Exposé'], barmode='group',
                         color_discrete_map={'Sécurisé':'#5CB85C', 'Exposé':'#D9534F'},
                         title="Revenu à Risque par Segment")
        fig_bar.update_layout(**layout_transparent, height=300, margin=dict(t=50, b=0, l=0, r=0))
        st.plotly_chart(fig_bar, use_container_width=True)

# =====================================================================
# PAGE 2 : ANALYSE CHURN (MARKETING)
# =====================================================================
elif "Page 2" in page:
    st.title("🎯 Analyse Churn : Profils & Comportements")
    
    st.sidebar.markdown("### Filtres d'Analyse")
    f_seg = st.sidebar.multiselect("Segment Client", df['customer_segment'].unique(), default=df['customer_segment'].unique())
    f_con = st.sidebar.multiselect("Type de Contrat", df['contract_type'].unique(), default=df['contract_type'].unique())
    f_seuil = st.sidebar.slider("Seuil de Risque (Proba minimum)", 0.0, 1.0, 0.0, 0.05)
    
    df_filt = df[(df['customer_segment'].isin(f_seg)) & (df['contract_type'].isin(f_con)) & (df['proba_churn'] >= f_seuil)]
    
    # KPIs Dynamiques
    st.markdown("**Aperçu du segment filtré :**")
    kc1, kc2, kc3 = st.columns(3)
    kc1.metric("Nb Clients Filtrés", len(df_filt))
    kc2.metric("Proba Churn Moyenne", f"{df_filt['proba_churn'].mean()*100:.1f} %")
    client_pire = df_filt.loc[df_filt['proba_churn'].idxmax()] if not df_filt.empty else None
    kc3.metric("Client le plus à risque", f"{client_pire['customer_id']}" if client_pire is not None else "N/A")
    
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fig_hist = px.histogram(df_filt, x='proba_churn', nbins=30, color_discrete_sequence=['#004B87'],
                                title="Distribution des Probabilités de Churn")
        fig_hist.add_vline(x=0.5, line_dash="dash", line_color="red")
        fig_hist.update_layout(**layout_transparent)
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with r1c2:
        var_cat = st.selectbox("Analyser le taux de risque par :", ['contract_type', 'payment_method', 'survey_response', 'signup_channel'])
        cat_data = df_filt.groupby(var_cat)['proba_churn'].mean().reset_index()
        fig_cat = px.bar(cat_data, x=var_cat, y='proba_churn', color='proba_churn', color_continuous_scale='Reds',
                         title=f"Probabilité Moyenne par {var_cat}")
        fig_cat.update_layout(**layout_transparent)
        st.plotly_chart(fig_cat, use_container_width=True)
        
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        fig_scatter = px.scatter(df_filt, x='engagement_score', y='proba_churn', color='customer_segment',
                                 size='monthly_fee', hover_data=['customer_id'],
                                 title="Engagement vs Risque (Taille = MRR)")
        fig_scatter.update_layout(**layout_transparent)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with r2c2:
        heatmap_data = pd.crosstab(df_filt['contract_type'], df_filt['customer_segment'], 
                                   values=df_filt['proba_churn'], aggfunc='mean').fillna(0)
        fig_heat = px.imshow(heatmap_data, text_auto=".2f", color_continuous_scale='Reds', aspect="auto",
                             title="Heatmap du Risque Moyen (Contrat x Segment)")
        fig_heat.update_layout(**layout_transparent)
        st.plotly_chart(fig_heat, use_container_width=True)

# =====================================================================
# PAGE 3 : IMPACT FINANCIER (FINANCE)
# =====================================================================
elif "Page 3" in page:
    st.title("💰 Impact Financier : Évaluation des Pertes")
    
    # Calculs Financiers
    loss_mensuel = df['expected_loss_mensuel'].sum()
    loss_total = df['expected_loss_total'].sum()
    mrr_secu = df[df['proba_churn'] < 0.4]['monthly_fee'].sum()
    mrr_exp = df[df['proba_churn'] >= 0.4]['monthly_fee'].sum()
    pire_client = df.loc[df['expected_loss_mensuel'].idxmax()]
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(f'<div class="kpi-container"><div class="kpi-title">Expected Loss Mensuel</div><div class="kpi-val red-text">{loss_mensuel:,.0f} €</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-container"><div class="kpi-title">Expected Loss Total</div><div class="kpi-val red-text">{loss_total:,.0f} €</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-container"><div class="kpi-title">MRR Sécurisé</div><div class="kpi-val green-text">{mrr_secu:,.0f} €</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-container"><div class="kpi-title">MRR Exposé</div><div class="kpi-val orange-text">{mrr_exp:,.0f} €</div></div>', unsafe_allow_html=True)
    c5.markdown(f'<div class="kpi-container"><div class="kpi-title">Max Perte Potentielle</div><div class="kpi-val red-text">{pire_client["expected_loss_mensuel"]:,.0f} €</div><div style="font-size:10px;">{pire_client["customer_id"]}</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    r1c1, r1c2 = st.columns([1, 1.5])
    with r1c1:
        mrr_crit = df[df['proba_churn'] >= 0.75]['monthly_fee'].sum()
        mrr_mod = df[(df['proba_churn'] >= 0.4) & (df['proba_churn'] < 0.75)]['monthly_fee'].sum()
        
        fig_water = go.Figure(go.Waterfall(
            orientation="v", measure=["absolute", "relative", "relative", "relative"],
            x=["MRR Total", "Sécurisé", "Modéré", "Critique"],
            y=[mrr_secu + mrr_mod + mrr_crit, -mrr_secu, -mrr_mod, -mrr_crit],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": ["#111", "#5CB85C", "#F0AD4E", "#D9534F"]}}
        ))
        fig_water.update_layout(title="Décomposition du MRR Total", **layout_transparent)
        st.plotly_chart(fig_water, use_container_width=True)
        
    with r1c2:
        top_10 = df.nlargest(10, 'expected_loss_mensuel').sort_values('expected_loss_mensuel', ascending=True)
        fig_barh = px.bar(top_10, x='expected_loss_mensuel', y='customer_id', orientation='h',
                          title="Top 10 Clients - Plus Fort Revenu à Risque", text_auto=".0f", color_discrete_sequence=['#D9534F'])
        fig_barh.update_layout(**layout_transparent)
        st.plotly_chart(fig_barh, use_container_width=True)
        
    st.markdown("#### Matrice Valeur vs Risque")
    fig_scat2 = px.scatter(df, x='total_revenue', y='proba_churn', size='monthly_fee', color='contract_type',
                           hover_data=['customer_id'], title="Quadrant Premium (Haut-Droite = Urgence Absolue)")
    fig_scat2.add_hline(y=0.75, line_dash="dash", line_color="red")
    fig_scat2.add_vline(x=df['total_revenue'].quantile(0.8), line_dash="dash", line_color="orange")
    fig_scat2.update_layout(**layout_transparent)
    st.plotly_chart(fig_scat2, use_container_width=True)

# =====================================================================
# PAGE 4 : SIMULATEUR CLIENT (CRM)
# =====================================================================
elif "Page 4" in page:
    st.title("⚙️ Simulateur Client : Plan d'Action CRM")
    
    c_id = st.selectbox("Sélectionnez un client à analyser :", df['customer_id'].unique())
    cust_data = df[df['customer_id'] == c_id].iloc[0].to_dict()
    
    st.markdown("### 1. Profil Actuel")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Segment", cust_data['customer_segment'])
    p2.metric("Contrat", cust_data['contract_type'])
    p3.metric("MRR", f"{cust_data['monthly_fee']} €")
    p4.metric("Tickets Support", cust_data['support_tickets'])
    
    st.markdown("---")
    st.markdown("### 2. Paramètres de Simulation")
    
    col_sim1, col_sim2, col_sim3 = st.columns(3)
    sim_logins = col_sim1.slider("Connexions Mensuelles", 0, 50, int(cust_data['monthly_logins']))
    sim_days = col_sim1.slider("Jours depuis dernière connexion", 0, 30, int(cust_data['last_login_days_ago']))
    
    sim_tickets = col_sim2.slider("Tickets Support Ouverts", 0, 10, int(cust_data['support_tickets']))
    sim_csat = col_sim2.slider("Note Satisfaction (CSAT)", 1.0, 5.0, float(cust_data['csat_score']) if pd.notna(cust_data['csat_score']) else 3.0, 0.5)
    
    sim_failures = col_sim3.slider("Échecs de Paiement", 0, 5, int(cust_data['payment_failures']))
    sim_contract = col_sim3.selectbox("Type de contrat (Simulation)", ["Monthly", "Yearly"], index=0 if cust_data['contract_type'] == "Monthly" else 1)
    
    # Reconstruction de la donnée pour prédiction
    input_dict = cust_data.copy()
    input_dict.update({
        'monthly_logins': sim_logins,
        'last_login_days_ago': sim_days,
        'support_tickets': sim_tickets,
        'csat_score': sim_csat,
        'payment_failures': sim_failures,
        'contract_type': sim_contract
    })
    
    input_df = pd.DataFrame([input_dict]).drop(columns=['churn', 'customer_id', 'proba_churn', 'risk_level', 'expected_loss_mensuel', 'expected_loss_total', 'engagement_score'], errors='ignore')
    
    processed_input = preprocessor.transform(input_df)
    sim_proba = model.predict_proba(processed_input)[0][1]
    act_proba = cust_data['proba_churn']
    delta = sim_proba - act_proba
    
    st.markdown("---")
    st.markdown("### 3. Résultat de la Simulation")
    
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Risque ACTUEL", f"{act_proba*100:.1f} %")
    rc2.metric("Risque SIMULÉ", f"{sim_proba*100:.1f} %", delta=f"{delta*100:.1f} %", delta_color="inverse")
    
    if sim_proba > 0.75:
        action = "🚨 **Appel commercial urgent + Négociation remise immédiate.**"
        color = "#FFCCCC"
    elif sim_proba > 0.40:
        action = "⚠️ **Campagne email ciblée + Audit de satisfaction technique.**"
        color = "#FFF0CC"
    else:
        action = "✅ **Client stabilisé. Maintien dans le flux relationnel standard.**"
        color = "#CCFFCC"
        
    rc3.markdown(f"""
    <div style='background-color: {color}; padding: 15px; border-radius: 8px; border: 1px solid #ccc;'>
        <strong>Plan d'Action Recommandé :</strong><br>{action}
    </div>
    """, unsafe_allow_html=True)