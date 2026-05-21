import os

# Chemins de base (relatifs pour faciliter le déploiement)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "customer_churn_business_dataset.csv")

# Colonnes cibles et identifiants
TARGET_COL = "churn"
ID_COL = "customer_id"

# Variables numériques (continues et discrètes)
NUMERICAL_FEATURES = [
    "age", "tenure_months", "monthly_logins", "weekly_active_days",
    "avg_session_time", "features_used", "usage_growth_rate",
    "last_login_days_ago", "monthly_fee", "total_revenue",
    "payment_failures", "support_tickets", "avg_resolution_time",
    "csat_score", "escalations", "email_open_rate", 
    "marketing_click_rate", "nps_score", "referral_count"
]

# Variables catégorielles (nominales et ordinales)
CATEGORICAL_FEATURES = [
    "gender","customer_segment", "signup_channel",
    "contract_type", "payment_method", "discount_applied",
    "price_increase_last_3m", "complaint_type", "survey_response"
]
