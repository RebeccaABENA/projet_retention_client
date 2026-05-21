from pydantic import BaseModel, Field

class CustomerInput(BaseModel):
    # Identifiant unique
    customer_id: str = Field(..., description="Identifiant unique du client (ex: CUST_00001)")
    
    # Informations Démographiques et Profil
    gender: str = Field(..., description="Genre du client (Male, Female)")
    age: int = Field(..., description="Âge du client en années")
    country: str = Field(..., description="Pays de résidence")
    city: str = Field(..., description="Ville de résidence")
    customer_segment: str = Field(..., description="Segment de marché (SME, Individual)")
    
    # Informations Contractuelles et Financières
    tenure_months: int = Field(..., description="Ancienneté du client en mois")
    signup_channel: str = Field(..., description="Canal d'acquisition (Web, Mobile, Referral, etc.)")
    contract_type: str = Field(..., description="Type de contrat (Monthly, Yearly)")
    monthly_fee: float = Field(..., description="Montant de l'abonnement mensuel en euros")
    total_revenue: float = Field(..., description="Total des revenus générés par ce client en euros")
    discount_applied: str = Field(..., description="Application d'une remise commerciale (Yes, No)")
    price_increase_last_3m: str = Field(..., description="Augmentation de tarif subie les 3 derniers mois (Yes, No)")
    payment_method: str = Field(..., description="Mode de paiement (Card, PayPal, Bank Transfer)")
    payment_failures: int = Field(..., description="Nombre d'échecs de paiement enregistrés")
    
    # Données d'Utilisation du Service / Produit
    monthly_logins: int = Field(..., description="Nombre de connexions au service par mois")
    weekly_active_days: int = Field(..., description="Nombre de jours actifs par semaine (0 à 7)")
    avg_session_time: float = Field(..., description="Durée moyenne d'une session en minutes")
    features_used: int = Field(..., description="Nombre de fonctionnalités clés utilisées par le client")
    usage_growth_rate: float = Field(..., description="Taux de croissance ou de baisse de l'utilisation")
    last_login_days_ago: int = Field(..., description="Nombre de jours depuis la dernière connexion")
    
    # Données Support, Satisfaction et Engagement
    support_tickets: int = Field(..., description="Nombre de tickets d'assistance ouverts")
    avg_resolution_time: float = Field(..., description="Temps moyen de résolution des tickets en heures")
    complaint_type: str = Field("No Complaint", description="Type de la dernière plainte déposée (Service, Billing, Technical...)")
    csat_score: float = Field(..., description="Note de satisfaction client - CSAT")
    escalations: int = Field(..., description="Nombre de réclamations escaladées vers un responsable")
    email_open_rate: float = Field(..., description="Taux d'ouverture des emails marketing (0.0 à 1.0)")
    marketing_click_rate: float = Field(..., description="Taux de clic sur les liens marketing (0.0 à 1.0)")
    nps_score: int = Field(..., description="Score de recommandation Net Promoter Score (-100 à 100)")
    survey_response: str = Field(..., description="Sentiment exprimé lors de l'enquête (Satisfied, Neutral, Unsatisfied)")
    referral_count: int = Field(..., description="Nombre de parrainages réussis par le client")

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CUST_00001",
                "gender": "Male",
                "age": 45,
                "country": "France",
                "city": "Paris",
                "customer_segment": "SME",
                "tenure_months": 24,
                "signup_channel": "Web",
                "contract_type": "Monthly",
                "monthly_fee": 49.99,
                "total_revenue": 1199.76,
                "discount_applied": "No",
                "price_increase_last_3m": "No",
                "payment_method": "Card",
                "payment_failures": 0,
                "monthly_logins": 22,
                "weekly_active_days": 5,
                "avg_session_time": 15.4,
                "features_used": 6,
                "usage_growth_rate": 0.12,
                "last_login_days_ago": 2,
                "support_tickets": 1,
                "avg_resolution_time": 2.5,
                "complaint_type": "No Complaint",
                "csat_score": 4.5,
                "escalations": 0,
                "email_open_rate": 0.75,
                "marketing_click_rate": 0.40,
                "nps_score": 50,
                "survey_response": "Satisfied",
                "referral_count": 2
            }
        }