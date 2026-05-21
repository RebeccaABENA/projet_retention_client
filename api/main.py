from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import os
from pydantic import BaseModel

# Import des schémas de validation que l'on va créer
try:
    from .schemas import CustomerInput
except ImportError:
    from schemas import CustomerInput

app = FastAPI(
    title="Système Intelligent de Prédiction du Churn Client",
    description="API REST pour évaluer le risque de rétention et de revenus",
    version="1.0"
)

# Variables globales pour stocker les artefacts chargés au démarrage
PREPROCESSOR = None
MODEL = None

@app.on_event("startup")
def load_artifacts():
    """Charge le pipeline de preprocessing et le meilleur modèle au démarrage de l'API."""
    global PREPROCESSOR, MODEL
    
    # Chemins vers les fichiers sauvegardés par main_train.py
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    
    # Ici, remplace par le nom du modèle candidat final choisi (ex: XGBoost.joblib)
    model_path = os.path.join(models_dir, "XGBoost.joblib") 
    
    if not os.path.exists(preprocessor_path) or not os.path.exists(model_path):
        raise RuntimeError("Les artefacts du modèle sont introuvables. Lancez d'abord l'entraînement.")
        
    PREPROCESSOR = joblib.load(preprocessor_path)
    MODEL = joblib.load(model_path)
    print("Pipeline de preprocessing et modèle chargés avec succès !")

@app.get("/health")
def health_check():
    """Endpoint de vérification de l'état de l'API."""
    if PREPROCESSOR is not None and MODEL is not None:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}

@app.post("/predict")
def predict_churn(customer_data: CustomerInput):
    """
    Reçoit les données d'un client, applique le pipeline de preprocessing
    et retourne la prédiction de Churn ainsi que la probabilité associée.
    """
    try:
        # 1. Convertir l'entrée Pydantic en DataFrame Pandas (format attendu par le pipeline)
        input_df = pd.DataFrame([customer_data.dict()])
        
        # 2. Utiliser le pipeline de preprocessing CHARGÉ (uniquement transform, PAS fit_transform)
        processed_data = PREPROCESSOR.transform(input_df)
        
        # 3. Effectuer la prédiction
        prediction = int(MODEL.predict(processed_data)[0])
        probability = float(MODEL.predict_proba(processed_data)[0][1])
        
        # 4. Évaluation du risque de revenus (Règle métier liée au sujet)
        risk_level = "Élevé" if probability > 0.7 else "Modéré" if probability > 0.4 else "Faible"
        revenue_at_risk = customer_data.monthly_fee if prediction == 1 else 0.0
        
        return {
            "customer_id": customer_data.customer_id,
            "churn_prediction": prediction,
            "churn_probability": round(probability, 4),
            "risk_level": risk_level,
            "revenue_at_risk_monthly": revenue_at_risk
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")