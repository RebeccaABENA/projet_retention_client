import os
import joblib
import pandas as pd
import numpy as np

# Modèles requis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

try:
    import config
except ImportError:
    from . import config

def get_models():
    """
    Initialise et retourne les 4 modèles requis par le projet.
    Intègre la gestion du déséquilibre de classe (Class Imbalance).
    """
    models = {
        # 1. Modèle de référence (Baseline)
        "Logistic_Regression": LogisticRegression(
            class_weight='balanced', 
            random_state=42, 
            max_iter=1000
        ),
        
        # 2. Modèle Ensembliste Classique
        "Random_Forest": RandomForestClassifier(
            class_weight='balanced', 
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1
        ),
        
        # 3. Modèle de Boosting Avancé
        # scale_pos_weight = (nb classe 0) / (nb classe 1) -> ~ 9000 / 1000 = 9
        "XGBoost": XGBClassifier(
            scale_pos_weight=9.0, 
            n_estimators=100, 
            learning_rate=0.05, 
            random_state=42, 
            n_jobs=-1
        ),
        
        # 4. Deep Learning (Perceptron Multicouche demandé dans le cadre théorique)
        "Multi_Layer_Perceptron": MLPClassifier(
            hidden_layer_sizes=(64, 32), 
            activation='relu', 
            solver='adam', 
            max_iter=500, 
            random_state=42
        )
    }
    return models

def train_and_save_models(X_train, y_train, preprocessor, output_dir="../models"):
    """
    Entraîne les 4 modèles sur les données pré-traitées et sauvegarde
    les artefacts (pipeline + modèles) pour la future API FastAPI.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Ajuster et transformer les données d'entraînement avec le préprocesseur
    print("Application du pipeline de preprocessing...")
    X_train_res = preprocessor.fit_transform(X_train)
    
    # Sauvegarde du préprocesseur (crucial pour l'API FastAPI plus tard)
    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))
    print("Pipeline de preprocessing sauvegardé avec succès.")
    
    trained_models = {}
    models = get_models()
    
    # 2. Boucle d'entraînement
    for name, model in models.items():
        print(f"Entraînement du modèle : {name}...")
        model.fit(X_train_res, y_train)
        
        # Sauvegarde du modèle entraîné
        model_path = os.path.join(output_dir, f"{name}.joblib")
        joblib.dump(model, model_path)
        print(f"Modèle {name} sauvegardé dans {model_path}")
        
        trained_models[name] = model
        
    return trained_models