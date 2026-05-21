# À exécuter pour lancer toute la chaîne (Pipeline -> Entraînement -> Évaluation)
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath("./src"))
from preprocessing import load_data, get_preprocessor, prepare_data
from modeling import train_and_save_models
from evaluation import evaluate_all_models

# 1. Chargement et split des données
print("Chargement des données...")
df = load_data("data/raw/customer_churn_business_dataset.csv")
X, y = prepare_data(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Récupération du pipeline et entraînement de la chaîne
preprocessor = get_preprocessor()
trained_models = train_and_save_models(X_train, y_train, preprocessor, output_dir="models")

# 3. Transformation des données de test et évaluation globale
X_test_processed = preprocessor.transform(X_test)
df_comparatif = evaluate_all_models(trained_models, X_test_processed, y_test, output_dir="reports")

# 4. Affichage du tableau comparatif final requis
print("\n=== TABLEAU COMPARATIF DES MODÈLES ===")
print(df_comparatif.to_string(index=False))