# Système Intelligent de Rétention Client & Évaluation du Risque de Revenus

> **Projet Data Science — EFREI M1 Data Engineering & AI — 2025-26**  
> Épreuve certifiante RNCP40875 — Bloc 2 : Piloter et implémenter des solutions d'IA

---

## Objectif du projet

Concevoir une plateforme intelligente de rétention client capable d'anticiper le risque de résiliation (churn) et d'évaluer l'impact financier associé, à partir de données comportementales issues d'un environnement business SaaS/télécom.

**Tâche prédictive choisie :** Classification binaire du churn (`churn = 0` ou `churn = 1`)

---

## Dataset

- **Source :** [Kaggle — Customer Churn Prediction Business Dataset](https://www.kaggle.com/datasets/miadul/customer-churn-prediction-business-dataset)
- **Fichier :** `customer_churn_business_dataset.csv`
- **Dimensions :** 10 000 clients × 32 variables
- **Variable cible :** `churn` (0 = client actif, 1 = client résilié)
- **Déséquilibre des classes :** 89.8% No Churn / 10.2% Churn

---

## Structure du projet

```
projet_retention_client/
│
├── data/
│   ├── raw/          ← CSV brut Kaggle (déposé manuellement, non versionné)
│   ├
│
├── notebooks/
│   └── 01_EDA.ipynb   ← Analyse exploratoire complète
│
├── src/
│   ├── config.py            ← Configuration centralisée (chemins, colonnes)
│   ├── preprocessing.py     ← Pipeline de nettoyage et transformation
│   ├── modeling.py          ← Entraînement et sauvegarde des 4 modèles
│   └── evaluation.py        ← Métriques, matrices de confusion, courbes ROC
│
├── app.py               ← Interface décisionnelle Streamlit (4 pages)
│
├── api/
│   └── main.py              ← API REST FastAPI (endpoints /predict et /health)
│
├── models/                  ← Modèles entraînés sauvegardés (non versionnés)
│   ├── preprocessor.joblib
│   ├── Logistic_Regression.joblib
│   ├── Random_Forest.joblib
│   ├── XGBoost.joblib
│   └── Multi_Layer_Perceptron.joblib
│
├── reports/
│   └── figures/             ← Visualisations EDA et évaluation (non versionnées)
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

Description des fichiers
src/config.py
Fichier de configuration centralisé. Contient :

DATA_PATH : chemin vers le dataset brut
TARGET_COL : variable cible (churn)
ID_COL : identifiant client (customer_id)
NUMERICAL_FEATURES : liste des 19 variables numériques
CATEGORICAL_FEATURES : liste des variables catégorielles


src/preprocessing.py
Pipeline de préparation des données scikit-learn. Contient trois fonctions :
load_data(filepath)
Charge le CSV brut depuis data/raw/.
get_preprocessor()
Construit le ColumnTransformer avec deux sous-pipelines :

Variables numériques : SimpleImputer(strategy='median') + RobustScaler
Variables catégorielles : SimpleImputer(strategy='most_frequent') + OneHotEncoder(drop='first', handle_unknown='ignore')


Le fit du pipeline est effectué uniquement sur le train set pour éviter tout data leakage. Le test set reçoit uniquement un transform.

prepare_data(df)
Sépare la matrice de features X du vecteur cible y.

src/modeling.py
Entraînement des 4 modèles requis. Contient deux fonctions :
get_models()
Initialise les 4 modèles avec gestion du déséquilibre de classes :

#Modèle de Gestion déséquilibre
1 LogisticRegression class_weight='balanced' 

2 RandomForestClassifierclass_weight='balanced'

3 XGBClassifierscale_pos_weight=9.0

4MLPClassifier(64, 32)architecture relu + adam

train_and_save_models(X_train, y_train, preprocessor)

Fit du pipeline sur le train set
Entraînement des 4 modèles en boucle
Sauvegarde de chaque modèle en .joblib dans models/
Sauvegarde du preprocessor.joblib (indispensable pour l'API)


src/evaluation.py

Évaluation comparative des modèles. Contient :
evaluate_all_models(models, X_test_processed, y_test)

Pour chaque modèle :

Calcule les 5 métriques : Accuracy, Precision, Recall, F1-Score, ROC-AUC
Génère la matrice de confusion (grille 2×2 sauvegardée)
Trace les courbes ROC superposées
Exporte le tableau comparatif en CSV dans reports/


notebooks/01_EDA.ipynb
Analyse Exploratoire des Données complète en 8 cellules :

Imports et configuration des graphiques
Chargement du dataset + inspection (shape, types, info)
Valeurs manquantes (complaint_type : 2045 NaN) et doublons
Distribution de la cible : déséquilibre 89.8% / 10.2% avec pourcentages
Histogrammes des variables numériques clés
Boxplots : total_revenue, avg_session_time, support_tickets, tenure_months vs churn
Matrice de corrélation + top corrélations avec le churn
Variables catégorielles : taux de churn par contract_type, payment_method, customer_segment, survey_response


main_train.py
Script d'orchestration principal. Lance toute la chaîne en un seul appel :
Chargement → Split stratifié → Preprocessing → Entraînement → Évaluation → Export
bashpython main_train.py

dashboard/app.py
Interface décisionnelle Streamlit orientée utilisateur métier.
0A completer


api/main.py
API REST FastAPI exposant le modèle XGBoost comme service d'inférence :

POST /predict : reçoit les features d'un client en JSON, retourne probabilité de churn + niveau de risque
GET /health : vérifie que le service est actif et que le modèle est chargé


Métriques d'évaluation
Compte tenu du déséquilibre des classes (89.8% / 10.2%) :

Métrique et Priorité Justification

1 Recall⭐ HauteMinimiser les faux négatifs (churners non détectés = perte financière)

2 F1-Score⭐ HauteCompromis Precision / Recall

3 ROC-AUC⭐ HautePerformance indépendante du seuil de décision

4 Accuracy⚠️ BasseTrompeuse avec un déséquilibre 90/10

Choix techniques justifiés
Décision et Justification 
1 RobustScalerRobuste aux valeurs aberrantes sans modifier les données brutes 

2 SimpleImputer(median)Robuste aux outliers contrairement à la moyenne

3 OneHotEncoder(drop='first') Évite la multicolinéarité parfaite, réduit la dimensionnalité

4 class_weight='balanced' Compense le déséquilibre 90/10 sans sur-échantillonnage artificiel

5 scale_pos_weight=9 (XGBoost)Équivalent de class_weight pour XGBoost

6 MLPClassifier sklearn Réseau multicouche (64, 32) relu + adam — Deep Learning valide sans dépendance TensorFlow

7 engagement_score Feature dérivée du sujet combinant logins, jours actifs et CSAT (0-100)

8 expected_loss_mensuel  monthly_fee × proba_churn — estimation métier de la perte attendue

Installation et lancement

1. Cloner le repository
bashgit clone https://github.com/ton-username/projet_retention_client.git
cd projet_retention_client
2. Créer et activer l'environnement virtuel
bashpython -m venv env

# Windows
env\Scripts\activate
# Mac / Linux
source env/bin/activate

3. Installer les dépendances
bashpip install -r requirements.txt
4. Placer le dataset
Télécharger customer_churn_business_dataset.csv depuis Kaggle
→ Le placer dans : data/raw/customer_churn_business_dataset.csv
5. Lancer l'entraînement complet
python main_train.py
Génère tous les .joblib dans models/ et les figures dans reports/.
6. Lancer le dashboard
streamlit run dashboard/app.py
7. Lancer l'API
uvicorn api.main:app --reload

API : http://localhost:8000
Documentation Swagger : http://localhost:8000/docs


Exemple d'appel API
bashcurl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "tenure_months": 5,
           "monthly_fee": 80,
           "payment_failures": 3,
           "csat_score": 2.0,
           "monthly_logins": 5,
           "contract_type": "monthly",
           "customer_segment": "individual"
         }'
         
Réponse attendue :
json{
  "churn_probability": 0.847,
  "churn_prediction": 1,
  "risk_level": "Critique"
}



Auteurs
Rebecca & Flavio

Projet réalisé dans le cadre du module Data Science — EFREI Paris — 2025-26
