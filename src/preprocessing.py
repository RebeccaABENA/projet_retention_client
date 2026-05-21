import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Import de la configuration locale
try:
    import config
except ImportError:
    from . import config

def load_data(filepath=None):
    """Charge les données brutes. Peut être adapté pour lire depuis une BDD."""
    if filepath is None:
        filepath = config.DATA_PATH
    return pd.read_csv(filepath)

def get_preprocessor():
    """
    Crée et retourne le pipeline de preprocessing scikit-learn.
    Assure la reproductibilité et prévient le data leakage.
    """
    # 1. Pipeline pour les variables numériques
    # On utilise la médiane pour être robuste aux valeurs aberrantes
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Pipeline pour les variables catégorielles
    # Encodage One-Hot avec drop='first' pour éviter la multicolinéarité parfaite
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    # 3. Combinaison dans le ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, config.NUMERICAL_FEATURES),
            ('cat', categorical_transformer, config.CATEGORICAL_FEATURES)
        ])
    
    return preprocessor

def prepare_data(df):
    """Sépare la matrice de features (X) du vecteur cible (y)."""
    X = df.drop(columns=[config.TARGET_COL, config.ID_COL], errors='ignore')
    y = df[config.TARGET_COL] if config.TARGET_COL in df.columns else None
    return X, y
