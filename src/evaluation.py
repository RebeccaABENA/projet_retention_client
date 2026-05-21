import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Métriques d'évaluation requises par le sujet
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

def evaluate_all_models(models, X_test_processed, y_test, output_dir="../reports"):
    """
    Calcule les métriques pour chaque modèle, génère un tableau comparatif,
    et trace les courbes ROC ainsi que les matrices de confusion.
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics_list = []
    
    # Configuration de la figure pour regrouper les matrices de confusion
    num_models = len(models)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    plt.figure(figsize=(10, 8)) # Pour la courbe ROC globale
    
    for idx, (name, model) in enumerate(models.items()):
        # 1. Prédictions de classes et probabilités
        y_pred = model.predict(X_test_processed)
        y_prob = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, "predict_proba") else None
        
        # 2. Calcul des métriques obligatoires du sujet
        metrics = {
            "Modèle": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
        }
        metrics_list.append(metrics)
        
        # 3. Tracé de la matrice de confusion pour ce modèle
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx], cbar=False)
        axes[idx].set_title(f"Matrice de Confusion : {name}")
        axes[idx].set_xlabel("Prédictions")
        axes[idx].set_ylabel("Réalité")
        
        # 4. Ajout à la courbe ROC collective
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {metrics['ROC-AUC']:.3f})")
            
    # Finalisation et sauvegarde des matrices de confusion
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "matrices_confusion.png"))
    plt.close(fig)
    
    # Finalisation et sauvegarde de la courbe ROC globale
    plt.plot([0, 1], [0, 1], 'k--', label="Hasard (AUC = 0.500)")
    plt.xlabel("Taux de Faux Positifs (FPR)")
    plt.ylabel("Taux de Vrais Positifs (TPR)")
    plt.title("Comparaison des Courbes ROC")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "courbes_roc.png"))
    plt.close()
    
    # 5. Création du DataFrame comparatif
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv(os.path.join(output_dir, "comparatif_modeles.csv"), index=False)
    
    print(f"\nÉvaluation terminée. Les graphiques et le tableau ont été sauvegardés dans : {output_dir}")
    return df_metrics