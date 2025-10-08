import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load data
X_train = pd.read_csv("data/stft_features/train_stft_scaled.csv")
X_test = pd.read_csv("data/stft_features/test_stft_scaled.csv")
y_train = np.load('data/processed/y_train_stft.npy', allow_pickle=True)
y_test = np.load('data/processed/y_test_stft.npy', allow_pickle=True)

print(f"Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train labels - Sepsis: {y_train.sum()}, No Sepsis: {len(y_train) - y_train.sum()}")
print(f"Test labels - Sepsis: {y_test.sum()}, No Sepsis: {len(y_test) - y_test.sum()}")

# Load best ensemble model
best_model = joblib.load("ensemble_models/final_best_model.pkl")

# Make predictions
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Calculate metrics
print("\n=== FINAL MODEL PERFORMANCE ===")
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Calculate additional clinical metrics
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print(f"\nClinical Metrics:")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Positive Predictive Value (Precision): {ppv:.4f}")
print(f"Negative Predictive Value: {npv:.4f}")

# Test other ensemble models for comparison
ensemble_models = {
    'VotingSoft': 'ensemble_models/ensemble_votingsoft.pkl',
    'XGBoost': 'ensemble_models/base_model_xgboost.pkl',
    'LightGBM': 'ensemble_models/base_model_lightgbm.pkl'
}

print("\n=== ENSEMBLE MODEL COMPARISON ===")
model_scores = {}

for name, path in ensemble_models.items():
    try:
        model = joblib.load(path)
        pred = model.predict(X_test)
        pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, pred_proba)
        model_scores[name] = auc
        print(f"{name}: ROC-AUC = {auc:.4f}")
    except Exception as e:
        print(f"Error evaluating {name}: {e}")

# Save results
results = {
    'model': 'final_best_model',
    'test_auc': roc_auc_score(y_test, y_pred_proba),
    'test_sensitivity': sensitivity,
    'test_specificity': specificity,
    'test_ppv': ppv,
    'test_npv': npv,
    'predictions': y_pred.tolist(),
    'probabilities': y_pred_proba.tolist(),
    'true_labels': y_test.tolist()
}

import json
with open('results/validation/final_validation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nValidation completed! Results saved to results/validation/final_validation_results.json")
print(f"Best model achieved ROC-AUC of {roc_auc_score(y_test, y_pred_proba):.4f} on test set")