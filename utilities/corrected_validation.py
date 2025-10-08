import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# Load feature selector
feature_selector = joblib.load("models/feature_selector.pkl")
print(f"Feature selector type: {type(feature_selector)}")

# Load data
X_train = pd.read_csv("data/stft_features/train_stft_scaled.csv")
X_test = pd.read_csv("data/stft_features/test_stft_scaled.csv")
y_train = np.load('data/processed/y_train_stft.npy', allow_pickle=True)
y_test = np.load('data/processed/y_test_stft.npy', allow_pickle=True)

print(f"Original data shapes - Train: {X_train.shape}, Test: {X_test.shape}")

# Transform data using feature selector
X_train_selected = feature_selector.transform(X_train)
X_test_selected = feature_selector.transform(X_test)

print(f"Selected features shapes - Train: {X_train_selected.shape}, Test: {X_test_selected.shape}")

# Get selected feature names
selected_features = X_train.columns[feature_selector.get_support()]
print(f"Selected {len(selected_features)} features:")
print(list(selected_features))

# Convert to DataFrame with proper column names
X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features)
X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features)

# Now test the final best model
best_model = joblib.load("ensemble_models/final_best_model.pkl")

print("\n=== Testing Final Best Model ===")
try:
    y_pred = best_model.predict(X_test_selected_df)
    y_pred_proba = best_model.predict_proba(X_test_selected_df)[:, 1]
    
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC: {auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Calculate clinical metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"\nClinical Metrics:")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Positive Predictive Value: {ppv:.4f}")
    print(f"Negative Predictive Value: {npv:.4f}")
    
    # Save results
    results = {
        'model': 'final_best_model_corrected',
        'test_auc': float(auc),
        'test_sensitivity': float(sensitivity),
        'test_specificity': float(specificity),
        'test_ppv': float(ppv),
        'test_npv': float(npv),
        'selected_features': list(selected_features),
        'confusion_matrix': cm.tolist(),
        'predictions': y_pred.tolist(),
        'probabilities': y_pred_proba.tolist(),
        'true_labels': y_test.tolist()
    }
    
    import json
    with open('results/validation/final_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nValidation completed successfully!")
    print(f"Results saved to results/validation/final_validation_results.json")
    
except Exception as e:
    print(f"Error: {e}")

# Test ensemble voting model too
print("\n=== Testing Ensemble Voting Model ===")
try:
    voting_model = joblib.load("ensemble_models/ensemble_votingsoft.pkl")
    y_pred_voting = voting_model.predict(X_test_selected_df)
    y_pred_proba_voting = voting_model.predict_proba(X_test_selected_df)[:, 1]
    
    auc_voting = roc_auc_score(y_test, y_pred_proba_voting)
    print(f"Voting Ensemble ROC-AUC: {auc_voting:.4f}")
    
except Exception as e:
    print(f"Error with voting model: {e}")