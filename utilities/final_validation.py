import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import json

# Load the selected features
with open('results/selected_features.txt', 'r') as f:
    selected_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"Selected features: {len(selected_features)}")

# Load data
X_train_full = pd.read_csv("data/stft_features/train_stft_scaled.csv")
X_test_full = pd.read_csv("data/stft_features/test_stft_scaled.csv")
y_train = np.load('data/processed/y_train_stft.npy', allow_pickle=True)
y_test = np.load('data/processed/y_test_stft.npy', allow_pickle=True)

print(f"Original data shapes - Train: {X_train_full.shape}, Test: {X_test_full.shape}")

# Select only the features that were used in training
X_train = X_train_full[selected_features]
X_test = X_test_full[selected_features]

print(f"Selected data shapes - Train: {X_train.shape}, Test: {X_test.shape}")

# Load and test the final best model
best_model = joblib.load("ensemble_models/final_best_model.pkl")

print("\n=== Final Best Model Performance ===")
try:
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
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
    
    # Test other models for comparison
    model_comparison = {}
    
    models_to_test = {
        'Ensemble_VotingSoft': 'ensemble_models/ensemble_votingsoft.pkl',
        'Base_XGBoost': 'ensemble_models/base_model_xgboost.pkl',
        'Base_LightGBM': 'ensemble_models/base_model_lightgbm.pkl',
        'Base_LogisticRegression': 'ensemble_models/base_model_logisticregression.pkl'
    }
    
    print(f"\n=== Model Comparison ===")
    model_comparison['Final_Best_Model'] = {
        'ROC_AUC': float(auc),
        'Sensitivity': float(sensitivity),
        'Specificity': float(specificity),
        'PPV': float(ppv),
        'NPV': float(npv)
    }
    
    for name, path in models_to_test.items():
        try:
            model = joblib.load(path)
            pred = model.predict(X_test)
            pred_proba = model.predict_proba(X_test)[:, 1]
            auc_comp = roc_auc_score(y_test, pred_proba)
            
            cm_comp = confusion_matrix(y_test, pred)
            tn_comp, fp_comp, fn_comp, tp_comp = cm_comp.ravel()
            sens_comp = tp_comp / (tp_comp + fn_comp) if (tp_comp + fn_comp) > 0 else 0
            spec_comp = tn_comp / (tn_comp + fp_comp) if (tn_comp + fp_comp) > 0 else 0
            
            model_comparison[name] = {
                'ROC_AUC': float(auc_comp),
                'Sensitivity': float(sens_comp),
                'Specificity': float(spec_comp)
            }
            
            print(f"{name}: ROC-AUC = {auc_comp:.4f}, Sensitivity = {sens_comp:.4f}, Specificity = {spec_comp:.4f}")
            
        except Exception as e:
            print(f"Error testing {name}: {e}")
    
    # Save comprehensive results
    results = {
        'validation_summary': {
            'best_model': 'final_best_model',
            'test_samples': int(len(y_test)),
            'sepsis_cases': int(y_test.sum()),
            'non_sepsis_cases': int(len(y_test) - y_test.sum()),
            'selected_features_count': len(selected_features)
        },
        'final_best_model_performance': {
            'test_auc': float(auc),
            'test_sensitivity': float(sensitivity),
            'test_specificity': float(specificity),
            'test_ppv': float(ppv),
            'test_npv': float(npv),
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'probabilities': y_pred_proba.tolist(),
            'true_labels': y_test.tolist()
        },
        'model_comparison': model_comparison,
        'selected_features': selected_features
    }
    
    with open('results/validation/comprehensive_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ VALIDATION COMPLETED SUCCESSFULLY!")
    print(f"📊 Final Best Model AUC: {auc:.4f}")
    print(f"📈 Sensitivity: {sensitivity:.4f}")
    print(f"📉 Specificity: {specificity:.4f}")
    print(f"💾 Results saved to results/validation/comprehensive_validation_results.json")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()