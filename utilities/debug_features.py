import joblib
import pandas as pd
import numpy as np

# Load the final best model and check its features
best_model = joblib.load("ensemble_models/final_best_model.pkl")

print(f"Model type: {type(best_model)}")
print(f"Expected features by model: {list(best_model.feature_names_in_)}")
print(f"Number of expected features: {len(best_model.feature_names_in_)}")

# Check the order and exact feature names
model_features = list(best_model.feature_names_in_)
print("\nModel expected features:")
for i, feat in enumerate(model_features):
    print(f"{i+1}: {feat}")

# Now load data and match the exact features
X_train_full = pd.read_csv("data/stft_features/train_stft_scaled.csv")
X_test_full = pd.read_csv("data/stft_features/test_stft_scaled.csv")

print(f"\nAvailable features in data: {X_train_full.shape[1]}")
print("First 10 available features:")
for i, feat in enumerate(X_train_full.columns[:10]):
    print(f"{i+1}: {feat}")

# Check which model features are missing in data
missing_features = [f for f in model_features if f not in X_train_full.columns]
print(f"\nMissing features in data: {len(missing_features)}")
if missing_features:
    print("Missing features:")
    for feat in missing_features[:10]:  # Show first 10
        print(f"  - {feat}")

# Check which data features are not used by model
extra_features = [f for f in X_train_full.columns if f not in model_features]
print(f"\nExtra features in data not used by model: {len(extra_features)}")

# Try to find the intersection
common_features = [f for f in model_features if f in X_train_full.columns]
print(f"\nCommon features: {len(common_features)}")

if len(common_features) == len(model_features):
    print("✅ All model features are available in data!")
    # Test with correct feature order
    X_test_model = X_test_full[model_features]
    y_test = np.load('data/processed/y_test_stft.npy', allow_pickle=True)
    
    y_pred = best_model.predict(X_test_model)
    y_pred_proba = best_model.predict_proba(X_test_model)[:, 1]
    
    from sklearn.metrics import roc_auc_score, classification_report
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n🎯 FINAL MODEL PERFORMANCE:")
    print(f"ROC-AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
else:
    print("❌ Feature mismatch detected")
    print(f"Need {len(model_features)} features, have {len(common_features)} common features")