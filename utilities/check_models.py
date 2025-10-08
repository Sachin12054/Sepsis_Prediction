import joblib
import pandas as pd
import numpy as np

# Load models and check their expected features
models_to_check = [
    'ensemble_models/final_best_model.pkl',
    'ensemble_models/base_model_xgboost.pkl',
    'models/xgboost_with_stft_model.pkl'
]

for model_path in models_to_check:
    try:
        model = joblib.load(model_path)
        print(f"\n=== {model_path} ===")
        print(f"Model type: {type(model)}")
        
        if hasattr(model, 'feature_names_in_'):
            print(f"Expected features: {len(model.feature_names_in_)}")
            print(f"First 10 features: {list(model.feature_names_in_[:10])}")
        elif hasattr(model, 'named_steps'):
            print("Pipeline detected")
            for step_name, step in model.named_steps.items():
                print(f"  Step {step_name}: {type(step)}")
                if hasattr(step, 'feature_names_in_'):
                    print(f"    Features: {len(step.feature_names_in_)}")
        else:
            print("No feature names available")
            
    except Exception as e:
        print(f"Error loading {model_path}: {e}")

# Check available data
print("\n=== Available Data ===")
stft_train = pd.read_csv("data/stft_features/train_stft_scaled.csv")
print(f"STFT features: {stft_train.shape[1]}")
print(f"STFT columns: {list(stft_train.columns[:10])}")

# Check what data the models expect by loading base model performance data
try:
    base_perf = pd.read_csv("ensemble_models/base_model_performance.csv")
    print(f"\nBase model performance data: {base_perf}")
except Exception as e:
    print(f"Could not load base performance data: {e}")

# Try the individual STFT model that should work
try:
    stft_model = joblib.load("models/xgboost_with_stft_model.pkl")
    print(f"\nSTFT Model features expected: {stft_model.n_features_in_}")
    
    # Test prediction with STFT model
    X_test = pd.read_csv("data/stft_features/test_stft_scaled.csv")
    pred = stft_model.predict(X_test)
    pred_proba = stft_model.predict_proba(X_test)[:, 1]
    
    y_test = np.load('data/processed/y_test_stft.npy', allow_pickle=True)
    from sklearn.metrics import roc_auc_score, classification_report
    
    auc = roc_auc_score(y_test, pred_proba)
    print(f"STFT Model AUC: {auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, pred))
    
except Exception as e:
    print(f"Error with STFT model: {e}")