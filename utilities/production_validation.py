import os
import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

print("=== PRODUCTION PIPELINE VALIDATION ===")

# Create necessary directories
os.makedirs("production_pipeline", exist_ok=True)
os.makedirs("production_pipeline/models", exist_ok=True)
os.makedirs("production_pipeline/configs", exist_ok=True)
os.makedirs("production_pipeline/documentation", exist_ok=True)

validation_results = {
    'timestamp': datetime.now().isoformat(),
    'validation_status': 'PASS',
    'checks': {}
}

# 1. Test model loading
print("✓ Testing model loading...")
try:
    best_model = joblib.load("ensemble_models/final_best_model.pkl")
    model_features = list(best_model.feature_names_in_)
    validation_results['checks']['model_loading'] = {
        'status': 'PASS',
        'model_type': str(type(best_model)),
        'expected_features': len(model_features),
        'message': 'Model loaded successfully'
    }
    print(f"  ✓ Model loaded: {type(best_model)}")
    print(f"  ✓ Expected features: {len(model_features)}")
except Exception as e:
    validation_results['checks']['model_loading'] = {
        'status': 'FAIL',
        'error': str(e)
    }
    validation_results['validation_status'] = 'FAIL'
    print(f"  ❌ Model loading failed: {e}")

# 2. Test data preprocessing
print("\n✓ Testing data preprocessing...")
try:
    # Load test data
    X_test_full = pd.read_csv("data/stft_features/test_stft_scaled.csv")
    X_test_model = X_test_full[model_features]
    
    validation_results['checks']['data_preprocessing'] = {
        'status': 'PASS',
        'available_features': X_test_full.shape[1],
        'model_features': X_test_model.shape[1],
        'test_samples': X_test_model.shape[0],
        'message': 'Data preprocessing successful'
    }
    print(f"  ✓ Data shape: {X_test_model.shape}")
    print(f"  ✓ Features aligned: {X_test_model.shape[1]} == {len(model_features)}")
    
except Exception as e:
    validation_results['checks']['data_preprocessing'] = {
        'status': 'FAIL',
        'error': str(e)
    }
    validation_results['validation_status'] = 'FAIL'
    print(f"  ❌ Data preprocessing failed: {e}")

# 3. Test prediction pipeline
print("\n✓ Testing prediction pipeline...")
try:
    # Make predictions
    predictions = best_model.predict(X_test_model)
    probabilities = best_model.predict_proba(X_test_model)[:, 1]
    
    validation_results['checks']['prediction_pipeline'] = {
        'status': 'PASS',
        'predictions_count': len(predictions),
        'sepsis_predictions': int(predictions.sum()),
        'avg_probability': float(probabilities.mean()),
        'max_probability': float(probabilities.max()),
        'min_probability': float(probabilities.min()),
        'message': 'Prediction pipeline working'
    }
    print(f"  ✓ Predictions generated: {len(predictions)}")
    print(f"  ✓ Sepsis predictions: {predictions.sum()}")
    print(f"  ✓ Avg probability: {probabilities.mean():.4f}")
    
except Exception as e:
    validation_results['checks']['prediction_pipeline'] = {
        'status': 'FAIL',
        'error': str(e)
    }
    validation_results['validation_status'] = 'FAIL'
    print(f"  ❌ Prediction pipeline failed: {e}")

# 4. Test performance validation
print("\n✓ Testing performance validation...")
try:
    y_test = np.load('data/processed/y_test_stft.npy', allow_pickle=True)
    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
    
    auc = roc_auc_score(y_test, probabilities)
    accuracy = accuracy_score(y_test, predictions)
    
    validation_results['checks']['performance_validation'] = {
        'status': 'PASS' if auc >= 0.8 else 'WARN',
        'roc_auc': float(auc),
        'accuracy': float(accuracy),
        'performance_threshold': 0.8,
        'message': f'Model performance: AUC={auc:.4f}, Accuracy={accuracy:.4f}'
    }
    
    if auc >= 0.9:
        performance_status = "EXCELLENT"
    elif auc >= 0.8:
        performance_status = "GOOD"
    else:
        performance_status = "NEEDS_IMPROVEMENT"
        
    print(f"  ✓ ROC-AUC: {auc:.4f} ({performance_status})")
    print(f"  ✓ Accuracy: {accuracy:.4f}")
    
except Exception as e:
    validation_results['checks']['performance_validation'] = {
        'status': 'FAIL',
        'error': str(e)
    }
    validation_results['validation_status'] = 'FAIL'
    print(f"  ❌ Performance validation failed: {e}")

# 5. Copy production model
print("\n✓ Setting up production model...")
try:
    import shutil
    shutil.copy("ensemble_models/final_best_model.pkl", "production_pipeline/models/sepsis_model.pkl")
    
    # Save model metadata
    model_metadata = {
        'model_type': str(type(best_model)),
        'features': model_features,
        'performance': validation_results['checks'].get('performance_validation', {}),
        'deployment_date': datetime.now().isoformat(),
        'version': '1.0.0'
    }
    
    with open('production_pipeline/models/model_metadata.json', 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    validation_results['checks']['model_deployment'] = {
        'status': 'PASS',
        'model_path': 'production_pipeline/models/sepsis_model.pkl',
        'metadata_path': 'production_pipeline/models/model_metadata.json',
        'message': 'Production model deployed successfully'
    }
    print(f"  ✓ Model copied to production_pipeline/models/")
    print(f"  ✓ Metadata saved")
    
except Exception as e:
    validation_results['checks']['model_deployment'] = {
        'status': 'FAIL',
        'error': str(e)
    }
    validation_results['validation_status'] = 'FAIL'
    print(f"  ❌ Model deployment failed: {e}")

# Save validation results
with open('production_pipeline/validation_results.json', 'w') as f:
    json.dump(validation_results, f, indent=2)

# Create simple API stub
api_code = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List
import json

app = FastAPI(title="Sepsis Prediction API", version="1.0.0")

# Load model and metadata
model = joblib.load("models/sepsis_model.pkl")
with open("models/model_metadata.json", "r") as f:
    model_metadata = json.load(f)

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    sepsis_probability: float
    sepsis_prediction: int
    risk_level: str
    model_version: str

@app.post("/predict", response_model=PredictionResponse)
async def predict_sepsis(request: PredictionRequest):
    try:
        # Validate input
        if len(request.features) != len(model_metadata["features"]):
            raise HTTPException(status_code=400, 
                              detail=f"Expected {len(model_metadata['features'])} features, got {len(request.features)}")
        
        # Create DataFrame with proper feature names
        features_df = pd.DataFrame([request.features], columns=model_metadata["features"])
        
        # Make prediction
        probability = model.predict_proba(features_df)[0, 1]
        prediction = int(model.predict(features_df)[0])
        
        # Determine risk level
        if probability >= 0.8:
            risk_level = "HIGH"
        elif probability >= 0.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return PredictionResponse(
            sepsis_probability=float(probability),
            sepsis_prediction=prediction,
            risk_level=risk_level,
            model_version=model_metadata["version"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_version": model_metadata["version"]}

@app.get("/model/info")
async def model_info():
    return model_metadata

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

with open('production_pipeline/api.py', 'w') as f:
    f.write(api_code)

print(f"\n=== PRODUCTION PIPELINE VALIDATION COMPLETE ===")
print(f"Overall Status: {validation_results['validation_status']}")
print(f"Results saved to: production_pipeline/validation_results.json")
print(f"API created at: production_pipeline/api.py")

if validation_results['validation_status'] == 'PASS':
    print("🎉 Production pipeline is ready for deployment!")
else:
    print("⚠️  Some issues found. Please review validation results.")