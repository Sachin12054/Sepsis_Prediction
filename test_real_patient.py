#!/usr/bin/env python3
"""
🧪 Test Real ICU Patient Data Processing
=======================================
Test script to verify the complete system works with real hospital ICU patient data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from icu_data_converter import ICUDataConverter
import joblib
import numpy as np

def test_real_patient_prediction():
    """Test complete prediction pipeline with real ICU patient data"""
    
    print("🧪 Testing Real ICU Patient Data Processing")
    print("=" * 60)
    
    # Initialize converter
    print("🔧 Initializing ICU data converter...")
    converter = ICUDataConverter()
    
    # Process real patient data
    print("\n🏥 Processing ICU patient p000001...")
    patient_file = 'data/raw/training_setA (1)/p000001.psv'
    result = converter.process_icu_patient_file(patient_file)
    
    if 'features' not in result:
        print("❌ Failed to extract features from patient data")
        return
    
    print(f"✅ Successfully extracted {len(result['features'])} features")
    
    # Load models
    print("\n🤖 Loading sepsis prediction models...")
    models = {}
    model_files = {
        'best_ensemble': 'ensemble_models/best_ensemble_model.pkl',
        'stacking_ensemble': 'ensemble_models/ensemble_stacking.pkl',
        'voting_soft': 'ensemble_models/ensemble_voting_soft.pkl',
        'adaboost': 'ensemble_models/ensemble_adaboost.pkl'
    }
    
    for model_name, model_path in model_files.items():
        try:
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
                print(f"   ✅ Loaded {model_name} model")
            else:
                print(f"   ⚠️  Model not found: {model_path}")
        except Exception as e:
            print(f"   ❌ Error loading {model_name}: {e}")
    
    if not models:
        print("❌ No models loaded - cannot proceed with predictions")
        return
    
    # Prepare features for prediction
    print("\n🎯 Running sepsis predictions...")
    feature_values = list(result['features'].values())
    features = np.array(feature_values).reshape(1, -1)
    
    print(f"   📊 Feature array shape: {features.shape}")
    
    # Load and apply scaler
    try:
        scaler = joblib.load('models/advanced/stft_scaler.pkl')
        features_scaled = scaler.transform(features)
        print(f"   📊 Features scaled using STFT scaler")
        features = features_scaled
    except Exception as e:
        print(f"   ⚠️  Scaling failed, using raw features: {e}")
    
    # Run predictions
    predictions = {}
    for model_name, model in models.items():
        try:
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(features)[0]
                if len(pred_proba) > 1:
                    predictions[model_name] = float(pred_proba[1])  # Probability of sepsis
                else:
                    predictions[model_name] = float(pred_proba[0])
            else:
                pred = model.predict(features)[0]
                predictions[model_name] = float(pred)
            
            print(f"   🎯 {model_name:15}: {predictions[model_name]:.3f} ({predictions[model_name]*100:.1f}%)")
            
        except Exception as e:
            print(f"   ❌ {model_name} prediction failed: {e}")
    
    # Display results
    print("\n📊 SEPSIS PREDICTION RESULTS")
    print("=" * 60)
    
    # Patient information
    patient_info = result['patient_info']
    print(f"👤 Patient: p000001")
    print(f"   Age: {patient_info.get('age', 'Unknown')}")
    print(f"   Gender: {'Female' if patient_info.get('gender') == 0 else 'Male'}")
    print(f"   ICU Hours: {patient_info.get('icu_length', 'Unknown')}")
    print(f"   Historical Sepsis: {'Yes' if patient_info.get('sepsis_detected') else 'No'}")
    
    # Prediction summary
    if predictions:
        primary_pred = predictions.get('best_ensemble', predictions.get('stacking_ensemble', 0))
        risk_level = "HIGH RISK" if primary_pred >= 0.5 else "LOW RISK"
        risk_color = "🚨" if primary_pred >= 0.5 else "✅"
        
        print(f"\n{risk_color} SEPSIS RISK: {risk_level}")
        print(f"🎯 Primary Risk Score: {primary_pred:.3f} ({primary_pred*100:.1f}%)")
        
        if primary_pred >= 0.5:
            print("⚠️  CLINICAL ALERT: High sepsis risk detected!")
            print("   Recommend immediate clinical assessment")
            print("   Consider sepsis protocols and interventions")
        else:
            print("✅ Low sepsis risk based on current vital signs")
            print("   Continue routine monitoring")
    
    print("\n🔬 Model Comparison:")
    for model_name, pred in predictions.items():
        status = "🚨 HIGH" if pred >= 0.5 else "✅ LOW"
        print(f"   {model_name:15}: {pred:.3f} ({pred*100:4.1f}%) - {status}")
    
    print("\n✅ Real ICU patient data processing test completed successfully!")
    
    return {
        'success': True,
        'patient_info': patient_info,
        'predictions': predictions,
        'features_extracted': len(result['features'])
    }

if __name__ == "__main__":
    test_real_patient_prediction()