#!/usr/bin/env python3
"""
🎯 Direct ICU Patient Analysis Demo
=================================
Demonstrates the complete ICU patient data processing without the web interface
"""

from icu_data_converter import ICUDataConverter
import joblib
import numpy as np
import os

def demo_icu_patient_analysis():
    """Complete demonstration of ICU patient sepsis analysis"""
    
    print("🏥 ICU PATIENT SEPSIS ANALYSIS DEMO")
    print("=" * 60)
    
    # Initialize converter
    print("🔧 Initializing ICU data converter...")
    converter = ICUDataConverter()
    
    # Process patient p000001
    patient_file = 'data/raw/training_setA (1)/p000001.psv'
    print(f"\n📋 Processing patient: {os.path.basename(patient_file)}")
    
    result = converter.process_icu_patient_file(patient_file)
    
    if 'features' not in result:
        print("❌ Failed to process patient data")
        return
    
    # Load models and scaler
    print("\n🤖 Loading AI models...")
    models = {}
    model_files = {
        'Best Ensemble': 'ensemble_models/best_ensemble_model.pkl',
        'Stacking Model': 'ensemble_models/ensemble_stacking.pkl',
        'Voting Soft': 'ensemble_models/ensemble_voting_soft.pkl',
        'AdaBoost': 'ensemble_models/ensemble_adaboost.pkl'
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
            print(f"   ✅ {name}")
    
    # Load scaler
    scaler = joblib.load('models/advanced/stft_scaler.pkl')
    print("   ✅ Feature Scaler")
    
    # Prepare features
    print(f"\n🔬 Preparing {len(result['features'])} STFT features...")
    feature_values = list(result['features'].values())
    features = np.array(feature_values).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    # Run predictions
    print("\n🎯 Running sepsis risk analysis...")
    predictions = {}
    
    for name, model in models.items():
        try:
            pred_proba = model.predict_proba(features_scaled)[0]
            sepsis_prob = pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]
            predictions[name] = float(sepsis_prob)
            risk_indicator = "🚨 HIGH" if sepsis_prob >= 0.5 else "✅ LOW"
            print(f"   {name:15}: {sepsis_prob:.3f} ({sepsis_prob*100:5.1f}%) - {risk_indicator}")
        except Exception as e:
            print(f"   ❌ {name}: {e}")
    
    # Clinical assessment
    print("\n" + "=" * 60)
    print("📊 CLINICAL SEPSIS RISK ASSESSMENT")
    print("=" * 60)
    
    # Patient information
    patient_info = result['patient_info']
    print(f"👤 Patient Demographics:")
    print(f"   Age: {patient_info.get('age', 'Unknown')} years")
    print(f"   Gender: {'Female' if patient_info.get('gender') == 0 else 'Male'}")
    print(f"   ICU Stay: {patient_info.get('icu_length', 'Unknown')} hours")
    print(f"   Historical Sepsis: {'Yes' if patient_info.get('sepsis_detected') else 'No'}")
    
    if predictions:
        # Primary risk assessment
        primary_risk = predictions.get('Best Ensemble', list(predictions.values())[0])
        
        print(f"\n🎯 PRIMARY RISK ASSESSMENT:")
        if primary_risk >= 0.7:
            risk_level = "🚨 CRITICAL RISK"
            recommendation = "IMMEDIATE medical intervention required"
        elif primary_risk >= 0.5:
            risk_level = "⚠️  HIGH RISK"
            recommendation = "Urgent clinical assessment recommended"
        elif primary_risk >= 0.3:
            risk_level = "🟡 MODERATE RISK"
            recommendation = "Enhanced monitoring advised"
        else:
            risk_level = "✅ LOW RISK"
            recommendation = "Continue standard care protocols"
        
        print(f"   Risk Score: {primary_risk:.3f} ({primary_risk*100:.1f}%)")
        print(f"   Assessment: {risk_level}")
        print(f"   Recommendation: {recommendation}")
        
        # Model consensus
        high_risk_models = sum(1 for pred in predictions.values() if pred >= 0.5)
        total_models = len(predictions)
        consensus = high_risk_models / total_models
        
        print(f"\n📈 MODEL CONSENSUS:")
        print(f"   High-risk predictions: {high_risk_models}/{total_models} models ({consensus*100:.0f}%)")
        
        if consensus >= 0.75:
            print("   🔴 Strong consensus for HIGH RISK")
        elif consensus >= 0.5:
            print("   🟡 Moderate consensus for HIGH RISK")
        elif consensus >= 0.25:
            print("   🟡 Mixed predictions - clinical judgment advised")
        else:
            print("   🟢 Strong consensus for LOW RISK")
        
        print(f"\n📋 DETAILED MODEL BREAKDOWN:")
        for name, pred in predictions.items():
            confidence = "High" if abs(pred - 0.5) > 0.3 else "Medium" if abs(pred - 0.5) > 0.1 else "Low"
            print(f"   {name:15}: {pred:.3f} (Confidence: {confidence})")
    
    print("\n" + "=" * 60)
    print("✅ Analysis completed successfully!")
    print("📱 For interactive analysis, access the web dashboard")
    print("🔗 Dashboard URL: http://localhost:5002")
    print("=" * 60)

if __name__ == "__main__":
    demo_icu_patient_analysis()