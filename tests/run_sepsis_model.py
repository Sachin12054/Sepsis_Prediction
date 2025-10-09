#!/usr/bin/env python3
"""
Sepsis Model Runner
==================

Direct command-line interface for sepsis prediction.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import argparse

def load_clinical_model():
    """Load the clinical model"""
    try:
        if os.path.exists('models/clinical_sepsis_model.pkl'):
            model = joblib.load('models/clinical_sepsis_model.pkl')
            print(f"✅ Loaded clinical model: {model['model_info']['algorithm']}")
            return model
        else:
            print("❌ Clinical model not found. Please run export_production_models.py first.")
            return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def predict_single_patient(model, features):
    """Predict sepsis risk for a single patient"""
    try:
        features = np.array(features).reshape(1, -1)
        
        # Get prediction and probability
        prediction = model['model'].predict(features)[0]
        probability = model['model'].predict_proba(features)[0, 1]
        
        # Apply clinical threshold
        threshold = model.get('threshold', 0.5)
        clinical_prediction = int(probability >= threshold)
        
        # Calculate risk level
        if probability >= 0.7:
            risk_level = 'HIGH'
            urgency = 'URGENT'
        elif probability >= 0.5:
            risk_level = 'MODERATE'
            urgency = 'MODERATE'
        elif probability >= 0.3:
            risk_level = 'LOW-MODERATE'
            urgency = 'MONITOR'
        else:
            risk_level = 'LOW'
            urgency = 'ROUTINE'
        
        return {
            'prediction': clinical_prediction,
            'probability': probability,
            'risk_level': risk_level,
            'urgency': urgency,
            'threshold': threshold
        }
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None

def predict_from_csv(model, csv_file):
    """Predict sepsis risk from CSV file"""
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        print(f"📄 Loaded {len(df)} patients from {csv_file}")
        
        # Get features (assume all columns are features)
        features = df.values
        
        # Make predictions
        predictions = model['model'].predict(features)
        probabilities = model['model'].predict_proba(features)[:, 1]
        
        # Apply clinical threshold
        threshold = model.get('threshold', 0.5)
        clinical_predictions = (probabilities >= threshold).astype(int)
        
        # Create results dataframe
        results_df = df.copy()
        results_df['sepsis_prediction'] = clinical_predictions
        results_df['sepsis_probability'] = probabilities
        results_df['risk_level'] = ['HIGH' if p >= 0.7 else 'MODERATE' if p >= 0.5 else 'LOW-MODERATE' if p >= 0.3 else 'LOW' for p in probabilities]
        
        # Save results
        output_file = csv_file.replace('.csv', '_predictions.csv')
        results_df.to_csv(output_file, index=False)
        
        # Print summary
        total_patients = len(results_df)
        sepsis_cases = clinical_predictions.sum()
        high_risk = sum([1 for p in probabilities if p >= 0.7])
        
        print(f"\n📊 PREDICTION RESULTS:")
        print(f"   Total patients: {total_patients}")
        print(f"   Sepsis predictions: {sepsis_cases} ({sepsis_cases/total_patients:.1%})")
        print(f"   High risk patients: {high_risk} ({high_risk/total_patients:.1%})")
        print(f"   Results saved to: {output_file}")
        
        return results_df
        
    except Exception as e:
        print(f"❌ Error processing CSV: {e}")
        return None

def test_model():
    """Test model with synthetic data"""
    model = load_clinical_model()
    if model is None:
        return
    
    print("\n🧪 TESTING MODEL WITH SYNTHETIC DATA:")
    print("=" * 50)
    
    # Get expected feature count
    feature_count = len(model.get('feature_names', [])) if 'feature_names' in model else 100
    
    # Create test patients
    np.random.seed(42)
    test_patients = [
        np.random.randn(feature_count),  # Healthy patient
        np.random.randn(feature_count) + 2,  # High-risk patient
        np.random.randn(feature_count) + 1,  # Moderate-risk patient
        np.random.randn(feature_count) - 1,  # Low-risk patient
        np.random.randn(feature_count) + 0.5  # Borderline patient
    ]
    
    patient_types = ['Healthy', 'High-risk', 'Moderate-risk', 'Low-risk', 'Borderline']
    
    for i, (features, patient_type) in enumerate(zip(test_patients, patient_types)):
        result = predict_single_patient(model, features)
        
        if result:
            print(f"\nPatient {i+1} ({patient_type}):")
            print(f"   Sepsis Risk: {'YES' if result['prediction'] else 'NO'}")
            print(f"   Probability: {result['probability']:.1%}")
            print(f"   Risk Level: {result['risk_level']}")
            print(f"   Urgency: {result['urgency']}")
    
    print(f"\n✅ Model testing completed successfully!")
    print(f"📊 Model performance:")
    print(f"   Algorithm: {model['model_info']['algorithm']}")
    print(f"   Sensitivity: {model['performance_metrics']['sensitivity']:.1%}")
    print(f"   Specificity: {model['performance_metrics']['specificity']:.1%}")

def main():
    parser = argparse.ArgumentParser(description='Sepsis Prediction Model Runner')
    parser.add_argument('--test', action='store_true', help='Test model with synthetic data')
    parser.add_argument('--csv', type=str, help='Predict from CSV file')
    parser.add_argument('--features', type=str, help='Comma-separated feature values for single prediction')
    
    args = parser.parse_args()
    
    print("SEPSIS PREDICTION MODEL RUNNER")
    print("=" * 40)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if args.test:
        test_model()
    elif args.csv:
        model = load_clinical_model()
        if model:
            predict_from_csv(model, args.csv)
    elif args.features:
        model = load_clinical_model()
        if model:
            try:
                features = [float(x.strip()) for x in args.features.split(',')]
                result = predict_single_patient(model, features)
                
                if result:
                    print("\n🩺 SINGLE PATIENT PREDICTION:")
                    print(f"   Sepsis Risk: {'YES' if result['prediction'] else 'NO'}")
                    print(f"   Probability: {result['probability']:.1%}")
                    print(f"   Risk Level: {result['risk_level']}")
                    print(f"   Urgency: {result['urgency']}")
                    print(f"   Clinical Threshold: {result['threshold']}")
                    
            except ValueError:
                print("❌ Invalid feature format. Please provide comma-separated numbers.")
    else:
        print("Usage:")
        print("  python run_sepsis_model.py --test")
        print("  python run_sepsis_model.py --csv patient_data.csv")
        print("  python run_sepsis_model.py --features '1.2,3.4,5.6,...'")

if __name__ == '__main__':
    main()
