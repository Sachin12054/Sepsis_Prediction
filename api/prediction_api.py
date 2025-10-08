#!/usr/bin/env python3
"""
Sepsis Prediction API
====================

Simple API for sepsis prediction using exported production models.
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from datetime import datetime
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
clinical_model = None
ensemble_models = None
validation_framework = None

def load_models():
    """Load exported models"""
    global clinical_model, ensemble_models, validation_framework
    
    try:
        if os.path.exists('models/clinical_sepsis_model.pkl'):
            clinical_model = joblib.load('models/clinical_sepsis_model.pkl')
            logger.info("Clinical model loaded")
        
        if os.path.exists('models/production/ensemble_models.pkl'):
            ensemble_models = joblib.load('models/production/ensemble_models.pkl')
            logger.info(f"{len(ensemble_models)} ensemble models loaded")
        
        if os.path.exists('models/production/validation_framework.pkl'):
            validation_framework = joblib.load('models/production/validation_framework.pkl')
            logger.info("Validation framework loaded")
            
        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': clinical_model is not None,
        'ensemble_models': len(ensemble_models) if ensemble_models else 0,
        'version': '1.0.0'
    })

@app.route('/api/predict', methods=['POST'])
def predict_sepsis():
    """Predict sepsis risk for a patient"""
    try:
        if clinical_model is None:
            return jsonify({'error': 'Models not loaded'}), 503
        
        data = request.json
        
        if 'features' not in data:
            return jsonify({'error': 'Features not provided'}), 400
        
        features = np.array(data['features']).reshape(1, -1)
        
        # Make prediction with clinical model
        prediction = clinical_model['model'].predict(features)[0]
        probability = clinical_model['model'].predict_proba(features)[0, 1]
        
        # Apply clinical threshold
        threshold = clinical_model.get('threshold', 0.5)
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
        
        # Ensemble prediction if available
        ensemble_predictions = []
        if ensemble_models:
            for model_name, model_data in ensemble_models.items():
                try:
                    model = model_data['model']
                    prob = model.predict_proba(features)[0, 1]
                    ensemble_predictions.append(prob)
                except:
                    pass
        
        ensemble_probability = np.mean(ensemble_predictions) if ensemble_predictions else probability
        ensemble_uncertainty = np.std(ensemble_predictions) if len(ensemble_predictions) > 1 else 0.0
        
        result = {
            'patient_id': data.get('patient_id', 'unknown'),
            'prediction': {
                'sepsis_risk': int(clinical_prediction),
                'probability': float(probability),
                'ensemble_probability': float(ensemble_probability),
                'uncertainty': float(ensemble_uncertainty),
                'confidence': float(1 - ensemble_uncertainty)
            },
            'clinical_assessment': {
                'risk_level': risk_level,
                'urgency': urgency,
                'threshold_used': threshold,
                'recommendation': get_clinical_recommendation(risk_level, ensemble_uncertainty)
            },
            'model_info': {
                'algorithm': clinical_model['model_info']['algorithm'],
                'version': clinical_model['model_info']['version'],
                'sensitivity': clinical_model['performance_metrics']['sensitivity'],
                'specificity': clinical_model['performance_metrics']['specificity']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction made: {result['prediction']['probability']:.3f} -> {risk_level}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

def get_clinical_recommendation(risk_level, uncertainty):
    """Get clinical recommendation based on risk level and uncertainty"""
    
    if uncertainty > 0.2:
        return "High uncertainty detected. Consider additional clinical assessment."
    
    recommendations = {
        'HIGH': 'Immediate clinical intervention recommended. Monitor vital signs closely.',
        'MODERATE': 'Close monitoring recommended. Consider laboratory tests and clinical assessment.',
        'LOW-MODERATE': 'Continue standard monitoring. Reassess if condition changes.',
        'LOW': 'Continue routine care. Monitor as per standard protocols.'
    }
    
    return recommendations.get(risk_level, 'Clinical assessment recommended.')

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction for multiple patients"""
    try:
        if clinical_model is None:
            return jsonify({'error': 'Models not loaded'}), 503
        
        data = request.json
        
        if 'patients' not in data:
            return jsonify({'error': 'Patient data not provided'}), 400
        
        results = []
        
        for i, patient_data in enumerate(data['patients']):
            if 'features' not in patient_data:
                continue
                
            features = np.array(patient_data['features']).reshape(1, -1)
            
            # Make prediction
            probability = clinical_model['model'].predict_proba(features)[0, 1]
            prediction = int(probability >= clinical_model.get('threshold', 0.5))
            
            # Calculate risk level
            if probability >= 0.7:
                risk_level = 'HIGH'
            elif probability >= 0.5:
                risk_level = 'MODERATE'
            elif probability >= 0.3:
                risk_level = 'LOW-MODERATE'
            else:
                risk_level = 'LOW'
            
            results.append({
                'patient_id': patient_data.get('patient_id', f'patient_{i}'),
                'sepsis_risk': prediction,
                'probability': float(probability),
                'risk_level': risk_level,
                'timestamp': datetime.now().isoformat()
            })
        
        return jsonify({
            'results': results,
            'total_patients': len(results),
            'high_risk_count': len([r for r in results if r['risk_level'] == 'HIGH']),
            'batch_timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    if clinical_model is None:
        return jsonify({'error': 'Models not loaded'}), 503
    
    info = {
        'model_info': clinical_model['model_info'],
        'performance_metrics': clinical_model['performance_metrics'],
        'clinical_validation': clinical_model['clinical_validation'],
        'threshold': clinical_model.get('threshold', 0.5),
        'ensemble_models': len(ensemble_models) if ensemble_models else 0,
        'feature_count': len(validation_framework['feature_names']) if validation_framework else 'unknown'
    }
    
    return jsonify(info)

if __name__ == '__main__':
    print("🚀 Starting Sepsis Prediction API...")
    
    if load_models():
        print("✅ Models loaded successfully")
        print(f"🌐 API server starting on http://localhost:5000")
        print(f"📊 Endpoints: /api/health, /api/predict, /api/batch_predict, /api/model_info")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("❌ Failed to load models. Please run export_production_models.py first.")
