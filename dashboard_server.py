#!/usr/bin/env python3
"""
🏥 Sepsis Prediction Dashboard Backend
=====================================
Flask server to handle model predictions for the HTML dashboard
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Add CORS headers manually
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

class SepsisPredictionAPI:
    def __init__(self):
        self.model = None
        self.model_path = "models/clinical_sepsis_model.pkl"
        self.load_model()
    
    def load_model(self):
        """Load the clinical sepsis model"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print(f"✅ Clinical model loaded successfully")
                return True
            else:
                print(f"❌ Model not found: {self.model_path}")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict(self, features):
        """Make sepsis predictions"""
        if self.model is None:
            return None
        
        try:
            # Ensure features is a 2D array
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Get prediction probabilities
            probabilities = self.model['model'].predict_proba(features)[:, 1]
            
            # Apply clinical threshold
            predictions = (probabilities >= self.model['threshold']).astype(int)
            
            return predictions, probabilities
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None, None
    
    def get_model_info(self):
        """Get model information"""
        if self.model is None:
            return None
        
        return {
            'algorithm': self.model['model_info']['algorithm'],
            'threshold': self.model['threshold'],
            'sensitivity': self.model['performance_metrics']['sensitivity'],
            'specificity': self.model['performance_metrics']['specificity'],
            'accuracy': self.model['performance_metrics']['accuracy'],
            'clinical_status': self.model['clinical_validation']['approval_status']
        }

# Initialize the API
api = SepsisPredictionAPI()

@app.route('/')
def dashboard():
    """Serve the dashboard HTML"""
    return app.send_static_file('sepsis_dashboard.html')

@app.route('/api/model_info')
def model_info():
    """Get model information"""
    info = api.get_model_info()
    if info:
        return jsonify({
            'status': 'success',
            'data': info
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded'
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.get_json()
        print(f"📊 Received prediction request")
        
        if 'features' not in data:
            print("❌ No features in request data")
            return jsonify({
                'status': 'error',
                'message': 'No features provided'
            }), 400
        
        print(f"📋 Raw features data type: {type(data['features'])}")
        print(f"📋 Raw features length: {len(data['features']) if data['features'] else 0}")
        
        # Convert to numpy array
        features = np.array(data['features'])
        print(f"📊 Features array shape: {features.shape}")
        
        # Handle empty features
        if features.size == 0:
            print("❌ Features array is empty")
            return jsonify({
                'status': 'error',
                'message': f'Expected 532 features, got 0'
            }), 400
        
        # Validate feature count
        if len(features.shape) == 1:
            feature_count = features.shape[0]
        else:
            feature_count = features.shape[-1]
            
        print(f"📊 Feature count: {feature_count}")
        
        if feature_count != 532:
            print(f"❌ Feature count mismatch: expected 532, got {feature_count}")
            return jsonify({
                'status': 'error',
                'message': f'Expected 532 features, got {feature_count}'
            }), 400
        
        # Make predictions
        predictions, probabilities = api.predict(features)
        
        if predictions is None:
            return jsonify({
                'status': 'error',
                'message': 'Prediction failed'
            }), 500
        
        # Format results
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            risk_level = 'HIGH RISK - SEPSIS ALERT' if pred == 1 else 'LOW RISK - LIKELY HEALTHY'
            clinical_action = 'IMMEDIATE CLINICAL REVIEW' if pred == 1 else 'CONTINUE MONITORING'
            
            results.append({
                'patient_id': i + 1,
                'prediction': int(pred),
                'probability': float(prob),
                'risk_level': risk_level,
                'clinical_action': clinical_action
            })
        
        # Calculate summary statistics
        total_patients = len(results)
        sepsis_count = sum(1 for r in results if r['prediction'] == 1)
        healthy_count = total_patients - sepsis_count
        risk_percentage = (sepsis_count / total_patients) * 100 if total_patients > 0 else 0
        
        return jsonify({
            'status': 'success',
            'data': {
                'results': results,
                'summary': {
                    'total_patients': total_patients,
                    'sepsis_count': sepsis_count,
                    'healthy_count': healthy_count,
                    'risk_percentage': round(risk_percentage, 1)
                },
                'model_info': api.get_model_info(),
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/api/sample_data/<data_type>')
def get_sample_data(data_type):
    """Generate sample data for testing"""
    try:
        patient_count = 20
        features_count = 532
        
        if data_type == 'sepsis':
            # High-risk patterns (should predict positive)
            data = np.random.uniform(0.4, 1.0, (patient_count, features_count))
            description = f"High-risk sepsis patients (should predict POSITIVE)"
        elif data_type == 'healthy':
            # Low-risk patterns (should predict negative)
            data = np.random.uniform(0.0, 0.3, (patient_count, features_count))
            description = f"Low-risk healthy patients (should predict NEGATIVE)"
        elif data_type == 'mixed':
            # Mixed dataset
            patient_count = 30
            data = np.zeros((patient_count, features_count))
            # First 10 high-risk
            data[:10] = np.random.uniform(0.4, 1.0, (10, features_count))
            # Rest low-risk
            data[10:] = np.random.uniform(0.0, 0.3, (20, features_count))
            description = "Mixed dataset: 10 high-risk + 20 low-risk patients"
        else:
            return jsonify({
                'status': 'error',
                'message': 'Invalid data type. Use: sepsis, healthy, or mixed'
            }), 400
        
        return jsonify({
            'status': 'success',
            'data': {
                'features': data.tolist(),
                'description': description,
                'patient_count': patient_count,
                'features_count': features_count
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error generating sample data: {str(e)}'
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if api.model is not None else "not_loaded"
    
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'timestamp': datetime.now().isoformat(),
        'message': '🏥 Sepsis Prediction API is running'
    })

if __name__ == '__main__':
    print("\n🏥 SEPSIS PREDICTION DASHBOARD SERVER")
    print("=" * 50)
    print(f"🔥 Model Status: {'✅ Loaded' if api.model else '❌ Not Loaded'}")
    print(f"🌐 Dashboard URL: http://localhost:5000")
    print(f"🔗 API Health: http://localhost:5000/api/health")
    print(f"📊 Model Info: http://localhost:5000/api/model_info")
    print("=" * 50)
    print("🚀 Starting server...")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False
    )