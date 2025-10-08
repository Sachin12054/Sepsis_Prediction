#!/usr/bin/env python3
"""
🔧 Enhanced Dashboard Server with Butterworth Filtering
=====================================================
Updated Flask server with Butterworth-enhanced STFT features for sepsis prediction
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os
import json
from datetime import datetime
import warnings
import sys

# Add signal processing module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'signal_processing'))

try:
    from signal_processing.enhanced_stft_integration import EnhancedSTFTProcessor, predict_with_butterworth_enhancement
    BUTTERWORTH_AVAILABLE = True
    print("✅ Butterworth filtering available")
except ImportError:
    BUTTERWORTH_AVAILABLE = False
    print("⚠️ Butterworth filtering not available - using standard processing")

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Add CORS headers manually
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

class EnhancedSepsisPredictionAPI:
    def __init__(self):
        self.model = None
        self.enhanced_model = None
        self.model_path = "models/clinical_sepsis_model.pkl"
        self.enhanced_model_path = "models/enhanced_butterworth_sepsis_model.pkl"
        self.butterworth_processor = None
        
        self.load_models()
    
    def load_models(self):
        """Load both standard and enhanced models"""
        # Load standard model
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print(f"✅ Standard clinical model loaded")
            else:
                print(f"❌ Standard model not found: {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading standard model: {e}")
        
        # Load/create enhanced model with Butterworth
        if BUTTERWORTH_AVAILABLE:
            try:
                if os.path.exists(self.enhanced_model_path):
                    self.enhanced_model = joblib.load(self.enhanced_model_path)
                    print(f"✅ Enhanced Butterworth model loaded")
                else:
                    print(f"🔧 Creating enhanced Butterworth model...")
                    from signal_processing.enhanced_stft_integration import integrate_butterworth_with_existing_model
                    self.enhanced_model = integrate_butterworth_with_existing_model()
                    print(f"✅ Enhanced Butterworth model created")
                
                self.butterworth_processor = EnhancedSTFTProcessor(sampling_rate=100)
                
            except Exception as e:
                print(f"❌ Error with enhanced model: {e}")
                self.enhanced_model = None
    
    def predict_standard(self, features):
        """Make predictions using standard model"""
        if self.model is None:
            return None, None
        
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
            print(f"❌ Standard prediction error: {e}")
            return None, None
    
    def predict_enhanced(self, patient_signals):
        """Make predictions using Butterworth-enhanced processing"""
        if not BUTTERWORTH_AVAILABLE or self.enhanced_model is None:
            return None, None
        
        try:
            # Convert patient signals to proper format
            if isinstance(patient_signals, np.ndarray):
                # Assume signals are arranged as [HR, SBP, DBP, MAP, Temp, Resp] columns
                signal_names = ['HR', 'SBP', 'DBP', 'MAP', 'Temp', 'Resp']
                patient_data = {}
                
                for i, name in enumerate(signal_names):
                    if i < patient_signals.shape[1]:
                        patient_data[name] = patient_signals[:, i]
            else:
                patient_data = patient_signals
            
            # Process with Butterworth enhancement
            result = predict_with_butterworth_enhancement(
                patient_data, 
                self.enhanced_model_path
            )
            
            if 'sepsis_probability' in result:
                predictions = np.array([result['sepsis_prediction']])
                probabilities = np.array([result['sepsis_probability']])
                return predictions, probabilities, result
            else:
                return None, None, result
                
        except Exception as e:
            print(f"❌ Enhanced prediction error: {e}")
            return None, None, None
    
    def get_model_info(self):
        """Get comprehensive model information"""
        info = {
            'standard_model': self.model is not None,
            'enhanced_model': self.enhanced_model is not None,
            'butterworth_available': BUTTERWORTH_AVAILABLE
        }
        
        if self.model:
            info.update({
                'algorithm': self.model['model_info']['algorithm'],
                'threshold': self.model['threshold'],
                'sensitivity': self.model['performance_metrics']['sensitivity'],
                'specificity': self.model['performance_metrics']['specificity'],
                'accuracy': self.model['performance_metrics']['accuracy'],
                'clinical_status': self.model['clinical_validation']['approval_status']
            })
        
        if self.enhanced_model and BUTTERWORTH_AVAILABLE:
            info.update({
                'enhancement': 'Butterworth + STFT',
                'enhanced_features': 532,
                'signal_processing': 'Clinical-grade filtering'
            })
        
        return info

# Initialize the enhanced API
api = EnhancedSepsisPredictionAPI()

@app.route('/')
def dashboard():
    """Serve the dashboard HTML"""
    return app.send_static_file('sepsis_dashboard_live.html')

@app.route('/api/model_info')
def model_info():
    """Get comprehensive model information"""
    info = api.get_model_info()
    return jsonify({
        'status': 'success',
        'data': info
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle prediction requests with optional Butterworth enhancement"""
    try:
        data = request.get_json()
        print(f"📊 Received prediction request")
        
        if 'features' not in data:
            print("❌ No features in request data")
            return jsonify({
                'status': 'error',
                'message': 'No features provided'
            }), 400
        
        # Get processing preference
        use_butterworth = data.get('use_butterworth', True)
        
        # Convert to numpy array
        features = np.array(data['features'])
        print(f"📊 Features array shape: {features.shape}")
        print(f"🔧 Butterworth enhancement: {'enabled' if use_butterworth else 'disabled'}")
        
        # Handle empty features
        if features.size == 0:
            print("❌ Features array is empty")
            return jsonify({
                'status': 'error',
                'message': f'Expected 532 features, got 0'
            }), 400
        
        results = []
        processing_info = {}
        
        # Try enhanced prediction first if requested and available
        if use_butterworth and BUTTERWORTH_AVAILABLE and api.enhanced_model:
            print("🔧 Using Butterworth-enhanced processing...")
            
            try:
                # Reshape features for signal processing (assume time series data)
                if len(features.shape) == 2 and features.shape[1] == 532:
                    # Standard 532 features - convert to time series format
                    # Assume features represent 6 signals with ~88 features each
                    n_patients = features.shape[0]
                    
                    for i in range(n_patients):
                        # Create synthetic physiological signals from features
                        # This is a reconstruction for demonstration
                        patient_signals = {
                            'HR': features[i, :88],
                            'SBP': features[i, 88:176], 
                            'DBP': features[i, 176:264],
                            'MAP': features[i, 264:352],
                            'Temp': features[i, 352:440],
                            'Resp': features[i, 440:532]
                        }
                        
                        predictions, probabilities, enhanced_result = api.predict_enhanced(patient_signals)
                        
                        if predictions is not None:
                            pred = int(predictions[0])
                            prob = float(probabilities[0])
                            
                            risk_level = 'HIGH RISK - SEPSIS ALERT' if pred == 1 else 'LOW RISK - LIKELY HEALTHY'
                            clinical_action = 'IMMEDIATE CLINICAL REVIEW' if pred == 1 else 'CONTINUE MONITORING'
                            
                            results.append({
                                'patient_id': i + 1,
                                'prediction': pred,
                                'probability': prob,
                                'risk_level': risk_level,
                                'clinical_action': clinical_action,
                                'processing': 'Butterworth + STFT Enhanced'
                            })
                        else:
                            raise Exception("Enhanced prediction failed")
                    
                    processing_info = {
                        'method': 'Butterworth + STFT Enhanced',
                        'signal_filtering': 'Clinical-grade Butterworth filters',
                        'feature_enhancement': 'Advanced STFT with noise reduction'
                    }
                    
                else:
                    raise Exception("Feature format not suitable for Butterworth processing")
                    
            except Exception as e:
                print(f"⚠️ Butterworth processing failed: {e}, falling back to standard...")
                use_butterworth = False
        
        # Use standard prediction if Butterworth failed or not requested
        if not use_butterworth or not results:
            print("📊 Using standard processing...")
            
            # Validate feature count for standard model
            if len(features.shape) == 1:
                feature_count = features.shape[0]
            else:
                feature_count = features.shape[-1]
                
            if feature_count != 532:
                print(f"❌ Feature count mismatch: expected 532, got {feature_count}")
                return jsonify({
                    'status': 'error',
                    'message': f'Expected 532 features, got {feature_count}'
                }), 400
            
            # Make standard predictions
            predictions, probabilities = api.predict_standard(features)
            
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
                    'clinical_action': clinical_action,
                    'processing': 'Standard STFT'
                })
            
            processing_info = {
                'method': 'Standard STFT',
                'feature_extraction': 'Original 532 STFT features',
                'enhancement': 'None'
            }
        
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
                'processing_info': processing_info,
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
    """Generate sample data for testing (now with Butterworth option)"""
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
                'features_count': features_count,
                'butterworth_compatible': True
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error generating sample data: {str(e)}'
        }), 500

@app.route('/api/health')
def health_check():
    """Enhanced health check endpoint"""
    model_status = "loaded" if api.model is not None else "not_loaded"
    enhanced_status = "loaded" if api.enhanced_model is not None else "not_loaded"
    
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'enhanced_model_status': enhanced_status,
        'butterworth_available': BUTTERWORTH_AVAILABLE,
        'timestamp': datetime.now().isoformat(),
        'message': '🔧 Enhanced Sepsis Prediction API with Butterworth Filtering',
        'capabilities': {
            'standard_stft': api.model is not None,
            'butterworth_enhanced': api.enhanced_model is not None and BUTTERWORTH_AVAILABLE,
            'clinical_grade_filtering': BUTTERWORTH_AVAILABLE
        }
    })

if __name__ == '__main__':
    print("\n🔧 ENHANCED SEPSIS PREDICTION DASHBOARD SERVER")
    print("=" * 60)
    print(f"🔥 Standard Model: {'✅ Loaded' if api.model else '❌ Not Loaded'}")
    print(f"⚡ Enhanced Model: {'✅ Loaded' if api.enhanced_model else '❌ Not Loaded'}")
    print(f"🎛️ Butterworth Filtering: {'✅ Available' if BUTTERWORTH_AVAILABLE else '❌ Not Available'}")
    print(f"🌐 Dashboard URL: http://localhost:5000")
    print(f"🔗 API Health: http://localhost:5000/api/health")
    print(f"📊 Model Info: http://localhost:5000/api/model_info")
    print("=" * 60)
    print("🚀 Starting enhanced server...")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False
    )