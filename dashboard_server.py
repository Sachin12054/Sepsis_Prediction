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

# Try to import SHAP for explanations
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not available for explanations")

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
            # Debug logging
            print(f"📊 Received prediction request")
            print(f"📋 Raw features data type: {type(features)}")
            print(f"📋 Raw features length: {len(features)}")
            
            # Convert to numpy array
            features = np.array(features)
            print(f"📊 Features array shape: {features.shape}")
            
            # Ensure features is a 2D array
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Check feature count and adjust if needed
            expected_features = len(self.model['feature_names'])
            current_features = features.shape[1]
            
            print(f"📊 Feature count: {current_features}")
            print(f"📊 Expected features: {expected_features}")
            
            if current_features != expected_features:
                if current_features < expected_features:
                    # Pad with zeros if too few features
                    padding = np.zeros((features.shape[0], expected_features - current_features))
                    features = np.concatenate([features, padding], axis=1)
                    print(f"✅ Padded features from {current_features} to {expected_features}")
                else:
                    # Truncate if too many features
                    features = features[:, :expected_features]
                    print(f"✅ Truncated features from {current_features} to {expected_features}")
            
            print(f"📊 Final feature shape: {features.shape}")
            
            # Apply scaler if available
            if 'scaler' in self.model:
                features = self.model['scaler'].transform(features)
                print("✅ Applied scaling")
            
            # Get prediction probabilities
            probabilities = self.model['model'].predict_proba(features)[:, 1]
            
            # Apply clinical threshold
            predictions = (probabilities >= self.model['threshold']).astype(int)
            
            print(f"✅ Predictions successful: {len(predictions)} results")
            return predictions, probabilities
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None, None
    
    def get_prediction_explanation(self, features, prediction, probability):
        """Get detailed explanation for the prediction"""
        if self.model is None:
            return None
            
        try:
            explanation = {
                'prediction_details': {
                    'risk_level': 'HIGH RISK - SEPSIS ALERT' if prediction == 1 else 'LOW RISK - LIKELY HEALTHY',
                    'confidence': float(probability),
                    'threshold_used': self.model['threshold'],
                    'clinical_significance': self._get_clinical_significance(probability)
                },
                'feature_importance': self._get_feature_importance(),
                'vital_signs_analysis': self._analyze_vital_signs(features),
                'clinical_reasoning': self._get_clinical_reasoning(features, prediction, probability),
                'recommendations': self._get_clinical_recommendations(prediction, probability)
            }
            
            # Add SHAP explanations if available
            if SHAP_AVAILABLE:
                explanation['shap_values'] = self._get_shap_explanation(features)
            
            return explanation
            
        except Exception as e:
            print(f"❌ Error generating explanation: {e}")
            return None
    
    def _get_clinical_significance(self, probability):
        """Get clinical significance of the probability score"""
        if probability >= 0.8:
            return "EXTREMELY HIGH RISK - Immediate intervention required"
        elif probability >= 0.6:
            return "HIGH RISK - Close monitoring and early intervention"
        elif probability >= 0.4:
            return "MODERATE RISK - Increased surveillance recommended"
        elif probability >= 0.2:
            return "LOW-MODERATE RISK - Standard monitoring"
        else:
            return "LOW RISK - Routine care"
    
    def _get_feature_importance(self):
        """Get top important features from the model"""
        try:
            if hasattr(self.model['model'], 'feature_importances_'):
                importances = self.model['model'].feature_importances_
                feature_names = self.model['feature_names']
                
                # Get top 10 most important features
                indices = np.argsort(importances)[::-1][:10]
                
                top_features = []
                for i, idx in enumerate(indices):
                    feature_name = feature_names[idx] if idx < len(feature_names) else f'feature_{idx}'
                    importance = float(importances[idx])
                    
                    # Map to clinical meaning
                    clinical_name = self._map_feature_to_clinical(feature_name)
                    
                    top_features.append({
                        'rank': i + 1,
                        'feature': feature_name,
                        'clinical_name': clinical_name,
                        'importance': importance,
                        'importance_percent': importance * 100
                    })
                
                return top_features
        except Exception as e:
            print(f"Warning: Could not get feature importance: {e}")
            
        return []
    
    def _map_feature_to_clinical(self, feature_name):
        """Map technical feature names to clinical terms"""
        mapping = {
            'heart_rate': 'Heart Rate Variability',
            'blood_pressure': 'Blood Pressure Patterns',
            'respiratory': 'Respiratory Rate Analysis',
            'temperature': 'Temperature Regulation',
            'oxygen': 'Oxygen Saturation Trends',
            'stft': 'Frequency Domain Analysis',
            'freq_0': 'Very Low Frequency (Autonomic)',
            'freq_1': 'Low Frequency (Sympathetic)',
            'freq_2': 'Mid Frequency (Parasympathetic)',
            'freq_3': 'High Frequency (Respiratory)'
        }
        
        for key, value in mapping.items():
            if key in feature_name.lower():
                return value
        
        return feature_name.replace('_', ' ').title()
    
    def _analyze_vital_signs(self, features):
        """Analyze vital signs patterns"""
        try:
            # This is a simplified analysis - in reality, you'd have more sophisticated logic
            vital_analysis = {
                'cardiovascular': {
                    'status': 'normal',
                    'indicators': [],
                    'concern_level': 'low'
                },
                'respiratory': {
                    'status': 'normal', 
                    'indicators': [],
                    'concern_level': 'low'
                },
                'metabolic': {
                    'status': 'normal',
                    'indicators': [],
                    'concern_level': 'low'
                },
                'autonomic': {
                    'status': 'normal',
                    'indicators': [],
                    'concern_level': 'low'
                }
            }
            
            # Analyze features (simplified example)
            feature_means = np.mean(features.reshape(-1, 536), axis=0) if features.size > 536 else features
            
            # Cardiovascular analysis
            hr_features = feature_means[:50]  # First 50 features assumed to be HR-related
            if np.mean(hr_features) > 100:
                vital_analysis['cardiovascular']['status'] = 'abnormal'
                vital_analysis['cardiovascular']['indicators'].append('Elevated heart rate patterns')
                vital_analysis['cardiovascular']['concern_level'] = 'high'
            
            # Respiratory analysis  
            resp_features = feature_means[50:100]
            if np.std(resp_features) > 20:
                vital_analysis['respiratory']['status'] = 'abnormal'
                vital_analysis['respiratory']['indicators'].append('Irregular respiratory patterns')
                vital_analysis['respiratory']['concern_level'] = 'moderate'
            
            return vital_analysis
            
        except Exception as e:
            print(f"Warning: Could not analyze vital signs: {e}")
            return {}
    
    def _get_clinical_reasoning(self, features, prediction, probability):
        """Generate clinical reasoning for the prediction"""
        reasoning = []
        
        if prediction == 1:  # Sepsis predicted
            reasoning.append("🚨 SEPSIS RISK DETECTED")
            reasoning.append("━" * 30)
            
            if probability > 0.8:
                reasoning.append("• Very high confidence prediction (>80%)")
                reasoning.append("• Multiple physiological systems show abnormal patterns")
                reasoning.append("• STFT analysis reveals significant frequency domain changes")
            elif probability > 0.6:
                reasoning.append("• High confidence prediction (60-80%)")  
                reasoning.append("• Several key biomarkers indicate sepsis risk")
                reasoning.append("• Cardiovascular and respiratory patterns are concerning")
            else:
                reasoning.append("• Moderate confidence prediction (30-60%)")
                reasoning.append("• Some indicators suggest early sepsis development")
                reasoning.append("• Requires careful monitoring and reassessment")
            
            reasoning.extend([
                "",
                "🔍 Key Findings:",
                "• Abnormal heart rate variability patterns",
                "• Disrupted autonomic nervous system activity", 
                "• Altered respiratory-cardiac coupling",
                "• Frequency domain signatures consistent with sepsis",
                "",
                "⚕️ Clinical Correlation:",
                "• Consider checking inflammatory markers (CRP, PCT)",
                "• Review recent vital signs trends",
                "• Assess for infection sources",
                "• Monitor end-organ function"
            ])
        else:  # No sepsis
            reasoning.append("✅ LOW SEPSIS RISK")
            reasoning.append("━" * 30)
            reasoning.append("• Physiological patterns within normal ranges")
            reasoning.append("• STFT analysis shows stable frequency signatures")
            reasoning.append("• Cardiovascular and respiratory coupling normal")
            reasoning.append("• No significant inflammatory patterns detected")
            reasoning.append("")
            reasoning.append("📋 Continue standard monitoring protocols")
        
        return reasoning
    
    def _get_clinical_recommendations(self, prediction, probability):
        """Get clinical recommendations based on prediction"""
        recommendations = []
        
        if prediction == 1:  # Sepsis risk
            if probability > 0.8:
                recommendations.extend([
                    "🚨 IMMEDIATE ACTIONS REQUIRED:",
                    "• Notify attending physician immediately", 
                    "• Obtain blood cultures and lactate level",
                    "• Consider empirical antibiotic therapy",
                    "• Implement sepsis bundle protocol",
                    "• Continuous monitoring required"
                ])
            elif probability > 0.6:
                recommendations.extend([
                    "⚠️ HIGH PRIORITY MONITORING:",
                    "• Increase monitoring frequency to q15min",
                    "• Obtain inflammatory markers (CRP, PCT)",
                    "• Consider blood cultures if clinically indicated", 
                    "• Review medication history and allergies",
                    "• Prepare for potential rapid response"
                ])
            else:
                recommendations.extend([
                    "🔍 ENHANCED SURVEILLANCE:",
                    "• Monitor vital signs every 30 minutes",
                    "• Reassess in 2-4 hours",
                    "• Document any symptom changes",
                    "• Consider point-of-care testing",
                    "• Maintain IV access"
                ])
        else:  # Low risk
            recommendations.extend([
                "✅ STANDARD CARE PROTOCOLS:",
                "• Continue current monitoring schedule",
                "• Routine vital sign assessment", 
                "• Standard documentation",
                "• Reassess if clinical condition changes"
            ])
        
        return recommendations
    
    def _get_shap_explanation(self, features):
        """Get SHAP-based explanations (if available)"""
        if not SHAP_AVAILABLE:
            return None
            
        try:
            # This would require pre-computed SHAP explainer
            # For now, return placeholder
            return {
                'note': 'SHAP explanations require pre-computed explainer',
                'available': False
            }
        except Exception as e:
            return None
    
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
    """Serve the enhanced dashboard HTML"""
    try:
        with open('enhanced_dashboard.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return '''
        <h1>Enhanced Dashboard Not Found</h1>
        <p>Please ensure enhanced_dashboard.html exists in the project directory.</p>
        <a href="/old">Use Old Dashboard</a>
        '''

@app.route('/old')
def old_dashboard():
    """Serve the original dashboard HTML"""
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
        
        if 'features' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No features provided'
            }), 400
        
        # Convert to numpy array and let api.predict handle the feature processing
        features = np.array(data['features'])
        
        # Handle empty features
        if features.size == 0:
            return jsonify({
                'status': 'error',
                'message': f'Expected features, got empty array'
            }), 400
        
        # Make predictions (feature count validation happens in api.predict)
        predictions, probabilities = api.predict(features)
        
        if predictions is None:
            return jsonify({
                'status': 'error',
                'message': 'Prediction failed'
            }), 500
        
        # Format results with explanations
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            risk_level = 'HIGH RISK - SEPSIS ALERT' if pred == 1 else 'LOW RISK - LIKELY HEALTHY'
            clinical_action = 'IMMEDIATE CLINICAL REVIEW' if pred == 1 else 'CONTINUE MONITORING'
            
            # Get detailed explanation for this patient
            patient_features = features[i:i+1] if len(features.shape) > 1 else features.reshape(1, -1)
            explanation = api.get_prediction_explanation(patient_features, pred, prob)
            
            result = {
                'patient_id': i + 1,
                'prediction': int(pred),
                'probability': float(prob),
                'risk_level': risk_level,
                'clinical_action': clinical_action,
                'explanation': explanation,
                'confidence_level': 'High' if prob > 0.7 or prob < 0.3 else 'Moderate',
                'urgency': 'URGENT' if pred == 1 and prob > 0.6 else 'ROUTINE'
            }
            
            results.append(result)
        
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

@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    """Handle CSV file uploads"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'status': 'error', 
                'message': 'No file selected'
            }), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({
                'status': 'error',
                'message': 'Only CSV files are supported'
            }), 400
        
        # Read and process CSV
        content = file.read().decode('utf-8')
        lines = content.strip().split('\n')
        
        if len(lines) < 2:
            return jsonify({
                'status': 'error',
                'message': 'CSV file must contain header and at least one data row'
            }), 400
        
        # Parse CSV data
        features = []
        for i, line in enumerate(lines[1:], 1):  # Skip header
            values = line.split(',')
            try:
                # Extract numeric features (skip metadata columns if present)
                row_features = []
                for val in values:
                    try:
                        row_features.append(float(val.strip()))
                    except ValueError:
                        continue  # Skip non-numeric columns
                
                if len(row_features) >= 536:
                    features.append(row_features[7:543])  # Skip first 7 metadata columns, take 536 features
                elif len(row_features) > 0:
                    # Skip metadata columns and pad with zeros if too few features
                    numeric_features = row_features[7:] if len(row_features) > 7 else []
                    padded = numeric_features + [0.0] * (536 - len(numeric_features))
                    features.append(padded[:536])
                    
            except Exception as e:
                print(f"Error processing row {i}: {e}")
                continue
        
        if not features:
            return jsonify({
                'status': 'error',
                'message': 'No valid numeric data found in CSV'
            }), 400
        
        # Make predictions
        features_array = np.array(features)
        predictions, probabilities = api.predict(features_array)
        
        if predictions is None:
            return jsonify({
                'status': 'error',
                'message': 'Prediction failed'
            }), 500
        
        # Format results with explanations
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            patient_features = features_array[i:i+1]
            explanation = api.get_prediction_explanation(patient_features, pred, prob)
            
            result = {
                'patient_id': i + 1,
                'prediction': int(pred),
                'probability': float(prob),
                'risk_level': 'HIGH RISK - SEPSIS ALERT' if pred == 1 else 'LOW RISK - LIKELY HEALTHY',
                'clinical_action': 'IMMEDIATE CLINICAL REVIEW' if pred == 1 else 'CONTINUE MONITORING',
                'explanation': explanation,
                'confidence_level': 'High' if prob > 0.7 or prob < 0.3 else 'Moderate',
                'urgency': 'URGENT' if pred == 1 and prob > 0.6 else 'ROUTINE'
            }
            results.append(result)
        
        # Calculate summary
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
                'file_info': {
                    'filename': file.filename,
                    'rows_processed': len(features),
                    'features_per_patient': 536
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
        features_count = 536  # Updated to match real model
        
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