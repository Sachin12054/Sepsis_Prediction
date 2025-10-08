#!/usr/bin/env python3
"""
🏥 ICU-Compatible Sepsis Prediction Dashboard
==========================================
Enhanced Flask server that handles both CSV test files and real ICU PSV patient data

Features:
- Auto-detects file format (CSV vs PSV)
- Processes real hospital ICU data
- Converts raw vital signs to 536 STFT features
- Maintains compatibility with existing test files
"""

from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import joblib
import traceback
import os
import sys

# Import our ICU converter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from icu_data_converter import ICUDataConverter

app = Flask(__name__)

# Global variables to store loaded models
models = {}
icu_converter = None

def load_models():
    """Load all pre-trained models"""
    global models, icu_converter
    
    print("🔄 Loading ensemble models...")
    model_files = {
        'best_ensemble': 'ensemble_models/final_best_model.pkl',
        'stacking_ensemble': 'ensemble_models/ensemble_stacking.pkl',
        'voting_soft': 'ensemble_models/ensemble_votingsoft.pkl',
        'adaboost_ensemble': 'ensemble_models/ensemble_adaboost.pkl'
    }
    
    for name, path in model_files.items():
        try:
            if os.path.exists(path):
                models[name] = joblib.load(path)
                print(f"✅ Loaded {name}")
            else:
                print(f"⚠️  Model file not found: {path}")
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")
    
    # Initialize ICU converter
    try:
        icu_converter = ICUDataConverter()
        print("✅ ICU Data Converter initialized")
    except Exception as e:
        print(f"❌ Error initializing ICU converter: {e}")
    
    print(f"📊 Total models loaded: {len(models)}")

def detect_file_format(file_content):
    """Detect if uploaded file is CSV or PSV format"""
    # Check first few lines for format indicators
    lines = file_content.split('\n')[:5]
    
    # PSV files typically have pipe separators and specific headers
    for line in lines:
        if '|' in line and any(header in line.lower() for header in ['hr', 'temp', 'sbp', 'dbp', 'map']):
            return 'psv'
    
    # CSV files typically have comma separators
    for line in lines:
        if ',' in line:
            return 'csv'
    
    return 'unknown'

def process_csv_file(file_content):
    """Process standard CSV test file (536 features from 543 columns)"""
    try:
        # Convert to pandas DataFrame
        from io import StringIO
        df = pd.read_csv(StringIO(file_content))
        
        # Should have 543 columns (7 metadata + 536 features)
        if df.shape[1] != 543:
            return None, f"CSV file should have 543 columns (7 metadata + 536 features), found {df.shape[1]}"
        
        # Extract feature columns (skip first 7 metadata columns)
        features_df = df.iloc[:, 7:]  # Skip metadata columns
        features = features_df.values
        
        return features, None
    except Exception as e:
        return None, f"Error processing CSV: {str(e)}"

def process_psv_file(file_content):
    """Process ICU PSV file and convert to 536 STFT features"""
    try:
        if icu_converter is None:
            return None, "ICU converter not initialized"
        
        # Convert PSV to STFT features
        features = icu_converter.convert_from_text(file_content)
        
        if features is None:
            return None, "Failed to extract STFT features from PSV file"
        
        # Ensure features are in correct format (N, 536)
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        return features, None
    except Exception as e:
        return None, f"Error processing PSV: {str(e)}"

def make_predictions(features):
    """Make predictions using all available models"""
    predictions = {}
    
    for model_name, model in models.items():
        try:
            pred = model.predict_proba(features)[0][1]  # Probability of sepsis (class 1)
            predictions[f"{model_name}_prediction"] = float(pred)
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
            predictions[f"{model_name}_prediction"] = None
    
    return predictions

@app.route('/')
def home():
    """Main dashboard page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        # Read file content
        file_content = file.read().decode('utf-8')
        
        # Detect file format
        file_format = detect_file_format(file_content)
        print(f"🔍 Detected file format: {file_format}")
        
        # Process based on format
        if file_format == 'csv':
            features, error = process_csv_file(file_content)
        elif file_format == 'psv':
            features, error = process_psv_file(file_content)
        else:
            return jsonify({'error': 'Unsupported file format. Please upload CSV or PSV files.'})
        
        if error:
            return jsonify({'error': error})
        
        if features is None:
            return jsonify({'error': 'Could not extract features from file'})
        
        print(f"📊 Extracted features shape: {features.shape}")
        
        # Make predictions
        predictions = make_predictions(features)
        
        # Add file info
        result = {
            'file_format': file_format,
            'filename': file.filename,
            'features_shape': list(features.shape),
            **predictions
        }
        
        # Calculate risk assessment
        valid_predictions = [p for p in predictions.values() if p is not None]
        if valid_predictions:
            avg_risk = np.mean(valid_predictions)
            result['average_risk'] = float(avg_risk)
            result['risk_level'] = 'HIGH' if avg_risk > 0.5 else 'MODERATE' if avg_risk > 0.3 else 'LOW'
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        print(traceback.format_exc())
        return jsonify({'error': f'Prediction failed: {str(e)}'})

@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    """Handle CSV file uploads for batch processing"""
    try:
        print("📊 Received prediction request")
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read file content
        file_content = file.read().decode('utf-8')
        print(f"📋 Raw features data type: {type(file_content)}")
        
        # Process CSV data
        lines = file_content.strip().split('\n')
        if len(lines) < 2:
            return jsonify({'error': 'CSV file must contain header and at least one data row'}), 400
        
        features = []
        for i, line in enumerate(lines[1:], 1):  # Skip header
            values = line.split(',')
            try:
                # Extract numeric features (skip first 7 metadata columns)
                row_features = []
                for j, val in enumerate(values):
                    if j < 7:  # Skip metadata columns
                        continue
                    try:
                        row_features.append(float(val.strip()))
                    except ValueError:
                        continue
                
                if len(row_features) >= 536:
                    features.append(row_features[:536])  # Take first 536 features
                elif len(row_features) > 0:
                    # Pad with zeros if too few features
                    padded = row_features + [0.0] * (536 - len(row_features))
                    features.append(padded[:536])
                    
            except Exception as e:
                print(f"Error processing row {i}: {e}")
                continue
        
        if not features:
            return jsonify({'error': 'No valid numeric data found in CSV'}), 400
        
        features_array = np.array(features)
        print(f"📋 Raw features length: {len(features)}")
        print(f"📊 Features array shape: {features_array.shape}")
        print(f"📊 Feature count: {features_array.shape[1]}")
        print(f"📊 Expected features: 536")
        print(f"📊 Final feature shape: {features_array.shape}")
        
        # Apply scaling if available
        try:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            features_array = scaler.fit_transform(features_array)
            print("✅ Applied scaling")
        except:
            print("⚠️ No scaling applied")
        
        # Make predictions with all models
        results = []
        print(f"✅ Predictions successful: {len(features)} results")
        
        for i, feature_row in enumerate(features_array):
            feature_row = feature_row.reshape(1, -1)
            predictions = make_predictions(feature_row)
            
            # Calculate average risk
            valid_predictions = [p for p in predictions.values() if p is not None]
            avg_risk = np.mean(valid_predictions) if valid_predictions else 0
            
            result = {
                'patient_id': i + 1,
                'prediction': 1 if avg_risk > 0.5 else 0,
                'probability': float(avg_risk),
                'risk_level': 'HIGH RISK - SEPSIS ALERT' if avg_risk > 0.5 else 'LOW RISK - LIKELY HEALTHY',
                'clinical_action': 'IMMEDIATE CLINICAL REVIEW' if avg_risk > 0.5 else 'CONTINUE MONITORING',
                'confidence_level': 'High' if avg_risk > 0.7 or avg_risk < 0.3 else 'Moderate',
                'urgency': 'URGENT' if avg_risk > 0.6 else 'ROUTINE',
                **predictions
            }
            results.append(result)
        
        # Calculate summary
        total_patients = len(results)
        sepsis_count = sum(1 for r in results if r['prediction'] == 1)
        healthy_count = total_patients - sepsis_count
        risk_percentage = (sepsis_count / total_patients) * 100 if total_patients > 0 else 0
        
        return jsonify({
            'status': 'success',
            'message': f'Processed {total_patients} patients successfully',
            'summary': {
                'total_patients': total_patients,
                'sepsis_cases': sepsis_count,
                'healthy_cases': healthy_count,
                'risk_percentage': round(risk_percentage, 2),
                'features_per_patient': 536
            },
            'results': results
        })
        
    except Exception as e:
        print(f"❌ Error in upload_csv: {e}")
        print(traceback.format_exc())
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/health')
def api_health():
    """API Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'icu_converter': icu_converter is not None,
        'expected_features': 536
    })

@app.route('/hybridaction/zybTrackerStatisticsAction')
def block_tracking():
    """Block tracking requests"""
    return jsonify({'status': 'blocked'}), 204

@app.route('/api/model_info')
def api_model_info():
    """Get model information"""
    return jsonify({
        'status': 'active',
        'models': list(models.keys()),
        'features_expected': 536,
        'supports_csv': True,
        'supports_psv': True
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'icu_converter': icu_converter is not None
    })

# HTML Template for the dashboard
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🏥 ICU Sepsis Prediction Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .upload-section {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            border: 3px dashed #dee2e6;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .upload-section:hover {
            border-color: #4facfe;
            background: #f0f8ff;
        }
        
        .file-input-wrapper {
            position: relative;
            display: inline-block;
            cursor: pointer;
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 15px 30px;
            border-radius: 50px;
            font-size: 1.1em;
            font-weight: bold;
            box-shadow: 0 10px 20px rgba(79, 172, 254, 0.3);
            transition: all 0.3s ease;
        }
        
        .file-input-wrapper:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px rgba(79, 172, 254, 0.4);
        }
        
        #fileInput {
            position: absolute;
            left: -9999px;
        }
        
        .predict-btn {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 50px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            margin-top: 20px;
            box-shadow: 0 10px 20px rgba(238, 90, 36, 0.3);
            transition: all 0.3s ease;
        }
        
        .predict-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px rgba(238, 90, 36, 0.4);
        }
        
        .predict-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .results {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-top: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            display: none;
        }
        
        .risk-indicator {
            text-align: center;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 30px;
            font-size: 1.5em;
            font-weight: bold;
        }
        
        .risk-high {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            color: white;
        }
        
        .risk-moderate {
            background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
            color: white;
        }
        
        .risk-low {
            background: linear-gradient(135deg, #1dd1a1 0%, #55efc4 100%);
            color: white;
        }
        
        .models-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .model-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        
        .model-card h5 {
            font-size: 1.2em;
            margin-bottom: 10px;
        }
        
        .model-card p {
            font-size: 2em;
            font-weight: bold;
        }
        
        .patient-info {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
        }
        
        .patient-info h4 {
            color: #333;
            margin-bottom: 15px;
        }
        
        .patient-info ul {
            list-style: none;
        }
        
        .patient-info li {
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #4facfe;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            background: #ff6b6b;
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            display: none;
        }
        
        .file-info {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 ICU Sepsis Prediction</h1>
            <p>Advanced AI-powered sepsis risk assessment for critical care</p>
        </div>
        
        <div class="content">
            <div class="upload-section">
                <h3>📁 Upload Patient Data</h3>
                <p style="margin: 15px 0;">Support for both CSV test files and real ICU PSV patient data</p>
                
                <label for="fileInput" class="file-input-wrapper">
                    📤 Choose File
                </label>
                <input type="file" id="fileInput" accept=".csv,.psv,.txt">
                
                <div id="selectedFile" style="margin-top: 15px; font-weight: bold;"></div>
                
                <button id="predictBtn" class="predict-btn" disabled>
                    🔮 Analyze Sepsis Risk
                </button>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing patient data...</p>
            </div>
            
            <div class="error" id="error"></div>
            
            <div class="results" id="results">
                <div id="fileInfo"></div>
                <div id="riskIndicator"></div>
                <div id="patientInfo"></div>
                <div id="modelPredictions"></div>
            </div>
        </div>
    </div>

    <script>
        const fileInput = document.getElementById('fileInput');
        const selectedFile = document.getElementById('selectedFile');
        const predictBtn = document.getElementById('predictBtn');
        const loading = document.getElementById('loading');
        const error = document.getElementById('error');
        const results = document.getElementById('results');

        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                selectedFile.textContent = `Selected: ${file.name}`;
                predictBtn.disabled = false;
            } else {
                selectedFile.textContent = '';
                predictBtn.disabled = true;
            }
        });

        predictBtn.addEventListener('click', function() {
            const file = fileInput.files[0];
            if (!file) return;

            // Hide previous results and errors
            results.style.display = 'none';
            error.style.display = 'none';
            loading.style.display = 'block';

            const formData = new FormData();
            formData.append('file', file);

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(result => {
                loading.style.display = 'none';
                
                if (result.error) {
                    showError(result.error);
                    return;
                }
                
                showResults(result);
            })
            .catch(err => {
                loading.style.display = 'none';
                showError('Network error: ' + err.message);
            });
        });

        function showError(message) {
            error.textContent = message;
            error.style.display = 'block';
        }

        function showResults(result) {
            // File info
            const fileInfo = document.getElementById('fileInfo');
            fileInfo.innerHTML = `
                <div class="file-info">
                    <h4>📋 File Information</h4>
                    <p><strong>Format:</strong> ${result.file_format.toUpperCase()}</p>
                    <p><strong>Features extracted:</strong> ${result.features_shape[1]} features</p>
                </div>
            `;

            // Risk indicator
            const riskIndicator = document.getElementById('riskIndicator');
            const riskLevel = result.risk_level || 'UNKNOWN';
            const avgRisk = result.average_risk || 0;
            
            riskIndicator.innerHTML = `
                <div class="risk-indicator risk-${riskLevel.toLowerCase()}">
                    <div>🚨 SEPSIS RISK: ${riskLevel}</div>
                    <div style="font-size: 0.8em; margin-top: 10px;">
                        Average Risk Score: ${(avgRisk * 100).toFixed(1)}%
                    </div>
                </div>
            `;

            // Patient info (for PSV files)
            const patientInfo = document.getElementById('patientInfo');
            if (result.file_format === 'psv') {
                let infoHtml = '<div class="patient-info"><h4>👤 Patient Information</h4><ul>';
                infoHtml += '<li><strong>File Type:</strong> ICU Patient Data (PSV)</li>';
                infoHtml += '<li><strong>Processing:</strong> Converted to 536 STFT features</li>';
                infoHtml += '<li><strong>Signal Analysis:</strong> Time-frequency domain transformation</li>';
                infoHtml += '</ul></div>';
                patientInfo.innerHTML = infoHtml;
            } else {
                patientInfo.innerHTML = '';
            }
            
            // Display model predictions
            let modelsHtml = '<h4>🤖 Model Predictions:</h4><div class="models-grid">';
            
            if (result.best_ensemble_prediction !== undefined && result.best_ensemble_prediction !== null) {
                modelsHtml += `<div class="model-card">
                    <h5>🏆 Best Ensemble</h5>
                    <p>${(result.best_ensemble_prediction * 100).toFixed(1)}%</p>
                </div>`;
            }
            
            if (result.stacking_ensemble_prediction !== undefined && result.stacking_ensemble_prediction !== null) {
                modelsHtml += `<div class="model-card">
                    <h5>🔗 Stacking Ensemble</h5>
                    <p>${(result.stacking_ensemble_prediction * 100).toFixed(1)}%</p>
                </div>`;
            }
            
            if (result.voting_soft_prediction !== undefined && result.voting_soft_prediction !== null) {
                modelsHtml += `<div class="model-card">
                    <h5>🗳️ Voting Soft</h5>
                    <p>${(result.voting_soft_prediction * 100).toFixed(1)}%</p>
                </div>`;
            }
            
            if (result.adaboost_ensemble_prediction !== undefined && result.adaboost_ensemble_prediction !== null) {
                modelsHtml += `<div class="model-card">
                    <h5>⚡ AdaBoost</h5>
                    <p>${(result.adaboost_ensemble_prediction * 100).toFixed(1)}%</p>
                </div>`;
            }
            
            modelsHtml += '</div>';
            
            document.getElementById('modelPredictions').innerHTML = modelsHtml;
            
            results.style.display = 'block';
        }
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    print("🏥 ICU-Compatible Sepsis Prediction Dashboard")
    print("============================================")
    
    # Load models on startup
    load_models()
    
    print("🚀 Starting Flask server...")
    print("📍 Dashboard will be available at: http://localhost:5000")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)