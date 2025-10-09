#!/usr/bin/env python3
"""
🏥 ICU-Compatible Sepsis Prediction Dashboard
==========================================
Enhanced Flask server that handles both CSV test files and real ICU PSV patient data

Features:
- Auto-detects file format (CSV vs PSV)
- Processes real hospital ICU data
- Converts raw vital signs to 532 STFT features
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

# Simple CORS handling without flask-cors
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Global variables for models, scaler, and converter
models = {}
stft_scaler = None
icu_converter = None

def load_models():
    """Load all sepsis prediction models"""
    global models, stft_scaler, icu_converter
    
    print("🔄 Loading sepsis prediction models...")
    
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
    
    # Load STFT scaler
    try:
        if os.path.exists('models/advanced/stft_scaler.pkl'):
            stft_scaler = joblib.load('models/advanced/stft_scaler.pkl')
            print("   ✅ Loaded STFT feature scaler")
        else:
            print("   ⚠️  STFT scaler not found")
    except Exception as e:
        print(f"   ❌ Error loading STFT scaler: {e}")
    
    # Initialize ICU converter
    try:
        icu_converter = ICUDataConverter()
        print("   ✅ ICU data converter initialized")
    except Exception as e:
        print(f"   ❌ Error initializing ICU converter: {e}")
    
    if not models:
        print("   ⚠️  No models loaded successfully!")
    else:
        print(f"   🎯 {len(models)} models ready for predictions")

def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def detect_file_format(file_content: str) -> str:
    """Detect if file is CSV or PSV format and what type of data it contains"""
    lines = file_content.strip().split('\n')
    if len(lines) < 2:
        return 'unknown'
    
    header = lines[0]
    
    # Check for PSV format (ICU data has specific columns like ICULOS, SepsisLabel)
    if '|' in header and ('ICULOS' in header or 'SepsisLabel' in header or 'HospAdmTime' in header):
        return 'psv_icu'
    elif ',' in header:
        # Count columns to determine CSV type
        column_count = len(header.split(','))
        
        # Check if it's raw patient data (has common ICU columns)
        if any(col.strip() in ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp', 'pH', 'Age', 'Gender', 'ICULOS', 'SepsisLabel'] 
               for col in header.split(',')):
            return 'csv_raw'
        # Check if it's pre-processed STFT features (532 columns)
        elif column_count == 532:
            return 'csv_stft'
        else:
            # Default to raw data if not 532 columns
            return 'csv_raw'
    else:
        return 'unknown'

def process_csv_stft_file(file_content: str) -> dict:
    """Process CSV file with pre-processed STFT features (532 columns)"""
    try:
        # Read CSV content
        lines = file_content.strip().split('\n')
        header = lines[0].split(',')
        data_lines = lines[1:]
        
        print(f"📊 Processing CSV STFT file with {len(data_lines)} rows, {len(header)} columns")
        
        # Convert to DataFrame
        data_rows = []
        for line in data_lines:
            if line.strip():
                row = [float(x) if x.strip() and x.strip() != 'nan' else np.nan 
                       for x in line.split(',')]
                data_rows.append(row)
        
        if not data_rows:
            raise ValueError("No valid data rows found")
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=header)
        
        # Ensure we have 532 features
        if len(df.columns) != 532:
            raise ValueError(f"Expected 532 features, got {len(df.columns)}")
        
        # Use first row for prediction
        features = df.iloc[0].values.reshape(1, -1)
        
        return {
            'success': True,
            'features': features,
            'data_type': 'csv_stft',
            'rows_processed': len(data_rows),
            'patient_info': {'source': 'CSV STFT features file'}
        }
        
    except Exception as e:
        print(f"❌ Error processing CSV STFT: {e}")
        return {
            'success': False,
            'error': f"CSV STFT processing error: {str(e)}",
            'data_type': 'csv_stft'
        }

def process_csv_raw_file(file_content: str) -> dict:
    """Process CSV file with raw patient data (similar to PSV but comma-separated)"""
    try:
        # Save content to temporary file
        temp_file = 'temp_patient.csv'
        with open(temp_file, 'w') as f:
            f.write(file_content)
        
        print(f"🏥 Processing raw CSV patient file...")
        
        # Read CSV directly with pandas, ensuring numeric columns are float
        try:
            data = pd.read_csv(temp_file)
            
            # Convert numeric columns to float to handle NaN values properly
            numeric_columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
                             'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
                             'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
                             'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
                             'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
                             'Fibrinogen', 'Platelets', 'Age', 'ICULOS']
            
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")
        
        print(f"   📊 Loaded {len(data)} time points with {len(data.columns)} variables")
        
        # Extract patient info
        info = icu_converter._extract_patient_info(data)
        info['data_type'] = 'CSV (raw patient data)'
        
        # Process vital signs using ICU converter
        processed_signals = icu_converter._process_vital_signs(data)
        
        # Generate STFT features
        features_dict = icu_converter._generate_stft_features(processed_signals)
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        if not features_dict:
            raise ValueError("No features generated from raw CSV data")
        
        # Convert features to array format expected by models
        feature_values = list(features_dict.values())
        features = np.array(feature_values).reshape(1, -1)
        
        print(f"   ✅ Generated {len(feature_values)} features for sepsis prediction")
        
        return {
            'success': True,
            'features': features,
            'data_type': 'csv_raw',
            'patient_info': info,
            'feature_count': len(feature_values)
        }
        
    except Exception as e:
        print(f"❌ Error processing raw CSV: {e}")
        traceback.print_exc()
        return {
            'success': False,
            'error': f"Raw CSV processing error: {str(e)}",
            'data_type': 'csv_raw'
        }

def process_psv_icu_file(file_content: str) -> dict:
    """Process PSV ICU patient file using our converter"""
    try:
        # Save content to temporary file
        temp_file = 'temp_patient.psv'
        with open(temp_file, 'w') as f:
            f.write(file_content)
        
        print(f"🏥 Processing ICU PSV file...")
        
        # Process with ICU converter
        result = icu_converter.process_icu_patient_file(temp_file)
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        if not result['features']:
            raise ValueError("No features generated from ICU data")
        
        # Convert features to array format expected by models
        feature_values = list(result['features'].values())
        features = np.array(feature_values).reshape(1, -1)
        
        return {
            'success': True,
            'features': features,
            'data_type': 'psv_icu',
            'patient_info': result['patient_info'],
            'feature_count': len(feature_values)
        }
        
    except Exception as e:
        print(f"❌ Error processing ICU PSV: {e}")
        traceback.print_exc()
        return {
            'success': False,
            'error': f"ICU data processing error: {str(e)}",
            'data_type': 'psv_icu'
        }

@app.route('/')
def home():
    """Serve the enhanced dashboard"""
    return render_template_string('''
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
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
            padding: 40px;
            max-width: 800px;
            width: 100%;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        
        .header h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }
        
        .header p {
            color: #7f8c8d;
            font-size: 1.1em;
        }
        
        .upload-section {
            margin-bottom: 30px;
        }
        
        .file-input-wrapper {
            position: relative;
            display: inline-block;
            width: 100%;
        }
        
        .file-input {
            width: 100%;
            padding: 15px;
            border: 2px dashed #3498db;
            border-radius: 10px;
            background: #f8f9fa;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .file-input:hover {
            border-color: #2980b9;
            background: #e3f2fd;
        }
        
        .file-input input[type="file"] {
            position: absolute;
            opacity: 0;
            width: 100%;
            height: 100%;
            cursor: pointer;
        }
        
        .predict-btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(45deg, #3498db, #2980b9);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 20px;
        }
        
        .predict-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(52, 152, 219, 0.3);
        }
        
        .predict-btn:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .results {
            margin-top: 30px;
            padding: 20px;
            border-radius: 10px;
            display: none;
        }
        
        .results.show {
            display: block;
        }
        
        .results.sepsis {
            background: linear-gradient(45deg, #e74c3c, #c0392b);
            color: white;
        }
        
        .results.no-sepsis {
            background: linear-gradient(45deg, #27ae60, #229954);
            color: white;
        }
        
        .results.error {
            background: linear-gradient(45deg, #f39c12, #e67e22);
            color: white;
        }
        
        .risk-score {
            font-size: 3em;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
        }
        
        .patient-info {
            background: rgba(255, 255, 255, 0.2);
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
        }
        
        .model-predictions {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .model-card {
            background: rgba(255, 255, 255, 0.2);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        
        .loading.show {
            display: block;
        }
        
        .file-info {
            margin-top: 10px;
            padding: 10px;
            background: #e8f5e8;
            border-radius: 5px;
            font-size: 0.9em;
            color: #2d5f3f;
        }
        
        .supported-formats {
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 ICU Sepsis Prediction</h1>
            <p>Advanced AI-powered sepsis detection for clinical decision support</p>
        </div>
        
        <div class="upload-section">
            <div class="file-input-wrapper">
                <div class="file-input">
                    <input type="file" id="fileInput" accept=".csv,.psv,.txt" />
                    <p>📁 Click to upload patient data (CSV with STFT features, CSV with raw data, or PSV ICU files)</p>
                </div>
            </div>
            <div id="fileInfo" class="file-info" style="display: none;"></div>
        </div>
        
        <button class="predict-btn" id="predictBtn" onclick="predictSepsis()" disabled>
            🔬 Analyze for Sepsis Risk
        </button>
        
        <div class="loading" id="loading">
            <p>🔄 Processing patient data...</p>
        </div>
        
        <div class="results" id="results">
            <div class="risk-score" id="riskScore"></div>
            <div id="predictionText"></div>
            <div class="patient-info" id="patientInfo"></div>
            <div class="model-predictions" id="modelPredictions"></div>
        </div>
        
        <div class="supported-formats">
            <h3>📋 Supported Data Formats:</h3>
            <ul>
                <li><strong>CSV with STFT Features:</strong> Pre-processed 532 STFT features</li>
                <li><strong>CSV with Raw Data:</strong> Raw patient vital signs (HR, O2Sat, Temp, SBP, etc.)</li>
                <li><strong>PSV ICU Files:</strong> Pipe-separated ICU patient data</li>
                <li><strong>Auto-Detection:</strong> System automatically detects and processes file format</li>
            </ul>
        </div>
    </div>

    <script>
        let selectedFile = null;
        
        document.getElementById('fileInput').addEventListener('change', function(e) {
            selectedFile = e.target.files[0];
            const fileInfo = document.getElementById('fileInfo');
            const predictBtn = document.getElementById('predictBtn');
            
            if (selectedFile) {
                fileInfo.innerHTML = `
                    <strong>📄 File Selected:</strong> ${selectedFile.name} (${(selectedFile.size / 1024).toFixed(1)} KB)<br>
                    <strong>🔍 Type:</strong> ${selectedFile.name.endsWith('.psv') ? 'ICU Patient Data (PSV)' : 'Test Data (CSV)'}
                `;
                fileInfo.style.display = 'block';
                predictBtn.disabled = false;
            } else {
                fileInfo.style.display = 'none';
                predictBtn.disabled = true;
            }
        });
        
        async function predictSepsis() {
            if (!selectedFile) {
                alert('Please select a patient data file first');
                return;
            }
            
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');
            const predictBtn = document.getElementById('predictBtn');
            
            // Show loading
            loading.classList.add('show');
            results.classList.remove('show');
            predictBtn.disabled = true;
            
            try {
                // Read file content
                const fileContent = await readFileAsText(selectedFile);
                
                // Send to server
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        file_content: fileContent,
                        filename: selectedFile.name
                    })
                });
                
                const result = await response.json();
                
                // Hide loading
                loading.classList.remove('show');
                
                // Show results
                displayResults(result);
                
            } catch (error) {
                loading.classList.remove('show');
                displayError(`Error processing file: ${error.message}`);
            } finally {
                predictBtn.disabled = false;
            }
        }
        
        function readFileAsText(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = (e) => resolve(e.target.result);
                reader.onerror = (e) => reject(new Error('Failed to read file'));
                reader.readAsText(file);
            });
        }
        
        function displayResults(result) {
            const results = document.getElementById('results');
            const riskScore = document.getElementById('riskScore');
            const predictionText = document.getElementById('predictionText');
            const patientInfo = document.getElementById('patientInfo');
            const modelPredictions = document.getElementById('modelPredictions');
            
            if (result.error) {
                results.className = 'results show error';
                riskScore.textContent = '⚠️ Error';
                predictionText.innerHTML = `<h3>Processing Error</h3><p>${result.error}</p>`;
                patientInfo.innerHTML = '';
                modelPredictions.innerHTML = '';
                return;
            }
            
            // Determine risk level
            const sepsisRisk = result.primary_prediction || result.best_ensemble_prediction || 0;
            const isHighRisk = sepsisRisk >= 0.5;
            
            // Set result style
            results.className = `results show ${isHighRisk ? 'sepsis' : 'no-sepsis'}`;
            
            // Display risk score
            riskScore.textContent = `${(sepsisRisk * 100).toFixed(1)}%`;
            
            // Display prediction text
            predictionText.innerHTML = `
                <h3>${isHighRisk ? '🚨 HIGH SEPSIS RISK' : '✅ LOW SEPSIS RISK'}</h3>
                <p>${isHighRisk ? 
                    'Immediate clinical attention recommended. Consider sepsis protocols.' : 
                    'Patient shows low sepsis risk based on current data.'}</p>
            `;
            
            // Display patient info
            if (result.patient_info) {
                const info = result.patient_info;
                let infoHtml = '<h4>👤 Patient Information:</h4><ul>';
                
                if (info.age !== undefined) infoHtml += `<li>Age: ${info.age}</li>`;
                if (info.gender !== undefined) infoHtml += `<li>Gender: ${info.gender === 0 ? 'Female' : 'Male'}</li>`;
                if (info.icu_length !== undefined) infoHtml += `<li>ICU Hours: ${info.icu_length}</li>`;
                if (info.data_type) infoHtml += `<li>Data Type: ${info.data_type}</li>`;
                if (info.sepsis_detected !== undefined) {
                    infoHtml += `<li>Historical Sepsis: ${info.sepsis_detected ? 'Yes' : 'No'}</li>`;
                }
                
                infoHtml += '</ul>';
                patientInfo.innerHTML = infoHtml;
            }
            
            // Display model predictions
            let modelsHtml = '<h4>🤖 Model Predictions:</h4>';
            if (result.best_ensemble_prediction !== undefined) {
                modelsHtml += `<div class="model-card">
                    <h5>� Best Ensemble</h5>
                    <p>${(result.best_ensemble_prediction * 100).toFixed(1)}%</p>
                </div>`;
            }
            if (result.stacking_ensemble_prediction !== undefined) {
                modelsHtml += `<div class="model-card">
                    <h5>📚 Stacking Model</h5>
                    <p>${(result.stacking_ensemble_prediction * 100).toFixed(1)}%</p>
                </div>`;
            }
            if (result.voting_soft_prediction !== undefined) {
                modelsHtml += `<div class="model-card">
                    <h5>🗳️ Voting Soft</h5>
                    <p>${(result.voting_soft_prediction * 100).toFixed(1)}%</p>
                </div>`;
            }
            if (result.adaboost_prediction !== undefined) {
                modelsHtml += `<div class="model-card">
                    <h5>� AdaBoost</h5>
                    <p>${(result.adaboost_prediction * 100).toFixed(1)}%</p>
                </div>`;
            }
            
            modelPredictions.innerHTML = modelsHtml;
            
            // Show results
            results.classList.add('show');
        }
        
        function displayError(message) {
            const results = document.getElementById('results');
            const riskScore = document.getElementById('riskScore');
            const predictionText = document.getElementById('predictionText');
            
            results.className = 'results show error';
            riskScore.textContent = '❌';
            predictionText.innerHTML = `<h3>Error</h3><p>${message}</p>`;
            results.classList.add('show');
        }
    </script>
</body>
</html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    """Enhanced prediction endpoint that handles both CSV and PSV files"""
    try:
        data = request.json
        file_content = data.get('file_content', '')
        filename = data.get('filename', '')
        
        print(f"\n🔍 Received file: {filename}")
        
        # Detect file format
        file_format = detect_file_format(file_content)
        print(f"   📋 Detected format: {file_format}")
        
        # Process based on format
        if file_format == 'csv_stft':
            process_result = process_csv_stft_file(file_content)
        elif file_format == 'csv_raw':
            process_result = process_csv_raw_file(file_content)
        elif file_format == 'psv_icu':
            process_result = process_psv_icu_file(file_content)
        else:
            return jsonify({
                'error': f'Unsupported file format. Expected CSV or PSV, got: {file_format}',
                'supported_formats': [
                    'CSV with 532 STFT features', 
                    'CSV with raw patient data', 
                    'PSV (pipe-separated ICU data)'
                ]
            })
        
        if not process_result['success']:
            return jsonify(process_result)
        
        # Get features and run predictions
        features = process_result['features']
        print(f"   🎯 Running predictions with {features.shape[1]} features...")
        
        # Scale features if scaler is available
        if stft_scaler is not None:
            try:
                features_scaled = stft_scaler.transform(features)
                print(f"   📊 Features scaled using STFT scaler")
                features = features_scaled
            except Exception as e:
                print(f"   ⚠️  Scaling failed, using raw features: {e}")
        
        predictions = {}
        
        # Run all available models
        for model_name, model in models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(features)[0]
                    if len(pred_proba) > 1:
                        predictions[f'{model_name}_prediction'] = float(pred_proba[1])
                    else:
                        predictions[f'{model_name}_prediction'] = float(pred_proba[0])
                else:
                    pred = model.predict(features)[0]
                    predictions[f'{model_name}_prediction'] = float(pred)
                    
                print(f"     ✅ {model_name}: {predictions[f'{model_name}_prediction']:.3f}")
            except Exception as e:
                print(f"     ❌ {model_name} failed: {e}")
        
        # Combine results
        result = {
            'success': True,
            'data_type': process_result['data_type'],
            'patient_info': process_result.get('patient_info', {}),
            **predictions
        }
        
        # Use best ensemble as primary prediction
        primary_prediction = result.get('best_ensemble_prediction', 
                                      result.get('stacking_ensemble_prediction', 0))
        result['primary_prediction'] = primary_prediction
        
        print(f"   🎯 Primary prediction: {primary_prediction:.3f}")
        
        # Convert NumPy types to native Python types for JSON serialization
        result = convert_numpy_types(result)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        traceback.print_exc()
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False
        })

@app.route('/health')
def health():
    """Enhanced health check"""
    try:
        status = {
            'status': 'healthy',
            'models_loaded': len(models),
            'icu_converter': icu_converter is not None,
            'supported_formats': [
                'CSV (532 STFT features)', 
                'CSV (raw patient data)', 
                'PSV (ICU data)'
            ],
            'available_models': list(models.keys())
        }
        
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        })

if __name__ == '__main__':
    # Load models and converter on startup
    load_models()
    
    print("\n🏥 ICU-Compatible Sepsis Prediction Dashboard")
    print("=" * 60)
    print("🔗 Dashboard: http://localhost:5001")
    print("🩺 Health Check: http://localhost:5001/health")
    print("📡 API Endpoint: http://localhost:5001/predict")
    print()
    print("📋 Supported Data Formats:")
    print("   • CSV: Pre-processed 532 STFT features")
    print("   • CSV: Raw patient vital signs")
    print("   • PSV: Raw ICU patient vital signs")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5001)