#!/usr/bin/env python3
"""
🧪 Quick Dashboard Test with Real ICU Patient
============================================
Simple test to verify the dashboard is working with real patient data
"""

import requests
import json
import os

def test_dashboard_with_icu_data():
    """Test the dashboard with real ICU patient data"""
    
    print("🧪 Testing ICU-Compatible Sepsis Dashboard")
    print("=" * 60)
    
    # Test health endpoint
    try:
        print("🩺 Testing health endpoint...")
        health_response = requests.get("http://localhost:5000/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print("   ✅ Dashboard is healthy!")
            print(f"   📊 Models loaded: {health_data.get('models_loaded', 0)}")
            print(f"   🔧 ICU converter: {health_data.get('icu_converter', False)}")
        else:
            print(f"   ❌ Health check failed: {health_response.status_code}")
            return
    except Exception as e:
        print(f"   ❌ Cannot connect to dashboard: {e}")
        print("   🔧 Make sure the server is running at http://localhost:5000")
        return
    
    # Test with real ICU patient data
    try:
        print("\n🏥 Testing with real ICU patient p000001...")
        
        # Read the patient file
        patient_file = "data/raw/training_setA (1)/p000001.psv"
        if not os.path.exists(patient_file):
            print(f"   ❌ Patient file not found: {patient_file}")
            return
        
        with open(patient_file, 'r') as f:
            file_content = f.read()
        
        # Send prediction request
        prediction_data = {
            'file_content': file_content,
            'filename': 'p000001.psv'
        }
        
        print("   📤 Sending patient data to dashboard...")
        pred_response = requests.post(
            "http://localhost:5000/predict", 
            json=prediction_data,
            timeout=30
        )
        
        if pred_response.status_code == 200:
            result = pred_response.json()
            
            if result.get('success'):
                print("   ✅ Prediction successful!")
                
                # Display results
                primary_risk = result.get('primary_prediction', 0)
                risk_level = "HIGH RISK 🚨" if primary_risk >= 0.5 else "LOW RISK ✅"
                
                print(f"\n📊 SEPSIS RISK ASSESSMENT:")
                print(f"   🎯 Primary Risk: {primary_risk:.3f} ({primary_risk*100:.1f}%)")
                print(f"   📋 Risk Level: {risk_level}")
                
                # Patient info
                patient_info = result.get('patient_info', {})
                print(f"\n👤 Patient Information:")
                print(f"   Age: {patient_info.get('age', 'Unknown')}")
                print(f"   Gender: {'Female' if patient_info.get('gender') == 0 else 'Male'}")
                print(f"   ICU Hours: {patient_info.get('icu_length', 'Unknown')}")
                print(f"   Data Type: {result.get('data_type', 'Unknown')}")
                
                # Model predictions
                print(f"\n🤖 Model Predictions:")
                for key, value in result.items():
                    if key.endswith('_prediction'):
                        model_name = key.replace('_prediction', '').replace('_', ' ').title()
                        risk_indicator = "🚨" if value >= 0.5 else "✅"
                        print(f"   {model_name:15}: {value:.3f} ({value*100:4.1f}%) {risk_indicator}")
                
                print(f"\n✅ Dashboard test completed successfully!")
                print(f"🔗 Access dashboard at: http://localhost:5000")
                
            else:
                print(f"   ❌ Prediction failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"   ❌ Request failed: {pred_response.status_code}")
            try:
                error_data = pred_response.json()
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   Response: {pred_response.text[:200]}...")
                
    except Exception as e:
        print(f"   ❌ Test failed: {e}")

if __name__ == "__main__":
    test_dashboard_with_icu_data()