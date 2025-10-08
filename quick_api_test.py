#!/usr/bin/env python3
"""
Quick API Test
==============

Test the API endpoints quickly
"""

import requests
import numpy as np
import json
import time

def test_api():
    """Test the API"""
    print("🧪 TESTING FIXED DASHBOARD API")
    print("=" * 40)
    
    base_url = "http://localhost:5000"
    
    # Wait for server to be ready
    print("Waiting for server...")
    time.sleep(3)
    
    # Test with 532 features (should auto-pad to 536)
    print("\n🔧 Testing with 532 features (should auto-pad)...")
    try:
        test_data = {
            "features": np.random.randn(3, 532).tolist()
        }
        
        response = requests.post(
            f"{base_url}/api/predict",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! Feature padding worked!")
            print(f"Processed {len(result['data']['results'])} patients")
        else:
            print(f"❌ FAILED: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test with 536 features (correct count)
    print("\n✅ Testing with 536 features (correct count)...")
    try:
        test_data = {
            "features": np.random.randn(3, 536).tolist()
        }
        
        response = requests.post(
            f"{base_url}/api/predict",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! Correct features worked!")
            print(f"Processed {len(result['data']['results'])} patients")
        else:
            print(f"❌ FAILED: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n🎉 API test complete!")

if __name__ == '__main__':
    test_api()