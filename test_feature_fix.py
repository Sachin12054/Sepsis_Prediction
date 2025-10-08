#!/usr/bin/env python3
"""
Dashboard Feature Test
====================

Test the fixed feature handling in dashboard server
"""

import requests
import numpy as np
import json

def test_feature_fix():
    """Test the feature mismatch fix"""
    print("🔧 TESTING FEATURE MISMATCH FIX")
    print("=" * 40)
    
    base_url = "http://localhost:5000"
    
    # Test with 536 features (correct count)
    try:
        print("Testing with 536 features (correct)...")
        test_data = {
            "features": np.random.randn(3, 536).tolist()
        }
        
        response = requests.post(
            f"{base_url}/api/predict",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 'success':
                print("✅ 536 features: SUCCESS")
                print(f"   Processed {len(result['data']['results'])} patients")
            else:
                print(f"❌ 536 features: FAILED - {result.get('message')}")
        else:
            print(f"❌ 536 features: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error with 536 features: {e}")
    
    # Test with 532 features (old count) - should be auto-corrected
    try:
        print("\nTesting with 532 features (should auto-fix)...")
        test_data = {
            "features": np.random.randn(3, 532).tolist()
        }
        
        response = requests.post(
            f"{base_url}/api/predict",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 'success':
                print("✅ 532 features: AUTO-CORRECTED to 536")
                print(f"   Processed {len(result['data']['results'])} patients")
            else:
                print(f"❌ 532 features: FAILED - {result.get('message')}")
        else:
            print(f"❌ 532 features: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error with 532 features: {e}")
    
    print("\n🎉 Feature fix testing complete!")

if __name__ == '__main__':
    test_feature_fix()