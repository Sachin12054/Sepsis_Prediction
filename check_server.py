#!/usr/bin/env python3
"""
Check if the dashboard server is ready
"""

import requests
import time
import sys

def check_server():
    for i in range(15):
        try:
            response = requests.get('http://localhost:5000/api/health', timeout=2)
            if response.status_code == 200:
                print('Backend server is ready and connected!')
                return True
        except:
            pass
        time.sleep(1)
    
    print('Server may still be starting, continuing...')
    return False

if __name__ == "__main__":
    check_server()