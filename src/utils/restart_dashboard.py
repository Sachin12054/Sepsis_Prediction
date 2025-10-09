#!/usr/bin/env python3
"""
Restart Dashboard Server
========================

Stop any running dashboard server and start a fresh one with the latest code
"""

import os
import sys
import subprocess
import time
import psutil

def kill_existing_servers():
    """Kill any existing dashboard servers"""
    killed = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline'] and any('dashboard_server' in cmd for cmd in proc.info['cmdline']):
                print(f"Killing existing server process {proc.info['pid']}")
                proc.kill()
                killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if killed > 0:
        print(f"✅ Killed {killed} existing server(s)")
        time.sleep(2)  # Wait for processes to fully terminate
    else:
        print("No existing servers found")

def start_fresh_server():
    """Start a fresh dashboard server"""
    print("🚀 Starting fresh dashboard server...")
    
    # Start the server
    subprocess.Popen([
        sys.executable, 'dashboard_server.py'
    ], cwd=os.getcwd())
    
    print("✅ Dashboard server started!")
    print("🌐 Dashboard: http://localhost:5000")
    print("🔗 Health check: http://localhost:5000/api/health")

def main():
    """Main function"""
    print("🔄 RESTARTING DASHBOARD SERVER")
    print("=" * 40)
    
    # Kill existing servers
    kill_existing_servers()
    
    # Start fresh server
    start_fresh_server()
    
    print("\n✅ Server restart complete!")
    print("The latest code with feature fix is now active.")

if __name__ == '__main__':
    main()