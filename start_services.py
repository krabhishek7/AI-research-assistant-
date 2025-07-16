#!/usr/bin/env python3
"""
Start script for the Academic Research Assistant
Launches both backend API and frontend Gradio interface
"""

import subprocess
import sys
import os
import time
import signal
from threading import Thread

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting FastAPI backend server...")
    
    # Set environment variables
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))
    
    # Start backend
    backend_cmd = [
        sys.executable, '-c', 
        """
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.api.main import app
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
"""
    ]
    
    backend_process = subprocess.Popen(
        backend_cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    return backend_process

def start_frontend():
    """Start the Gradio frontend"""
    print("🎨 Starting Gradio frontend...")
    
    # Set environment variables
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))
    
    # Start frontend
    frontend_cmd = [
        sys.executable, '-c',
        """
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frontend.app import create_interface
demo = create_interface()
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    debug=True,
    quiet=False
)
"""
    ]
    
    frontend_process = subprocess.Popen(
        frontend_cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    return frontend_process

def check_health():
    """Check if services are healthy"""
    import requests
    try:
        # Check backend health
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend API is healthy")
        else:
            print(f"❌ Backend API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend API not accessible: {e}")
        return False
    
    try:
        # Check frontend
        response = requests.get("http://localhost:7860", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend is accessible")
        else:
            print(f"❌ Frontend not accessible: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Frontend not accessible: {e}")
        return False
    
    return True

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\n🛑 Shutting down services...")
    sys.exit(0)

def main():
    """Main function to start all services"""
    print("🔬 Academic Research Assistant - Starting Services")
    print("=" * 60)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start backend in thread
    backend_thread = Thread(target=start_backend)
    backend_thread.daemon = True
    backend_thread.start()
    
    # Wait for backend to start
    print("⏳ Waiting for backend to start...")
    time.sleep(5)
    
    # Start frontend in thread
    frontend_thread = Thread(target=start_frontend)
    frontend_thread.daemon = True
    frontend_thread.start()
    
    # Wait for frontend to start
    print("⏳ Waiting for frontend to start...")
    time.sleep(5)
    
    # Check health
    if check_health():
        print("\n🎉 All services are running successfully!")
        print("\n📍 Access URLs:")
        print("   • Frontend (Gradio): http://localhost:7860")
        print("   • Backend API: http://localhost:8000")
        print("   • API Documentation: http://localhost:8000/docs")
        print("\n🔍 Available Features:")
        print("   • Search Papers across ArXiv and local database")
        print("   • Get detailed paper information")
        print("   • Find related papers using semantic search")
        print("   • Compare multiple papers")
        print("   • Generate citations in multiple formats")
        print("   • System statistics and health monitoring")
        print("\n✨ The Academic Research Assistant is ready to use!")
        print("   Press Ctrl+C to stop all services")
        
        # Keep the script running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Services stopped by user")
    else:
        print("\n❌ Failed to start services properly")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 