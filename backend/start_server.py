#!/usr/bin/env python3
"""
Startup script for the backend server with dependency management
"""
import os
import sys
import subprocess

def check_and_install_dependencies():
    """Check and install required dependencies"""
    try:
        import spacy
        print(f"SpaCy version: {spacy.__version__}")
        
        # Check if spaCy version is compatible
        if spacy.__version__.startswith('3.7.2'):
            print("Upgrading spaCy to compatible version...")
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "spacy==3.7.4"], check=True)
            
    except ImportError:
        print("SpaCy not found, installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "spacy==3.7.4"], check=True)
    
    try:
        import pydantic
        print(f"Pydantic version: {pydantic.__version__}")
    except ImportError:
        print("Pydantic not found, installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pydantic==2.5.0"], check=True)

def start_server():
    """Start the FastAPI server"""
    try:
        # Set environment variables for better compatibility
        os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))
        
        # Import and start the server
        import uvicorn
        uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
        
    except Exception as e:
        print(f"Error starting server: {e}")
        print("Trying to start with basic configuration...")
        
        # Fallback: start without ML dependencies
        os.environ['DISABLE_ML'] = 'true'
        import uvicorn
        uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    print("Checking dependencies...")
    check_and_install_dependencies()
    
    print("Starting server...")
    start_server()