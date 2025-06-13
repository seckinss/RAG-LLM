#!/usr/bin/env python3
"""
Run script for the RAG-based Q&A System.
This script provides a convenient way to run the different components of the system.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

# Add the current directory to Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Filter out env directories for watchfiles
ignore_dirs = [
    "venv",
    "rag_venv_clean"
]

def create_directories():
    """Create the necessary directories for the system."""
    # Define directories
    dirs = [
        "data",
        "data/raw",
        "data/processed",
        "data/vectordb",
        ".cache"
    ]
    
    # Create directories
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    print("✅ Directories created")

def process_documents():
    """Process the documents in the data/raw directory."""
    cmd = [sys.executable, "-m", "src.data_processing.preprocessor"]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=BASE_DIR))

def generate_embeddings():
    """Generate embeddings for the processed documents."""
    cmd = [sys.executable, "-m", "src.data_processing.embedder"]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=BASE_DIR))

def run_api():
    """Run the FastAPI backend."""
    base_cmd = ["uvicorn", "src.api.api:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
    
    # Add reload exclude options for directories in ignore_dirs
    # ignore_dirs is defined globally
    reload_excludes = []
    for d in ignore_dirs:
        reload_excludes.extend(["--reload-exclude", d])
    
    cmd = base_cmd + reload_excludes
    print(f"Running: {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=dict(os.environ, PYTHONPATH=BASE_DIR))

def run_webapp():
    """Run the Streamlit web application."""
    cmd = ["streamlit", "run", "webapp/app.py"]
    print(f"Running: {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=dict(os.environ, PYTHONPATH=BASE_DIR))

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run the RAG-based Q&A System")
    parser.add_argument("--setup", action="store_true", help="Setup the system")
    parser.add_argument("--process", action="store_true", help="Process documents")
    parser.add_argument("--embed", action="store_true", help="Generate embeddings")
    parser.add_argument("--api", action="store_true", help="Run the API")
    parser.add_argument("--webapp", action="store_true", help="Run the web application")
    parser.add_argument("--all", action="store_true", help="Run everything")
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Create directories
    if args.setup or args.all:
        create_directories()
    
    # Process documents
    if args.process or args.all:
        process_documents()
    
    # Generate embeddings
    if args.embed or args.all:
        generate_embeddings()
    
    # Run API and web app
    processes = []
    try:
        if args.api or args.all:
            api_process = run_api()
            processes.append(api_process)
            # Wait for API to start
            print("Waiting for API to start...")
            time.sleep(3)
        
        if args.webapp or args.all:
            webapp_process = run_webapp()
            processes.append(webapp_process)
        
        # Keep the script running until interrupted
        if processes:
            print("\n✅ Press Ctrl+C to stop the system\n")
            for p in processes:
                p.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        for p in processes:
            p.terminate()

if __name__ == "__main__":
    main() 