#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

PROJECT_ZIP_FILE="rag_project.zip"
PROJECT_DIRECTORY_NAME="rag-llm" # Assuming this is the root folder name inside the zip

# --- 1. Unzip the Project ---
if [ ! -f "$PROJECT_ZIP_FILE" ]; then
    echo "❌ Error: Project zip file '$PROJECT_ZIP_FILE' not found!"
    echo "Please make sure the zip file is in the same directory as this script."
    exit 1
fi

echo "🚀 Unzipping project from $PROJECT_ZIP_FILE..."
unzip -q "$PROJECT_ZIP_FILE" # -q for quiet
echo "✅ Project unzipped."

# --- 2. Navigate into Project Directory ---
if [ ! -d "$PROJECT_DIRECTORY_NAME" ]; then
    echo "❌ Error: Project directory '$PROJECT_DIRECTORY_NAME' not found after unzipping."
    echo "Please check the contents of your zip file. The root folder should be named '$PROJECT_DIRECTORY_NAME'."
    exit 1
fi
cd "$PROJECT_DIRECTORY_NAME"
echo "📂 Changed directory to $(pwd)"

# --- 3. Create and Activate Virtual Environment ---
VENV_NAME="venv"
if [ ! -d "$VENV_NAME" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv "$VENV_NAME"
    echo "✅ Virtual environment '$VENV_NAME' created."
else
    echo "🐍 Virtual environment '$VENV_NAME' already exists."
fi

echo "🚀 Activating virtual environment..."
source "$VENV_NAME/bin/activate"
echo "✅ Virtual environment activated."

# --- 4. Install Dependencies ---
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: 'requirements.txt' not found in project directory."
    deactivate
    exit 1
fi
echo "🛠️ Installing dependencies from requirements.txt..."
pip install -r requirements.txt
echo "✅ Dependencies installed."

# --- 5. Run the Application ---
# Your run.py script handles directory creation, processing, embedding, and running services.
# We'll use the --all flag to run everything.
# Adjust this command if you only want to run specific parts (e.g., --api --webapp).

# Ensure HUGGINGFACEHUB_API_TOKEN is set (example for Colab/Linux, adjust if needed)
# For local use, you might have it in a .env file which run.py should load.
# If running in an environment where .env is not automatically picked up by python-dotenv for subprocesses,
# you might need to export it or ensure it's loaded.
# For Colab, you would set this using Colab secrets and os.environ in your Python code.

# if [ -z "$HUGGINGFACEHUB_API_TOKEN" ]; then
#   echo "⚠️ Warning: HUGGINGFACEHUB_API_TOKEN is not set. The LLM might fail to load."
#   echo "Please set it as an environment variable or ensure your .env file is correctly configured."
# fi

echo "⚙️ Running project setup (directories, data processing, embeddings) and services via run.py --all..."
# The run.py script will handle PYTHONPATH adjustments if necessary.
python3 run.py --api

# --- Deactivate virtual environment upon exit (optional, as script exit will do this) ---
echo "🔴 Deactivating virtual environment."
deactivate

echo "🎉 Script finished." 