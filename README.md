# Lightweight RAG-based Q&A System

A question-answering system that uses Retrieval-Augmented Generation (RAG) to provide accurate answers based on a document corpus. This project combines vector similarity search with open-source LLMs to create a powerful, yet lightweight chatbot.

## Features

- Document preprocessing and chunking
- Embedding generation using sentence-transformers
- Vector storage with ChromaDB
- Semantic search for relevant document retrieval
- Local LLM integration for answer generation
- Web interface built with Streamlit
- FastAPI backend for processing requests

## Project Structure

```
.
├── data/                # Directory to store datasets
├── src/
│   ├── data_processing/ # Text extraction, chunking, and embedding
│   ├── retrieval/       # Vector database and retrieval logic
│   ├── models/          # LLM integration
│   ├── utils/           # Helper functions and configuration
│   └── api/             # FastAPI backend
├── webapp/              # Streamlit web application
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```


## Environment Variables 
# API Settings
```bash
# API_HOST would be our domain for Web App
API_HOST=
API_PORT=8000

# Web App Settings
WEBAPP_HOST=0.0.0.0
WEBAPP_PORT=8501

# LLM Model Settings
# Using Mistral-7B-Instruct-v0.3
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3

# Hugging Face Settings
# Get your token from: https://huggingface.co/settings/tokens
HUGGINGFACE_TOKEN=

# Vector Database Settings
# Change this to use a different distance metric
# Examples: cosine, l2, ip, for all-MiniLM-L6-v2 best would be cosine
DISTANCE_METRIC=cosine 


# NGROK Settings
# Use it for Colab API / Colab Web App
NGROK_AUTHTOKEN=
```

## Local Setup

1. Clone this repository:
```bash
git clone [repository-url]
cd rag-qa-system
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download and prepare a dataset (e.g., from Wikipedia, Project Gutenberg, etc.)
5. Run the data processing pipeline:
```bash
python -m src.data_processing.preprocessor --input_dir data/raw --output_dir data/processed
```

6. Generate embeddings and store them in ChromaDB:
```bash
python -m src.data_processing.embedder --input_dir data/processed --db_path data/vectordb
```

7. Start the FastAPI backend:
```bash
uvicorn src.api.api:app --reload
```

8. Launch the Streamlit web interface:
```bash
streamlit run webapp/app.py
```

## Colab API Setup

For running the pre-prepared RAG system in Google Colab with API access:

### 1. Upload and Extract Project

```python
# Upload the rag-llm.zip file to Colab
from google.colab import files
uploaded = files.upload()

# Extract the project
!unzip -q rag-llm.zip
%cd rag-llm
```

### 2. Install Dependencies

```python
# Install required packages
!pip install -r requirements.txt
```

### 3. Configure Authentication Tokens

Set up your authentication tokens in Colab secrets or environment variables:

```python
import os
from google.colab import userdata

# Set Hugging Face token for model access
os.environ["HUGGINGFACE_TOKEN"] = userdata.get('HUGGINGFACE_TOKEN')

# Set ngrok authtoken for stable tunneling (recommended)
os.environ["NGROK_AUTHTOKEN"] = userdata.get('NGROK_AUTHTOKEN')
```

To get these tokens:
- **Hugging Face Token**: Get from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- **Ngrok Token**: Get from [https://dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken)

### 4. Start the API

```python
# Run the API with ngrok tunneling (data preparation already included)
!python -m src.api.api
```

The API will automatically:
- Start a FastAPI server on the configured port
- Create an ngrok tunnel for external access
- Display the public URL for API access
- Load the pre-processed documents and embeddings

### 5. API Endpoints

Once running, you can access these endpoints:

- **Health Check**: `GET {public_url}/health`
- **Document Retrieval**: `POST {public_url}/retrieve`
- **Question Answering**: `POST {public_url}/query`

### 6. Example Usage

```python
import requests

# Replace with your ngrok public URL
API_URL = "https://your-ngrok-url.ngrok.io"

# Health check
response = requests.get(f"{API_URL}/health")
print("Health:", response.json())

# Ask a question
query_data = {
    "query": "What is the main topic of the documents?",
    "top_k": 5,
    "include_sources": True,
    "temperature": 0.7
}

response = requests.post(f"{API_URL}/query", json=query_data)
result = response.json()
print("Answer:", result["answer"])
print("Sources:", len(result["sources"]))
```

### 7. Alternative: Run Streamlit in Colab

```python
# Run Streamlit with ngrok tunneling
# I will run it on my local in demo but its an option
!streamlit run webapp/app.py &
!ngrok http 8501
```

### Notes for Colab Usage

- The system will use available GPU resources automatically if CUDA is available
- Model loading may take several minutes on first run
- Ngrok tunnels are temporary and will change on each restart, however public domain won't change (I have used free version of ngrok so I have just one public domain)
## Usage

1. Open the web interface at http://localhost:8501
2. Enter your question in the provided field
3. Review the answer generated by the system along with source references

## Implementation Details

- **Text Processing**: Documents are split into chunks of 512-1024 tokens
- **Embeddings**: Generated using a lightweight model (all-MiniLM-L6-v2)
- **Vector Database**: ChromaDB for efficient similarity search
- **LLM**: Uses an open-source LLM for response generation
- **Retrieval**: Top-k semantic search to find relevant context
- **Web Interface**: Interactive UI built with Streamlit

## Customization

- Change embedding model in `src/data_processing/embedder.py`
- Adjust chunk size in `src/data_processing/preprocessor.py`
- Configure LLM parameters in `src/models/llm.py`
- Modify retrieval settings in `src/retrieval/retriever.py`
- Change UI / UX in `webapp/app.py`
