import os
import json
import logging
import argparse
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.utils.config import (
    EMBEDDING_MODEL_NAME,
    PROCESSED_DATA_DIR,
    VECTOR_DB_DIR,
    DISTANCE_METRIC
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentEmbedder:
    """Class for generating embeddings from processed documents and storing them in a vector database."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, db_path: str = VECTOR_DB_DIR):
        """Initialize the document embedder.
        
        Args:
            model_name: Name of the embedding model to use
            db_path: Path to the vector database
        """
        self.model_name = model_name
        self.db_path = Path(db_path)
        
        # Create the directory if it doesn't exist
        os.makedirs(self.db_path, exist_ok=True)
        
        # Initialize the embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Initialize the vector database
        logger.info(f"Initializing ChromaDB at: {db_path}")
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create a collection for the documents if it doesn't exist
        self.collection = self.chroma_client.get_or_create_collection(
            name="document_chunks",
            metadata={"hnsw:space": DISTANCE_METRIC}
        )
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        return self.model.encode(texts).tolist()
    
    def load_processed_documents(self, processed_dir: str) -> List[Dict[str, Any]]:
        """Load processed document chunks from JSON files.
        
        Args:
            processed_dir: Directory containing processed document chunks
            
        Returns:
            List of document chunks with metadata
        """
        processed_dir = Path(processed_dir)
        all_chunks = []
        
        # Find all JSON files in the processed directory
        json_files = list(processed_dir.glob("**/*.json"))
        logger.info(f"Found {len(json_files)} processed document files")
        
        for json_file in tqdm(json_files, desc="Loading processed documents"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                all_chunks.extend(chunks)
                logger.debug(f"Loaded {len(chunks)} chunks from {json_file}")
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
        
        logger.info(f"Loaded a total of {len(all_chunks)} chunks")
        return all_chunks
    
    def embed_and_store(self, processed_dir: str, batch_size: int = 32) -> None:
        """Embed document chunks and store them in the vector database.
        
        Args:
            processed_dir: Directory containing processed document chunks
            batch_size: Number of chunks to process at once
        """
        # Load all processed document chunks
        chunks = self.load_processed_documents(processed_dir)
        
        if not chunks:
            logger.warning("No chunks found to embed.")
            return
        
        # Process chunks in batches to avoid memory issues
        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding and storing chunks"):
            batch = chunks[i:i + batch_size]
            
            # Extract the text, ids, and metadata
            texts = [chunk["text"] for chunk in batch]
            ids = [f"{chunk['metadata']['filename']}_{chunk['chunk_id']}" for chunk in batch]
            metadatas = [
                {
                    **chunk["metadata"],
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"]
                }
                for chunk in batch
            ]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Add to the collection
            try:
                self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=texts
                )
                logger.debug(f"Added batch of {len(batch)} embeddings to collection")
            except Exception as e:
                logger.error(f"Error adding embeddings to collection: {e}")
        
        # Log the total count in the collection
        logger.info(f"Total documents in collection: {self.collection.count()}")

def main():
    """Main function to run the embedder from the command line."""
    parser = argparse.ArgumentParser(description="Generate embeddings for processed documents")
    parser.add_argument("--input_dir", type=str, default=PROCESSED_DATA_DIR,
                        help="Directory containing processed document chunks")
    parser.add_argument("--db_path", type=str, default=VECTOR_DB_DIR,
                        help="Path to the vector database")
    parser.add_argument("--model_name", type=str, default=EMBEDDING_MODEL_NAME,
                        help="Name of the embedding model to use")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of chunks to process at once")
    
    args = parser.parse_args()
    
    # Create embedder and process the documents
    embedder = DocumentEmbedder(
        model_name=args.model_name,
        db_path=args.db_path
    )
    
    embedder.embed_and_store(args.input_dir, args.batch_size)
    logger.info("Embedding generation complete")

if __name__ == "__main__":
    main()
