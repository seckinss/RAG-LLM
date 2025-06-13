import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings

from src.utils.config import VECTOR_DB_DIR, DISTANCE_METRIC

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VectorStore:
    """Interface to the vector database for storing and retrieving document embeddings."""
    
    def __init__(self, db_path: str = VECTOR_DB_DIR, collection_name: str = "document_chunks"):
        """Initialize the vector store.
        
        Args:
            db_path: Path to the vector database
            collection_name: Name of the collection to use
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        
        # Create the directory if it doesn't exist
        os.makedirs(self.db_path, exist_ok=True)
        
        # Initialize the ChromaDB client
        logger.info(f"Connecting to ChromaDB at: {db_path}")
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # Get the collection (or create it if it doesn't exist)
        try:
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name
            )
            logger.info(f"Connected to existing collection: {collection_name} with {self.collection.count()} documents")
        except ValueError:
            logger.warning(f"Collection {collection_name} not found. Creating new collection.")
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": DISTANCE_METRIC}
            )
    
    def count(self) -> int:
        """Get the number of documents in the collection.
        
        Returns:
            Number of documents in the collection
        """
        return self.collection.count()
    
    def add(self, texts: List[str], ids: List[str], metadatas: List[Dict[str, Any]], embeddings: Optional[List[List[float]]] = None) -> None:
        """Add documents to the vector store.
        
        Args:
            texts: List of document texts
            ids: List of document IDs
            metadatas: List of document metadata
            embeddings: Optional list of pre-computed embeddings
        """
        self.collection.upsert(
            documents=texts,
            ids=ids,
            metadatas=metadatas,
            embeddings=embeddings
        )
        logger.info(f"Added {len(texts)} documents to the vector store")
    
    def similarity_search(
        self, 
        query: str,
        embedding: Optional[List[float]] = None,
        n_results: int = 3,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search for similar documents in the vector store.
        
        Args:
            query: Query text
            embedding: Optional pre-computed query embedding
            n_results: Number of results to return
            where: Optional filter to apply to the search
            
        Returns:
            Dictionary with search results
        """
        results = self.collection.query(
            query_texts=[query] if embedding is None else None,
            query_embeddings=[embedding] if embedding is not None else None,
            n_results=n_results,
            where=where
        )
        
        logger.info(f"Found {len(results['documents'][0]) if results['documents'] else 0} results for query: {query[:50]}...")
        return results
    
    def get(self, ids: List[str]) -> Dict[str, Any]:
        """Get documents by ID.
        
        Args:
            ids: List of document IDs
            
        Returns:
            Dictionary with matching documents
        """
        return self.collection.get(ids=ids)
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents from the vector store.
        
        Args:
            ids: List of document IDs to delete
        """
        self.collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents from the vector store")
    
    def reset(self) -> None:
        """Delete all documents from the collection."""
        self.chroma_client.delete_collection(name=self.collection_name)
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": DISTANCE_METRIC}
        )
        logger.info(f"Reset collection {self.collection_name}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection.
        
        Returns:
            Dictionary with collection metadata
        """
        count = self.collection.count()
        return {
            "name": self.collection_name,
            "count": count,
            "path": str(self.db_path),
        }
