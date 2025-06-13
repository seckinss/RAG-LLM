import logging
from typing import List, Dict, Any, Optional
import time

from sentence_transformers import SentenceTransformer

from src.retrieval.vector_store import VectorStore
from src.utils.config import (
    EMBEDDING_MODEL_NAME,
    VECTOR_DB_DIR,
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Retriever:
    """Class for retrieving relevant documents for a query."""
    
    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        vector_db_path: str = VECTOR_DB_DIR,
        top_k: int = TOP_K_RESULTS,
        similarity_threshold: float = SIMILARITY_THRESHOLD
    ):
        """Initialize the retriever.
        
        Args:
            embedding_model_name: Name of the embedding model to use
            vector_db_path: Path to the vector database
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score for retrieval
        """
        self.embedding_model_name = embedding_model_name
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        # Initialize the embedding model
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize the vector store
        self.vector_store = VectorStore(db_path=vector_db_path)
        logger.info(f"Vector store initialized with {self.vector_store.count()} documents")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for a text string.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        return self.embedding_model.encode(text).tolist()
    
    def retrieve(
        self, 
        query: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            top_k: Optional override for number of results
            where: Optional filter to apply to the search
            
        Returns:
            List of retrieved documents with metadata and similarity scores
        """
        start_time = time.time()
        
        # Generate embedding for the query
        embedding = self.generate_embedding(query)
        
        # Set the number of results to retrieve
        k = top_k if top_k is not None else self.top_k
        
        # Perform the similarity search
        results = self.vector_store.similarity_search(
            query=query,
            embedding=embedding,
            n_results=k * 2,  # Retrieve more results than needed for filtering
            where=where
        )
        
        # Extract the documents, metadatas, and scores
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        # Check if we have results
        if not documents:
            logger.warning(f"No results found for query: {query[:50]}...")
            return []
        
        # Convert distances to similarity scores (assuming cosine distance)
        similarity_scores = [1 - dist for dist in distances]
        
        # Filter results by similarity threshold and take top-k
        filtered_results = []
        for doc, meta, score in zip(documents, metadatas, similarity_scores):
            if score >= self.similarity_threshold:
                filtered_results.append({
                    "text": doc,
                    "metadata": meta,
                    "similarity_score": score
                })
        
        # Sort by similarity score and take top-k
        filtered_results = sorted(filtered_results, key=lambda x: x["similarity_score"], reverse=True)[:k]
        
        elapsed_time = time.time() - start_time
        logger.info(f"Retrieved {len(filtered_results)} documents in {elapsed_time:.4f}s")
        
        return filtered_results
    
    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """Format the retrieved documents into a context string.
        
        Args:
            results: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant information found."
        
        context_parts = []
        
        for i, doc in enumerate(results):
            meta = doc["metadata"]
            source = meta.get("source", "Unknown source")
            score = doc["similarity_score"]
            
            context_parts.append(f"[Document {i+1}] Relevance: {score:.4f})\n{doc['text']}\n")
        
        return "\n".join(context_parts)
    
    def get_relevant_context(self, query: str) -> str:
        """Get relevant context for a query.
        
        Args:
            query: Query text
            
        Returns:
            Formatted context string with relevant information
        """
        results = self.retrieve(query)
        return self.format_context(results)
