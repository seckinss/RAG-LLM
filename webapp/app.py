import os
import sys
import logging
from typing import Dict, Any
import time
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from src.utils.config import (
    API_HOST,
    APP_TITLE,
    APP_DESCRIPTION,
    TOP_K_RESULTS
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


API_URL = f"https://{API_HOST}"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9

def check_api_health() -> Dict[str, Any]:
    """Check the health of the API.
    
    Returns:
        Dictionary with health information
    """
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"API health check failed with status code {response.status_code}")
            return {
                "status": "error",
                "retriever_ready": False,
                "llm_ready": False,
                "document_count": 0
            }
    except Exception as e:
        logger.error(f"Error checking API health: {e}")
        return {
            "status": "error",
            "retriever_ready": False,
            "llm_ready": False,
            "document_count": 0
        }

def query_api(
    query: str,
    top_k: int = TOP_K_RESULTS,
    include_sources: bool = True,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P
) -> Dict[str, Any]:
    """Query the API for an answer.
    
    Args:
        query: User query
        top_k: Number of documents to retrieve
        include_sources: Whether to include sources in the response
        temperature: Temperature for generation
        top_p: Top-p sampling parameter
        
    Returns:
        Dictionary with the response from the API
    """
    try:
        logger.info(f"Querying API with query: {query}")
        logger.info(f"API URL: {API_URL}/query")
        response = requests.post(
            f"{API_URL}/query",
            json={
                "query": query,
                "top_k": top_k,
                "include_sources": include_sources,
                "temperature": temperature,
                "top_p": top_p
            }
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"API query failed with status code {response.status_code}")
            return {
                "query": query,
                "answer": f"Error: API returned status code {response.status_code}",
                "sources": [],
                "elapsed_time": 0.0
            }
    except Exception as e:
        logger.error(f"Error querying API: {e}")
        return {
            "query": query,
            "answer": f"Error: {str(e)}",
            "sources": [],
            "elapsed_time": 0.0
        }

def initialize_session_state():
    """Initialize the session state."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "api_health" not in st.session_state:
        st.session_state.api_health = check_api_health()

def render_chat_message(message: Dict[str, Any], is_user: bool):
    """Render a chat message.
    
    Args:
        message: Message to render
        is_user: Whether the message is from the user
    """
    if is_user:
        st.chat_message("user").write(message["text"])
    else:
        with st.chat_message("assistant"):
            st.write(message["text"])
            
            if "sources" in message and message["sources"]:
                with st.expander("Show sources"):
                    for i, source in enumerate(message["sources"]):
                        source_filename = source.get("metadata", {}).get("filename", "Unknown")
                        st.markdown(f"**Source {i+1}: {source_filename}** (Score: {source['similarity_score']:.4f})")
                        st.text(source["text"][:500] + "..." if len(source["text"]) > 500 else source["text"])
                        st.divider()
            
            # Show timing info
            if "elapsed_time" in message:
                st.caption(f"Response time: {message['elapsed_time']:.2f} seconds")

def render_chat_history():
    """Render the chat history."""
    for message in st.session_state.chat_history:
        render_chat_message(message, message["role"] == "user")

def render_health_info():
    """Render API health information."""
    health = st.session_state.api_health
    status = health.get("status", "unknown")
    retriever_ready = health.get("retriever_ready", False)
    llm_ready = health.get("llm_ready", False)
    document_count = health.get("document_count", 0)
    
    status_color = "green" if status == "ok" else "red"
    retriever_color = "green" if retriever_ready else "red"
    llm_color = "green" if llm_ready else "red"
    
    st.sidebar.markdown(
        f"""
        ### System Status
        API Status: :{status_color}[{status}]  
        Retriever: :{retriever_color}[{'Ready' if retriever_ready else 'Not Ready'}]  
        LLM: :{llm_color}[{'Ready' if llm_ready else 'Not Ready'}]  
        Document Count: {document_count}
        """
    )

def render_settings():
    """Render settings controls."""
    st.sidebar.markdown("### Settings")
    
    top_k = st.sidebar.slider(
        "Top-k Documents",
        min_value=1,
        max_value=10,
        value=TOP_K_RESULTS,
        help="Number of documents to retrieve for context"
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_TEMPERATURE,
        step=0.1,
        help="Controls randomness in response generation"
    )
    
    top_p = st.sidebar.slider(
        "Top-p",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_TOP_P,
        step=0.1,
        help="Controls diversity of response generation"
    )
    
    include_sources = st.sidebar.checkbox(
        "Include Sources",
        value=True,
        help="Whether to include source documents in the response"
    )
    
    return {
        "top_k": top_k,
        "temperature": temperature,
        "top_p": top_p,
        "include_sources": include_sources
    }

def render_sidebar():
    """Render the sidebar."""
    st.sidebar.title("About")
    st.sidebar.info(APP_DESCRIPTION)
    
    render_health_info()
    settings = render_settings()
    
    if st.sidebar.button("Refresh Health Status"):
        st.session_state.api_health = check_api_health()
        st.rerun()
    
    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    return settings

def main():
    """Main entry point for the Streamlit app."""
    # Set page config
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Render the title
    st.title(f"{APP_TITLE} 🔍")
    
    # Render the sidebar and get settings
    settings = render_sidebar()
    
    # Render chat history
    render_chat_history()
    
    # Get user input
    user_query = st.chat_input("Ask a question...")
    
    # Handle user input
    if user_query:
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "text": user_query
        })
        
        # Show user message
        st.chat_message("user").write(user_query)
        
        # Display thinking message
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("_Thinking..._")
            
            # Query the API
            start_time = time.time()
            response = query_api(
                query=user_query,
                top_k=settings["top_k"],
                include_sources=settings["include_sources"],
                temperature=settings["temperature"],
                top_p=settings["top_p"]
            )
            
            # Clear thinking message
            thinking_placeholder.empty()
            
            # Show the answer
            st.write(response["answer"])
            
            # Show sources if available
            if settings["include_sources"] and "sources" in response and response["sources"]:
                with st.expander("Show sources"):
                    for i, source in enumerate(response["sources"]):
                        st.markdown(f"**Source {i+1}** (Score: {source['similarity_score']:.4f})")
                        st.text(source["text"][:500] + "..." if len(source["text"]) > 500 else source["text"])
                        st.divider()
            
            elapsed_time = response.get("elapsed_time", time.time() - start_time)
            st.caption(f"Response time: {elapsed_time:.2f} seconds")
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "text": response["answer"],
            "sources": response.get("sources", []),
            "elapsed_time": elapsed_time
        })

if __name__ == "__main__":
    main()
