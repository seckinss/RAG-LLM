import os
import argparse
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import fitz  # PyMuPDF
import pdfplumber
import re
import json
from tqdm import tqdm

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from src.utils.config import CHUNK_SIZE, CHUNK_OVERLAP, RAW_DATA_DIR, PROCESSED_DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Class for preprocessing text documents."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """Initialize the text preprocessor.
        
        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Amount of overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        try:
            text = ""
            # Try with PyMuPDF first
            try:
                with fitz.open(file_path) as doc:
                    for page in doc:
                        text += page.get_text()
            except Exception as e:
                logger.warning(f"PyMuPDF extraction failed for {file_path}, trying pdfplumber: {e}")
                # Fallback to pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() or ""
            
            # Clean up the text
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return ""
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from a txt file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Extracted text as a string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return ""
    
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """Split text into chunks.
        
        Args:
            text: Input text
            metadata: Additional metadata for the document
            
        Returns:
            List of Document objects with text chunks and metadata
        """
        if not text:
            return []
        
        metadata = metadata or {}
        chunks = self.text_splitter.create_documents([text], [metadata])
        return chunks
    
    def process_document(self, file_path: str) -> List[Document]:
        """Process a document from start to finish.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of Document objects with text chunks and metadata
        """
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        metadata = {
            "source": str(file_path),
            "filename": file_path.name,
        }
        
        # Extract text based on file extension
        if file_ext == ".pdf":
            text = self.extract_text_from_pdf(str(file_path))
        elif file_ext == ".txt":
            text = self.extract_text_from_txt(str(file_path))
        else:
            logger.warning(f"Unsupported file type: {file_ext} for {file_path}")
            text = ""
        
        # Split the text and return documents
        return self.split_text(text, metadata)
    
    def process_directory(self, input_dir: str, output_dir: str) -> None:
        """Process all documents in a directory.
        
        Args:
            input_dir: Directory containing documents to process
            output_dir: Directory to save processed documents
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all PDF and TXT files
        files = list(input_dir.glob("**/*.pdf")) + list(input_dir.glob("**/*.txt"))
        logger.info(f"Found {len(files)} documents to process")
        
        for file_path in tqdm(files, desc="Processing documents"):
            try:
                # Process the document
                chunks = self.process_document(file_path)
                
                if not chunks:
                    logger.warning(f"No text extracted from {file_path}")
                    continue
                
                # Save the chunks as JSON
                rel_path = file_path.relative_to(input_dir)
                output_path = output_dir / f"{rel_path.stem}.json"
                
                # Ensure parent directory exists
                os.makedirs(output_path.parent, exist_ok=True)
                
                # Convert chunks to dictionaries
                chunks_dict = [
                    {
                        "text": chunk.page_content,
                        "metadata": chunk.metadata,
                        "chunk_id": i
                    }
                    for i, chunk in enumerate(chunks)
                ]
                
                # Save the chunks
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(chunks_dict, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Processed {file_path} -> {output_path} ({len(chunks)} chunks)")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

def main():
    """Main function to run the preprocessor from the command line."""
    parser = argparse.ArgumentParser(description="Process documents into chunks")
    parser.add_argument("--input_dir", type=str, default=RAW_DATA_DIR, 
                        help="Directory containing documents to process")
    parser.add_argument("--output_dir", type=str, default=PROCESSED_DATA_DIR,
                        help="Directory to save processed documents")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE,
                        help="Size of each text chunk")
    parser.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP,
                        help="Amount of overlap between chunks")
    
    args = parser.parse_args()
    
    # Create preprocessor and process the directory
    preprocessor = TextPreprocessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    preprocessor.process_directory(args.input_dir, args.output_dir)
    logger.info("Document processing complete")

if __name__ == "__main__":
    main()
