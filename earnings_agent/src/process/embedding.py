# src/process/embedding.py
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import pickle
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import EMBEDDING_MODEL, DATA_DIR, EMBEDDING_DIMENSION

class EmbeddingModel:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
        # Create cache directory
        self.cache_dir = DATA_DIR / "embeddings"
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_path(self, text: str) -> Path:
        """
        Get the cache path for a given text.
        
        Args:
            text: The text to embed
            
        Returns:
            Path to the cache file
        """
        # Use a hash of the text as the filename
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self.cache_dir / f"{text_hash}.pkl"
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        # Check if we have this embedding cached
        cache_path = self._get_cache_path(text)
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Generate the embedding
        embedding = self.model.encode(text)
        
        # Cache the embedding
        with open(cache_path, 'wb') as f:
            pickle.dump(embedding, f)
        
        return embedding
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            2D array of embeddings
        """
        embeddings = np.zeros((len(texts), EMBEDDING_DIMENSION))
        
        for i, text in enumerate(texts):
            embeddings[i] = self.embed_text(text)
        
        return embeddings

def embed_transcript_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add embeddings to transcript chunks.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Chunks with embeddings added
    """
    model = EmbeddingModel()
    
    # Extract texts to embed
    texts = [chunk['text'] for chunk in chunks]
    
    # Get embeddings
    embeddings = model.embed_texts(texts)
    
    # Add embeddings to
    # src/process/embedding.py (continued)
    # Add embeddings to chunks
    for i, chunk in enumerate(chunks):
        chunk['embedding'] = embeddings[i]
    
    return chunks

def get_chunk_embeddings(symbol: str, quarter: int = None, year: int = None) -> List[Dict[str, Any]]:
    """
    Get embeddings for a transcript's chunks, creating them if they don't exist.
    
    Args:
        symbol: Stock ticker symbol
        quarter: Fiscal quarter
        year: Fiscal year
        
    Returns:
        List of chunks with embeddings
    """
    from src.fetch.transcript import get_transcript
    from src.process.chunking import split_transcript_into_chunks
    
    # Get the transcript
    transcript = get_transcript(symbol, quarter, year)
    
    # Split into chunks
    chunks = split_transcript_into_chunks(transcript)
    
    # Add embeddings
    chunks_with_embeddings = embed_transcript_chunks(chunks)
    
    return chunks_with_embeddings

if __name__ == "__main__":
    # Test the embedding functions
    from src.fetch.transcript import _get_demo_transcript
    from src.process.chunking import split_transcript_into_chunks
    
    transcript = _get_demo_transcript("AAPL")
    chunks = split_transcript_into_chunks(transcript)
    
    # Only use the first 2 chunks for testing to save time
    test_chunks = chunks[:2]
    
    chunks_with_embeddings = embed_transcript_chunks(test_chunks)
    
    print(f"Embedded {len(chunks_with_embeddings)} chunks")
    print(f"Embedding dimension: {len(chunks_with_embeddings[0]['embedding'])}")
    print(f"Sample embedding values: {chunks_with_embeddings[0]['embedding'][:5]}...")