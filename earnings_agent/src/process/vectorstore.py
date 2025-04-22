# src/process/vectorstore.py
import faiss
import numpy as np
import pickle
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import VECTOR_STORE_PATH, EMBEDDING_DIMENSION

class FAISSVectorStore:
    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        """
        Initialize the FAISS vector store.
        
        Args:
            dimension: Dimension of embedding vectors
        """
        self.dimension = dimension
        self.index = None
        self.chunks = []
        self.vector_store_path = VECTOR_STORE_PATH
        self.vector_store_path.mkdir(exist_ok=True)
    
    def _get_index_path(self, symbol: str, quarter: int = None, year: int = None) -> Path:
        """
        Get the path for storing the FAISS index.
        
        Args:
            symbol: Stock ticker symbol
            quarter: Fiscal quarter
            year: Fiscal year
            
        Returns:
            Path to the index file
        """
        quarter_year = f"_q{quarter}_{year}" if quarter and year else ""
        return self.vector_store_path / f"{symbol.lower()}{quarter_year}_index.faiss"
    
    def _get_chunks_path(self, symbol: str, quarter: int = None, year: int = None) -> Path:
        """
        Get the path for storing chunk metadata.
        
        Args:
            symbol: Stock ticker symbol
            quarter: Fiscal quarter
            year: Fiscal year
            
        Returns:
            Path to the chunks file
        """
        quarter_year = f"_q{quarter}_{year}" if quarter and year else ""
        return self.vector_store_path / f"{symbol.lower()}{quarter_year}_chunks.pkl"
    
    def load(self, symbol: str, quarter: int = None, year: int = None) -> bool:
        """
        Load a saved index and chunks.
        
        Args:
            symbol: Stock ticker symbol
            quarter: Fiscal quarter
            year: Fiscal year
            
        Returns:
            True if loaded successfully, False otherwise
        """
        index_path = self._get_index_path(symbol, quarter, year)
        chunks_path = self._get_chunks_path(symbol, quarter, year)
        
        if not index_path.exists() or not chunks_path.exists():
            return False
        
        try:
            self.index = faiss.read_index(str(index_path))
            
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            
            return True
            
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False
    
    def save(self, symbol: str, quarter: int = None, year: int = None) -> bool:
        """
        Save the index and chunks.
        
        Args:
            symbol: Stock ticker symbol
            quarter: Fiscal quarter
            year: Fiscal year
            
        Returns:
            True if saved successfully, False otherwise
        """
        if self.index is None:
            return False
        
        index_path = self._get_index_path(symbol, quarter, year)
        chunks_path = self._get_chunks_path(symbol, quarter, year)
        
        try:
            faiss.write_index(self.index, str(index_path))
            
            with open(chunks_path, 'wb') as f:
                pickle.dump(self.chunks, f)
            
            return True
            
        except Exception as e:
            print(f"Error saving vector store: {e}")
            return False
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of chunks with embeddings
        """
        # If index doesn't exist, create it
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        # Extract embeddings
        embeddings = np.array([chunk['embedding'] for chunk in chunks], dtype=np.float32)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store chunks without embeddings (to save memory)
        for chunk in chunks:
            # Create a copy without the embedding
            chunk_copy = chunk.copy()
            if 'embedding' in chunk_copy:
                del chunk_copy['embedding']
            self.chunks.append(chunk_copy)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of similar chunks with scores
        """
        if self.index is None:
            return []
        
        # Make sure we have a 2D array (even for single vector)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        # Get results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks) and idx >= 0:  # Ensure index is valid
                result = self.chunks[idx].copy()
                result['score'] = float(distances[0][i])
                results.append(result)
        
        return results

def index_transcript(symbol: str, quarter: int = None, year: int = None) -> bool:
    """
    Index a transcript in the vector store.
    
    Args:
        symbol: Stock ticker symbol
        quarter: Fiscal quarter
        year: Fiscal year
        
    Returns:
        True if indexed successfully, False otherwise
    """
    from src.process.embedding import get_chunk_embeddings
    
    # Get chunks with embeddings
    chunks = get_chunk_embeddings(symbol, quarter, year)
    
    if not chunks:
        return False
    
    # Create and save vector store
    vector_store = FAISSVectorStore()
    vector_store.add_chunks(chunks)
    success = vector_store.save(symbol, quarter, year)
    
    return success

def semantic_search(symbol: str, query: str, k: int = 5, quarter: int = None, year: int = None) -> List[Dict[str, Any]]:
    """
    Search for relevant chunks in a transcript.
    
    Args:
        symbol: Stock ticker symbol
        query: Search query
        k: Number of results to return
        quarter: Fiscal quarter
        year: Fiscal year
        
    Returns:
        List of relevant chunks with scores
    """
    from src.process.embedding import EmbeddingModel
    
    # Load vector store
    vector_store = FAISSVectorStore()
    if not vector_store.load(symbol, quarter, year):
        # If not found, try to index it
        success = index_transcript(symbol, quarter, year)
        if not success:
            return []
        vector_store.load(symbol, quarter, year)
    
    # Get query embedding
    model = EmbeddingModel()
    query_embedding = model.embed_text(query)
    
    # Search
    results = vector_store.search(query_embedding, k)
    
    return results

if __name__ == "__main__":
    # Test indexing and searching
    symbol = "AAPL"
    
    # Index the transcript
    print(f"Indexing transcript for {symbol}...")
    success = index_transcript(symbol)
    
    if success:
        print("Transcript indexed successfully")
        
        # Test searching
        test_query = "What was the revenue this quarter?"
        print(f"Searching for: '{test_query}'")
        
        results = semantic_search(symbol, test_query, k=3)
        
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (Score: {result['score']:.4f}):")
            print(f"Speakers: {result['speakers']}")
            print(f"Text: {result['text'][:200]}...")
    else:
        print("Failed to index transcript")