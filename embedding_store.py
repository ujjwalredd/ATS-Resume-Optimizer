"""
Embedding Store Module
Creates and manages vector embeddings for semantic search
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import pickle


class EmbeddingStore:
    """Manages embeddings using sentence-transformers and FAISS"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", db_path: str = "embeddings_db"):
        self.model_name = model_name
        self.db_path = db_path
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []
        self.dimension = self.model.get_sentence_embedding_dimension()
        self._ensure_db_dir()
    
    def _ensure_db_dir(self):
        """Create database directory if it doesn't exist"""
        os.makedirs(self.db_path, exist_ok=True)
    
    def create_embeddings(self, texts: List[str], metadata: List[Dict] = None) -> np.ndarray:
        """Create embeddings for a list of texts"""
        print(f"Creating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        if metadata is None:
            metadata = [{"text": text, "index": i} for i, text in enumerate(texts)]
        
        self.metadata.extend(metadata)
        return embeddings
    
    def build_index(self, embeddings: np.ndarray, texts: List[str], metadata: List[Dict] = None):
        """Build FAISS index from embeddings"""
        if metadata is None:
            metadata = [{"text": text, "index": i} for i, text in enumerate(texts)]
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index (L2 normalized for cosine similarity)
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product for cosine similarity
        self.index.add(embeddings.astype('float32'))
        self.metadata = metadata
        
        print(f"Built index with {self.index.ntotal} vectors")
    
    def add_to_index(self, embeddings: np.ndarray, texts: List[str], metadata: List[Dict] = None):
        """Add new embeddings to existing index"""
        if self.index is None:
            self.build_index(embeddings, texts, metadata)
            return
        
        if metadata is None:
            start_idx = len(self.metadata)
            metadata = [{"text": text, "index": start_idx + i} for i, text in enumerate(texts)]
        
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        self.metadata.extend(metadata)
        print(f"Added {len(texts)} vectors. Total: {self.index.ntotal}")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[float, Dict]]:
        """Search for similar texts using semantic similarity"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search (k cannot exceed index size)
        search_k = min(k, self.index.ntotal)
        if search_k == 0:
            return []
        
        distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata) and idx >= 0:
                results.append((float(dist), self.metadata[idx]))
        
        return results
    
    def get_relevant_entries(self, query: str, threshold: float = 0.6, top_k: int = 10) -> List[Dict]:
        """Get relevant entries above similarity threshold"""
        results = self.search(query, k=top_k)
        relevant = [meta for score, meta in results if score >= threshold]
        return relevant
    
    def save(self, filepath: str = None):
        """Save index and metadata to disk"""
        if filepath is None:
            filepath = os.path.join(self.db_path, "index")
        
        if self.index is None:
            raise ValueError("No index to save")
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save metadata
        with open(f"{filepath}.metadata", 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Save config
        config = {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "size": self.index.ntotal
        }
        with open(f"{filepath}.config", 'w') as f:
            json.dump(config, f)
        
        print(f"Saved index to {filepath}")
    
    def load(self, filepath: str = None):
        """Load index and metadata from disk"""
        if filepath is None:
            filepath = os.path.join(self.db_path, "index")
        
        if not os.path.exists(f"{filepath}.faiss"):
            raise FileNotFoundError(f"Index file not found: {filepath}.faiss")
        
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.faiss")
        
        # Load metadata
        with open(f"{filepath}.metadata", 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Load config
        with open(f"{filepath}.config", 'r') as f:
            config = json.load(f)
            self.model_name = config["model_name"]
            self.dimension = config["dimension"]
        
        # Reinitialize model if needed
        if not hasattr(self, 'model') or self.model is None:
            self.model = SentenceTransformer(self.model_name)
        
        print(f"Loaded index from {filepath} ({self.index.ntotal} vectors)")

