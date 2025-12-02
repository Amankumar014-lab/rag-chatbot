"""
Embeddings Module for RAG Pipeline
Uses SentenceTransformers to create embeddings and FAISS for vector storage
"""

import numpy as np
import faiss
import pickle
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from pathlib import Path


class EmbeddingManager:
    """Manage document embeddings and FAISS vector store"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model
        
        Args:
            model_name: SentenceTransformer model name
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.documents = []
        
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def create_embeddings(self, documents: List[Dict[str, Any]], batch_size: int = 32) -> np.ndarray:
        """
        Create embeddings for a list of document chunks
        
        Args:
            documents: List of document dictionaries with 'text' field
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings
        """
        texts = [doc['text'] for doc in documents]
        
        print(f"Creating embeddings for {len(texts)} documents...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"Created embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray, use_gpu: bool = False):
        """
        Build FAISS index for fast similarity search
        
        Args:
            embeddings: Numpy array of embeddings
            use_gpu: Whether to use GPU for FAISS (if available)
        """
        print("Building FAISS index...")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index - using IndexFlatIP for cosine similarity (inner product on normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Optionally move to GPU
        if use_gpu and faiss.get_num_gpus() > 0:
            print("Using GPU for FAISS")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def save_index(self, index_path: str = "faiss_index.bin", docs_path: str = "documents.pkl"):
        """
        Save FAISS index and documents to disk
        
        Args:
            index_path: Path to save FAISS index
            docs_path: Path to save documents
        """
        # Save FAISS index
        if self.index is not None:
            # If GPU index, move to CPU first
            if hasattr(self.index, 'index'):  # GPU index
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, index_path)
            else:
                faiss.write_index(self.index, index_path)
            print(f"Saved FAISS index to {index_path}")
        
        # Save documents
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
        print(f"Saved {len(self.documents)} documents to {docs_path}")
    
    def load_index(self, index_path: str = "faiss_index.bin", docs_path: str = "documents.pkl", use_gpu: bool = False):
        """
        Load FAISS index and documents from disk
        
        Args:
            index_path: Path to FAISS index
            docs_path: Path to documents
            use_gpu: Whether to use GPU
        """
        # Load FAISS index
        if Path(index_path).exists():
            self.index = faiss.read_index(index_path)
            
            # Optionally move to GPU
            if use_gpu and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            
            print(f"Loaded FAISS index with {self.index.ntotal} vectors")
        else:
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        # Load documents
        if Path(docs_path).exists():
            with open(docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            print(f"Loaded {len(self.documents)} documents")
        else:
            raise FileNotFoundError(f"Documents file not found: {docs_path}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents using query
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of document dictionaries with similarity scores
        """
        if self.index is None:
            raise ValueError("Index not built or loaded. Call build_faiss_index() or load_index() first.")
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):  # Valid index
                result = self.documents[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results
    
    def process_and_index(self, documents: List[Dict[str, Any]], save: bool = True):
        """
        Complete pipeline: create embeddings, build index, and optionally save
        
        Args:
            documents: List of document chunks
            save: Whether to save index and documents
        """
        self.documents = documents
        
        # Create embeddings
        embeddings = self.create_embeddings(documents)
        
        # Build FAISS index
        self.build_faiss_index(embeddings)
        
        # Save if requested
        if save:
            self.save_index()
        
        print("Indexing complete!")


def main():
    """Test the embeddings manager"""
    from data_processor import DataProcessor
    
    # Load and process documents
    print("="*80)
    print("LOADING AND PROCESSING DOCUMENTS")
    print("="*80)
    processor = DataProcessor(json_dir="MyFixit-Dataset-master/jsons", chunk_size=150)
    documents = processor.process_all_guides()
    
    # Create embeddings and build index
    print("\n" + "="*80)
    print("CREATING EMBEDDINGS AND BUILDING FAISS INDEX")
    print("="*80)
    embedding_manager = EmbeddingManager(model_name="all-MiniLM-L6-v2")
    embedding_manager.process_and_index(documents, save=True)
    
    # Test search
    print("\n" + "="*80)
    print("TESTING SEARCH")
    print("="*80)
    test_queries = [
        "How do I replace iPhone screen?",
        "Fix laptop not turning on",
        "Replace phone battery"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-"*80)
        results = embedding_manager.search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (Score: {result['similarity_score']:.4f}):")
            print(f"Title: {result['metadata']['title']}")
            print(f"Category: {result['metadata']['category']}")
            print(f"Text: {result['text'][:200]}...")


if __name__ == "__main__":
    main()
