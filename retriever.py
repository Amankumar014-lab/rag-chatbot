"""
Retriever Module for RAG Pipeline
Handles document retrieval and context preparation for LLM
"""

from typing import List, Dict, Any
from embeddings import EmbeddingManager


class Retriever:
    """Retrieve relevant documents for RAG"""
    
    def __init__(self, embedding_manager: EmbeddingManager = None, top_k: int = 5):
        """
        Initialize retriever
        
        Args:
            embedding_manager: EmbeddingManager instance with loaded index
            top_k: Number of documents to retrieve
        """
        self.embedding_manager = embedding_manager
        self.top_k = top_k
        
        if embedding_manager is None:
            # Initialize and load saved index
            self.embedding_manager = EmbeddingManager()
            try:
                self.embedding_manager.load_index()
                print("Loaded existing FAISS index")
            except FileNotFoundError:
                print("No existing index found. You need to create one first.")
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User query string
            top_k: Number of results (uses default if None)
            
        Returns:
            List of relevant document chunks with metadata and scores
        """
        k = top_k if top_k is not None else self.top_k
        
        results = self.embedding_manager.search(query, top_k=k)
        return results
    
    def format_context(self, retrieved_docs: List[Dict[str, Any]], include_metadata: bool = True) -> str:
        """
        Format retrieved documents into context string for LLM
        
        Args:
            retrieved_docs: List of retrieved document dictionaries
            include_metadata: Whether to include source metadata
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            # Add metadata header if requested
            if include_metadata:
                title = doc['metadata'].get('title', 'Unknown')
                category = doc['metadata'].get('category', 'Unknown')
                context_parts.append(f"[Document {i} - {title} | Category: {category}]")
            else:
                context_parts.append(f"[Document {i}]")
            
            # Add document text
            context_parts.append(doc['text'])
            context_parts.append("")  # Blank line between documents
        
        return "\n".join(context_parts)
    
    def get_sources(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Extract source information from retrieved documents
        
        Args:
            retrieved_docs: List of retrieved document dictionaries
            
        Returns:
            List of source information dictionaries
        """
        sources = []
        
        for doc in retrieved_docs:
            source = {
                'title': doc['metadata'].get('title', 'Unknown'),
                'category': doc['metadata'].get('category', 'Unknown'),
                'url': doc['metadata'].get('url', ''),
                'score': f"{doc.get('similarity_score', 0):.4f}"
            }
            sources.append(source)
        
        return sources
    
    def retrieve_and_format(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """
        Complete retrieval pipeline: retrieve docs and format for LLM
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with context, sources, and retrieved_docs
        """
        # Retrieve documents
        retrieved_docs = self.retrieve(query, top_k)
        
        # Format context for LLM
        context = self.format_context(retrieved_docs)
        
        # Extract sources
        sources = self.get_sources(retrieved_docs)
        
        return {
            'context': context,
            'sources': sources,
            'retrieved_docs': retrieved_docs
        }


def main():
    """Test the retriever"""
    print("="*80)
    print("TESTING RETRIEVER")
    print("="*80)
    
    # Initialize retriever
    retriever = Retriever(top_k=3)
    
    # Test queries
    test_queries = [
        "How do I fix a broken iPhone screen?",
        "My laptop won't turn on, what should I check?",
        "How to replace a phone battery safely?",
        "Configure WiFi router settings"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-"*80)
        
        # Retrieve and format
        result = retriever.retrieve_and_format(query, top_k=3)
        
        # Display sources
        print("\nRetrieved Sources:")
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. {source['title']} (Score: {source['score']})")
            print(f"   Category: {source['category']}")
            if source['url']:
                print(f"   URL: {source['url']}")
        
        # Display formatted context (preview)
        print(f"\nFormatted Context (preview):")
        print(result['context'][:500])
        print("...\n")


if __name__ == "__main__":
    main()
