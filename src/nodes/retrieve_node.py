"""
RAG Retrieval Node
==================

Retrieves relevant documents from FAISS.
"""

from src.retriever import RAGRetriever
from src.state import ChatbotState


class RetrieveNode:
    """Node for RAG retrieval"""
    
    def __init__(self):
        """Initialize RAG retriever"""
        try:
            self.retriever = RAGRetriever()
            print("  ✓ RAG retrieval node initialized")
        except Exception as e:
            print(f"  ⚠ Warning: RAG retrieval initialization failed: {e}")
            self.retriever = None
    
    def __call__(self, state: ChatbotState) -> ChatbotState:
        """
        Retrieve relevant documents from Pinecone
        
        Args:
            state: Current chatbot state
            
        Returns:
            Updated state with retrieved documents
        """
        if self.retriever is None:
            state['retrieved_documents'] = []
            state['retrieved_context'] = "No retrieval system available."
            return state
        
        query = state['user_query']
        intent = state.get('predicted_intent')
        
        print(f"  → Retrieving relevant documents from FAISS...")
        
        # Retrieve documents
        documents = self.retriever.retrieve(query)
        
        # Format context
        context = self.retriever.format_context(documents)
        
        # Update state
        state['retrieved_documents'] = documents
        state['retrieved_context'] = context
        
        print(f"  ✓ Retrieved {len(documents)} documents")
        
        return state
