"""
Customer Support RAG Graph
===========================

LangGraph state machine for customer support chatbot.
"""

from langgraph.graph import StateGraph, END

from src.state import ChatbotState
from src.nodes import IntentNode, RetrieveNode, GenerateNode


class CustomerSupportGraph:
    """LangGraph state machine for customer support"""
    
    def __init__(self):
        """Initialize graph with nodes"""
        print("Initializing Customer Support Graph...")
        
        # Initialize nodes
        self.intent_node = IntentNode()
        self.retrieve_node = RetrieveNode()
        self.generate_node = GenerateNode()
        
        # Build graph
        self.graph = self._build_graph()
        
        print("✓ Graph initialized\n")
    
    def _should_retrieve(self, state: ChatbotState) -> str:
        """
        Conditional edge: Determine if retrieval is needed
        
        Args:
            state: Current state
            
        Returns:
            Next node name
        """
        bucket = state['bucket']
        
        # Only retrieve for BUCKET_B
        if bucket == 'BUCKET_B':
            return "retrieve"
        else:
            return "generate"
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state machine
        
        Graph flow:
            START → intent → [conditional] → retrieve → generate → END
                                    ↓
                                  generate → END (if BUCKET_A or BUCKET_C)
        """
        # Create graph
        workflow = StateGraph(ChatbotState)
        
        # Add nodes
        workflow.add_node("intent", self.intent_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("generate", self.generate_node)
        
        # Set entry point
        workflow.set_entry_point("intent")
        
        # Add conditional edge after intent
        workflow.add_conditional_edges(
            "intent",
            self._should_retrieve,
            {
                "retrieve": "retrieve",
                "generate": "generate"
            }
        )
        
        # Add edges
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        # Compile
        return workflow.compile()
    
    def process(self, user_query: str) -> dict:
        """
        Process user query through the graph
        
        Args:
            user_query: User's question
            
        Returns:
            Final state with response
        """
        print(f"\n{'='*80}")
        print(f"Processing: {user_query}")
        print('='*80)
        
        # Initialize state
        initial_state = {
            'user_query': user_query,
            'predicted_intent': '',
            'confidence': 0.0,
            'bucket': '',
            'retrieved_documents': [],
            'retrieved_context': '',
            'final_response': '',
            'messages': [],
            'cost_tier': '',
            'action': ''
        }
        
        # Run graph
        final_state = self.graph.invoke(initial_state)
        
        print('='*80 + '\n')
        
        return final_state
    
    def get_response(self, user_query: str) -> str:
        """
        Convenience method to get just the response
        
        Args:
            user_query: User's question
            
        Returns:
            Generated response
        """
        state = self.process(user_query)
        return state['final_response']
