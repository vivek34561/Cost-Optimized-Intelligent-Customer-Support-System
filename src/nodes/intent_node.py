"""
Intent Classification Node
===========================

Routes user query through intent classifier.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from intent_router import IntentRouter
from src.state import ChatbotState


class IntentNode:
    """Node for intent classification and routing"""
    
    def __init__(self):
        """Initialize intent router"""
        self.router = IntentRouter()
        print("  ✓ Intent classification node initialized")
    
    def __call__(self, state: ChatbotState) -> ChatbotState:
        """
        Classify intent and determine routing bucket
        
        Args:
            state: Current chatbot state
            
        Returns:
            Updated state with routing information
        """
        query = state['user_query']
        
        print(f"  → Classifying intent...")
        
        # Route message
        routing_result = self.router.route_message(query)
        
        # Update state
        state['predicted_intent'] = routing_result['predicted_intent']
        state['confidence'] = routing_result['confidence']
        state['bucket'] = routing_result['bucket']
        state['cost_tier'] = routing_result['cost_tier']
        state['action'] = routing_result['action']
        
        print(f"  ✓ Intent: {routing_result['predicted_intent']} "
              f"({routing_result['confidence']:.1%} confidence)")
        print(f"  ✓ Bucket: {routing_result['bucket']} "
              f"({routing_result['cost_tier']} cost)")
        
        return state
