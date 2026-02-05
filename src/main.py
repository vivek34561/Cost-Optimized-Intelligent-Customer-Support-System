"""
Main Chatbot Interface
=======================

Complete chatbot with graph-based routing and response generation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.graph import CustomerSupportGraph


class CustomerSupportChatbot:
    """Main chatbot interface"""
    
    def __init__(self):
        """Initialize chatbot with graph"""
        print("="*80)
        print("INITIALIZING CUSTOMER SUPPORT CHATBOT")
        print("="*80)
        
        self.graph = CustomerSupportGraph()
        
        print("="*80)
        print("✅ CHATBOT READY")
        print("="*80 + "\n")
    
    def chat(self, user_query: str) -> dict:
        """
        Process user query and return complete result
        
        Args:
            user_query: User's question
            
        Returns:
            Dictionary with routing info and response
        """
        state = self.graph.process(user_query)
        
        return {
            'user_message': user_query,
            'intent': state['predicted_intent'],
            'confidence': state['confidence'],
            'bucket': state['bucket'],
            'cost_tier': state['cost_tier'],
            'response': state['final_response']
        }
    
    def get_response(self, user_query: str) -> str:
        """
        Get just the response text
        
        Args:
            user_query: User's question
            
        Returns:
            Response string
        """
        return self.graph.get_response(user_query)


def interactive_chat():
    """Interactive chat session"""
    print("="*80)
    print("CUSTOMER SUPPORT CHATBOT - Interactive Mode")
    print("="*80)
    print("Type 'quit', 'exit', or 'q' to end the session\n")
    
    chatbot = CustomerSupportChatbot()
    
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye! Thanks for chatting.")
                break
            
            if not user_input:
                continue
            
            result = chatbot.chat(user_input)
            
            print(f"\n📊 Routing: {result['bucket']} | "
                  f"{result['intent']} ({result['confidence']:.0%})")
            print(f"🤖 Bot: {result['response']}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def test_chatbot():
    """Test chatbot with sample queries"""
    print("="*80)
    print("TESTING CUSTOMER SUPPORT CHATBOT")
    print("="*80 + "\n")
    
    chatbot = CustomerSupportChatbot()
    
    test_queries = [
        "What payment methods do you accept?",  # BUCKET_A
        "How do I cancel my order?",             # BUCKET_B (RAG)
        "I'm very upset with your service!",     # BUCKET_C
    ]
    
    for query in test_queries:
        print(f"\n{'─'*80}")
        result = chatbot.chat(query)
        print(f"\n💬 User: {query}")
        print(f"📊 {result['bucket']} | {result['intent']} ({result['confidence']:.0%})")
        print(f"🤖 Bot: {result['response']}")
        print('─'*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'interactive':
        interactive_chat()
    else:
        test_chatbot()
