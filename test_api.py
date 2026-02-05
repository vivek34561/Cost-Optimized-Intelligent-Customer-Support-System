"""
Test FastAPI Endpoint
======================

Simple script to test the chatbot API.
"""

import requests
import json


def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get("http://localhost:8000/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_chat(message: str):
    """Test chat endpoint"""
    print(f"Testing /chat with message: '{message}'")
    
    response = requests.post(
        "http://localhost:8000/chat",
        json={"message": message, "session_id": "test-123"}
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n📊 Intent: {result['intent']} ({result['confidence']:.1%} confidence)")
        print(f"📦 Bucket: {result['bucket']} ({result['cost_tier']} cost)")
        print(f"🤖 Response: {result['response']}\n")
    else:
        print(f"Error: {response.text}\n")


def test_intents():
    """Test intents endpoint"""
    print("Testing /intents endpoint...")
    response = requests.get("http://localhost:8000/intents")
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Total Intents: {result['total_intents']}")
    for bucket, info in result['buckets'].items():
        print(f"  {bucket}: {info['count']} intents - {info['description']}")
    print()


def test_stats():
    """Test stats endpoint"""
    print("Testing /stats endpoint...")
    response = requests.get("http://localhost:8000/stats")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


if __name__ == "__main__":
    print("="*80)
    print("TESTING CUSTOMER SUPPORT CHATBOT API")
    print("="*80)
    print("\nMake sure the API is running: python api.py\n")
    
    try:
        # Test all endpoints
        test_health()
        test_intents()
        test_stats()
        
        # Test chat with different queries
        test_queries = [
            "How do I track my order?",
            "I want to cancel my subscription",
            "I'm very unhappy with your service!"
        ]
        
        for query in test_queries:
            print("─"*80)
            test_chat(query)
        
        print("="*80)
        print("✅ All tests completed!")
        print("="*80)
    
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API")
        print("Make sure the API is running: python api.py")
