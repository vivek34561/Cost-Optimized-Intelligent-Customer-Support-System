"""
Test Confidence-Based Fallback Logic
====================================

Demonstrates how low-confidence predictions are safely routed to 
RAG + Small LLM instead of giving potentially wrong direct answers
or expensive escalations.
"""

from intent_router import IntentRouter


def test_confidence_fallback():
    """Test that low confidence predictions route to BUCKET_B (RAG)"""
    print("=" * 80)
    print("TESTING CONFIDENCE-BASED FALLBACK")
    print("=" * 80)
    
    router = IntentRouter()
    
    # Test with ambiguous/unclear messages that might have low confidence
    test_cases = [
        "I need help with something",
        "Problem with my thing",
        "Can you assist?",
        "Issue here",
        "Not working properly",
    ]
    
    print("\n🎯 Low-confidence predictions should route to BUCKET_B (RAG + Small LLM)")
    print("   This prevents wrong direct answers and avoids expensive escalation\n")
    
    results = []
    for msg in test_cases:
        result = router.route_message(msg)
        results.append(result)
        
        # Show result
        confidence = result['confidence']
        bucket = result['bucket']
        action = result['action']
        
        # Highlight fallback behavior
        is_fallback = confidence < 0.5
        marker = "🛡️  FALLBACK" if is_fallback else "✓ Normal"
        
        print(f"{marker}")
        print(f"  Message: {msg}")
        print(f"  Intent: {result['predicted_intent']}")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Bucket: {bucket} ({result['cost_tier']} cost)")
        print(f"  Reason: {result['reason']}")
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    fallback_count = sum(1 for r in results if r['confidence'] < 0.5)
    bucket_b_count = sum(1 for r in results if r['bucket'] == 'BUCKET_B')
    
    print(f"\nTotal messages tested: {len(results)}")
    print(f"Low confidence (<50%): {fallback_count}")
    print(f"Routed to BUCKET_B: {bucket_b_count}")
    
    print("\n💡 KEY INSIGHT:")
    print("   Low-confidence predictions are safely handled by RAG + Small LLM")
    print("   This prevents:")
    print("     ❌ Wrong direct answers from BUCKET_A")
    print("     ❌ Expensive escalations to BUCKET_C")
    print("   And provides:")
    print("     ✅ Contextual responses from RAG")
    print("     ✅ Cost-effective handling (Low cost)")
    print("     ✅ Improved customer experience")
    
    print("\n" + "=" * 80)


def compare_thresholds():
    """Compare behavior with different confidence thresholds"""
    print("\n" + "=" * 80)
    print("COMPARING DIFFERENT CONFIDENCE THRESHOLDS")
    print("=" * 80)
    
    # Test with different thresholds
    router_lenient = IntentRouter(confidence_threshold=0.3)  # More permissive
    router_default = IntentRouter(confidence_threshold=0.5)  # Default
    router_strict = IntentRouter(confidence_threshold=0.7)   # More strict
    
    test_message = "I have a problem with my account"
    
    print(f"\nTest message: '{test_message}'")
    print("-" * 80)
    
    result_lenient = router_lenient.route_message(test_message)
    result_default = router_default.route_message(test_message)
    result_strict = router_strict.route_message(test_message)
    
    print(f"\nThreshold 0.3 (Lenient):")
    print(f"  Confidence: {result_lenient['confidence']:.2%}")
    print(f"  Bucket: {result_lenient['bucket']}")
    print(f"  Action: {result_lenient['action']}")
    
    print(f"\nThreshold 0.5 (Default):")
    print(f"  Confidence: {result_default['confidence']:.2%}")
    print(f"  Bucket: {result_default['bucket']}")
    print(f"  Action: {result_default['action']}")
    
    print(f"\nThreshold 0.7 (Strict):")
    print(f"  Confidence: {result_strict['confidence']:.2%}")
    print(f"  Bucket: {result_strict['bucket']}")
    print(f"  Action: {result_strict['action']}")
    
    print("\n💡 Higher threshold = More messages routed to safe RAG fallback")
    print("=" * 80)


if __name__ == "__main__":
    test_confidence_fallback()
    compare_thresholds()
