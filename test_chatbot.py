#!/usr/bin/env python3
"""
Test script to debug chatbot responses
"""

from llm_service import llm_service

def test_chatbot():
    print("🤖 Testing Chatbot Responses")
    print("=" * 50)
    
    test_messages = [
        "Hello",
        "I feel sad",
        "I'm stressed about work", 
        "What is depression?",
        "I need help",
        "I'm feeling anxious",
        "How are you?",
        "I can't sleep",
        "I feel lonely"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n{i}. User: {message}")
        result = llm_service.generate_response(message)
        print(f"   Bot: {result['response']}")
        print(f"   Source: {result['source']} (confidence: {result['confidence']:.2f})")
        
        if result['source'] == 'api':
            print(f"   Provider: {result.get('provider', 'unknown')}")
        elif result['source'] == 'intent':
            print(f"   Intent: {result.get('intent', 'unknown')}")
        elif result['source'] == 'faq':
            print(f"   Matched: {result.get('matched_question', 'unknown')}")

if __name__ == "__main__":
    test_chatbot()
