#!/usr/bin/env python3
"""
Free Chatbot Setup Script
This script helps you set up the free LLM chatbot options.
"""

import os
import sys

def setup_environment():
    """Set up environment variables for free APIs"""
    print("🤖 Free Chatbot Setup")
    print("=" * 50)
    
    print("\nChoose your free option:")
    print("1. Completely FREE (no API key needed) - Recommended")
    print("2. Hugging Face FREE (30K requests/month)")
    print("3. OpenAI FREE ($5 credit for new users)")
    print("4. Local only (intents + FAQ)")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        print("\n✅ Using completely FREE API (no setup needed)")
        print("   - Uses Hugging Face free inference API")
        print("   - No API key required")
        print("   - Rate limited but works great")
        return "free"
        
    elif choice == "2":
        print("\n🔑 Hugging Face Setup:")
        print("1. Go to: https://huggingface.co/settings/tokens")
        print("2. Create a free account")
        print("3. Generate a new token")
        print("4. Copy the token")
        
        token = input("\nEnter your Hugging Face token: ").strip()
        if token:
            os.environ['HUGGINGFACE_TOKEN'] = token
            print("✅ Hugging Face token set!")
            return "huggingface"
        else:
            print("❌ No token provided, using free option instead")
            return "free"
            
    elif choice == "3":
        print("\n🔑 OpenAI Setup:")
        print("1. Go to: https://platform.openai.com/api-keys")
        print("2. Create a free account")
        print("3. Generate a new API key")
        print("4. Copy the API key")
        
        api_key = input("\nEnter your OpenAI API key: ").strip()
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
            print("✅ OpenAI API key set!")
            return "openai"
        else:
            print("❌ No API key provided, using free option instead")
            return "free"
            
    elif choice == "4":
        print("\n✅ Using local intents + FAQ only")
        print("   - No API calls")
        print("   - Uses your intents.json and FAQ.csv")
        return "local"
        
    else:
        print("❌ Invalid choice, using free option")
        return "free"

def update_llm_service(provider):
    """Update the LLM service configuration"""
    if provider == "local":
        # Disable API usage
        with open('llm_service.py', 'r') as f:
            content = f.read()
        
        content = content.replace(
            "llm_service = SimpleLLMService(use_api=True, api_provider='free')",
            "llm_service = SimpleLLMService(use_api=False)"
        )
        
        with open('llm_service.py', 'w') as f:
            f.write(content)
        
        print("✅ Updated to use local intents + FAQ only")
    else:
        # Update to use the chosen provider
        with open('llm_service.py', 'r') as f:
            content = f.read()
        
        content = content.replace(
            "llm_service = SimpleLLMService(use_api=True, api_provider='free')",
            f"llm_service = SimpleLLMService(use_api=True, api_provider='{provider}')"
        )
        
        with open('llm_service.py', 'w') as f:
            f.write(content)
        
        print(f"✅ Updated to use {provider} API")

def main():
    print("Setting up your free mental health chatbot...")
    
    # Check if required files exist
    if not os.path.exists('intents.json'):
        print("❌ intents.json not found!")
        return
    
    if not os.path.exists('Mental_Health_FAQ.csv'):
        print("❌ Mental_Health_FAQ.csv not found!")
        return
    
    # Setup environment
    provider = setup_environment()
    
    # Update LLM service
    update_llm_service(provider)
    
    print("\n🎉 Setup complete!")
    print("\nTo start your chatbot:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run the app: python app.py")
    print("3. Go to: http://localhost:5000")
    print("4. Login and try the chat in the dashboard!")
    
    if provider == "free":
        print("\n💡 Note: The free API may take a few seconds to respond on first use.")
        print("   This is normal as the model loads.")

if __name__ == "__main__":
    main()
