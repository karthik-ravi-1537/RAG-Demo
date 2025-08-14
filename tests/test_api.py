#!/usr/bin/env python3
"""
Test to verify OpenAI API integration.
Moved from root to tests directory for organization.
"""
import sys
import os
from pathlib import Path

# Add project root to path (now from tests subdirectory)
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

def test_openai_api():
    """Test OpenAI API connection."""
    import openai
    
    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "test_key_placeholder":
        print("❌ No valid OpenAI API key found")
        assert False, "No valid OpenAI API key found"
    
    print("✅ OpenAI API key found")
    
    # Test a simple API call
    client = openai.OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Say 'Hello from RAG Demo!' in exactly those words."}
        ],
        max_tokens=20,
        temperature=0
    )
    
    result = response.choices[0].message.content.strip()
    print(f"✅ OpenAI API test successful: {result}")
    
    assert result is not None
    assert len(result) > 0

def test_rag_with_llm():
    """Test RAG system with actual LLM."""
    from vanilla_rag.vanilla_rag import VanillaRAG
    from core.data_models import RAGConfig, ChunkingConfig, EmbeddingConfig
    
    # Create a simple config for testing
    config = RAGConfig(
        chunking=ChunkingConfig(strategy="semantic", chunk_size=300),
        embedding=EmbeddingConfig(model_name="all-MiniLM-L6-v2"),
        retrieval_top_k=2
    )
    
    rag = VanillaRAG(config)
    
    # Add a simple test document
    import tempfile
    test_content = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create 
    intelligent machines that can perform tasks that typically require human intelligence.
    Machine learning is a subset of AI that enables computers to learn from data.
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_file = f.name
    
    rag.add_documents([temp_file])
    
    # Test query with actual LLM
    result = rag.query("What is artificial intelligence?")
    
    print(f"✅ RAG with LLM test successful!")
    print(f"📊 Confidence: {result.confidence_score:.3f}")
    print(f"📄 Chunks retrieved: {len(result.retrieved_chunks)}")
    print(f"💬 Response preview: {result.response[:100]}...")
    
    # Clean up
    os.unlink(temp_file)
    
    # Assertions
    assert result is not None
    assert result.confidence_score >= 0
    assert len(result.retrieved_chunks) > 0
    assert len(result.response) > 0

def main():
    """Run API tests."""
    print("🚀 Testing RAG Demo with Real OpenAI API")
    print("=" * 50)
    
    # Test OpenAI API connection
    print("\n🔑 Testing OpenAI API connection...")
    api_works = test_openai_api()
    
    if api_works:
        print("\n🤖 Testing RAG system with actual LLM...")
        rag_works = test_rag_with_llm()
        
        if rag_works:
            print("\n🎉 All tests passed! RAG Demo is ready with real LLM integration.")
        else:
            print("\n⚠️  RAG test failed, but API connection works.")
    else:
        print("\n⚠️  OpenAI API test failed. Check your API key.")

if __name__ == "__main__":
    main()