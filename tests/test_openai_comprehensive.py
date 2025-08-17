#!/usr/bin/env python3
"""
Comprehensive OpenAI API integration tests.
"""

import os
import tempfile

import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def has_valid_openai_key():
    """Check if we have a valid OpenAI API key."""
    api_key = os.getenv("OPENAI_API_KEY")
    return (
        api_key
        and api_key != "test_key_placeholder"
        and not api_key.startswith("your_")
        and api_key.startswith("sk-")
        and len(api_key) > 20
    )


@pytest.mark.skipif(not has_valid_openai_key(), reason="No valid OpenAI API key available")
class TestOpenAIComprehensive:
    """Comprehensive OpenAI integration tests."""

    def test_api_key_format_validation(self):
        """Test OpenAI API key format validation."""
        api_key = os.getenv("OPENAI_API_KEY")

        # Format checks
        assert isinstance(api_key, str), "API key should be string"
        assert len(api_key) > 50, "API key should be substantial length"
        assert api_key.startswith("sk-"), "OpenAI API key should start with 'sk-'"
        assert not api_key.isspace(), "API key should not be whitespace"

        # Content checks
        placeholder_indicators = ["test", "placeholder", "example", "your_key_here"]
        for indicator in placeholder_indicators:
            assert indicator.lower() not in api_key.lower(), f"API key contains placeholder: {indicator}"

        print(f"✅ API key format valid: {api_key[:15]}...")

    def test_openai_library_availability(self):
        """Test OpenAI library is available and importable."""
        try:
            import openai

            assert hasattr(openai, "OpenAI"), "OpenAI client class should be available"
            print("✅ OpenAI library available")
        except ImportError:
            pytest.fail("OpenAI library not installed")

    def test_openai_client_initialization(self):
        """Test OpenAI client initialization."""
        try:
            import openai

            api_key = os.getenv("OPENAI_API_KEY")

            client = openai.OpenAI(api_key=api_key)
            assert client is not None, "Client should be initialized"
            print("✅ OpenAI client initialized")

        except Exception as e:
            pytest.fail(f"OpenAI client initialization failed: {e}")

    def test_openai_chat_api_call(self):
        """Test OpenAI Chat API call."""
        try:
            import openai

            api_key = os.getenv("OPENAI_API_KEY")
            client = openai.OpenAI(api_key=api_key)

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Say 'OpenAI test successful' exactly."}],
                max_tokens=10,
                temperature=0,
            )

            result = response.choices[0].message.content.strip()
            assert "OpenAI test successful" in result, f"Unexpected response: {result}"
            print(f"✅ Chat API call successful: {result}")

        except Exception as e:
            pytest.fail(f"OpenAI Chat API call failed: {e}")

    def test_openai_embedding_provider(self):
        """Test OpenAI embedding provider."""
        try:
            from core.embedding_system import OpenAIEmbeddingProvider

            api_key = os.getenv("OPENAI_API_KEY")

            provider = OpenAIEmbeddingProvider(api_key=api_key)

            # Test properties
            assert provider.model_name == "text-embedding-ada-002"
            assert provider.dimension == 1536
            assert provider.api_key == api_key

            print("✅ OpenAI embedding provider initialized")

        except Exception as e:
            pytest.fail(f"OpenAI embedding provider test failed: {e}")

    def test_openai_embedding_generation(self):
        """Test OpenAI embedding generation."""
        try:
            from core.embedding_system import OpenAIEmbeddingProvider

            api_key = os.getenv("OPENAI_API_KEY")

            provider = OpenAIEmbeddingProvider(api_key=api_key)

            # Test single embedding
            texts = ["This is a test sentence for embedding generation."]
            embeddings = provider.generate_embeddings(texts)

            assert embeddings.shape[0] == 1, "Should generate 1 embedding"
            assert embeddings.shape[1] == 1536, "Should be 1536-dimensional"
            assert embeddings.dtype == float, "Embeddings should be float type"

            # Test multiple embeddings
            texts = ["First sentence.", "Second sentence.", "Third sentence."]
            embeddings = provider.generate_embeddings(texts)

            assert embeddings.shape[0] == 3, "Should generate 3 embeddings"
            assert embeddings.shape[1] == 1536, "Should be 1536-dimensional"

            print(f"✅ Embedding generation successful: {embeddings.shape}")

        except Exception as e:
            pytest.fail(f"OpenAI embedding generation failed: {e}")

    def test_rag_with_openai_embeddings(self):
        """Test RAG system with OpenAI embeddings."""
        try:
            from core.data_models import ChunkingConfig, EmbeddingConfig, RAGConfig
            from vanilla_rag.vanilla_rag import VanillaRAG

            # Configure RAG with OpenAI embeddings
            config = RAGConfig(
                chunking=ChunkingConfig(strategy="fixed_size", chunk_size=200, min_chunk_size=50),
                embedding=EmbeddingConfig(model_name="text-embedding-ada-002", api_key=os.getenv("OPENAI_API_KEY")),
                retrieval_top_k=3,
                similarity_threshold=0.5,
            )

            rag = VanillaRAG(config)

            # Create comprehensive test document
            test_content = """
            Artificial Intelligence (AI) is a transformative technology that is revolutionizing industries worldwide.
            Machine learning, a subset of AI, enables computers to learn patterns from data without explicit programming.
            Deep learning uses neural networks with multiple layers to process complex information and make intelligent decisions.
            Natural language processing (NLP) helps computers understand, interpret, and generate human language.
            Computer vision allows machines to interpret and understand visual information from the world around them.
            Robotics combines AI with mechanical engineering to create intelligent machines that can perform physical tasks.
            """

            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write(test_content)
                temp_file = f.name

            # Test document addition
            rag.add_documents([temp_file])
            assert len(rag.documents) == 1, "Should have 1 document"
            assert len(rag.chunks) > 0, "Should have created chunks"

            # Test querying
            queries = [
                "What is artificial intelligence?",
                "How does machine learning work?",
                "What is natural language processing?",
            ]

            for query in queries:
                result = rag.query(query)

                assert len(result.retrieved_chunks) > 0, f"Should retrieve chunks for: {query}"
                assert result.confidence_score > 0, f"Should have confidence for: {query}"
                assert len(result.response) > 0, f"Should generate response for: {query}"
                assert result.total_time > 0, f"Should track time for: {query}"

                print(
                    f"✅ Query '{query}': {len(result.retrieved_chunks)} chunks, confidence={result.confidence_score:.3f}"
                )

            # Cleanup
            os.unlink(temp_file)

            print("✅ RAG with OpenAI embeddings test successful")

        except Exception as e:
            pytest.fail(f"RAG with OpenAI embeddings failed: {e}")

    def test_openai_embedding_config(self):
        """Test OpenAI embedding configuration."""
        try:
            from core.data_models import EmbeddingConfig

            api_key = os.getenv("OPENAI_API_KEY")

            # Test default OpenAI config
            config = EmbeddingConfig(model_name="text-embedding-ada-002", api_key=api_key)

            assert config.model_name == "text-embedding-ada-002"
            assert config.api_key == api_key
            assert config.dimension == 1536
            assert config.batch_size == 100
            assert config.normalize is True

            # Test different OpenAI models
            models = ["text-embedding-ada-002", "text-embedding-3-small"]
            for model in models:
                config = EmbeddingConfig(model_name=model, api_key=api_key)
                assert config.model_name == model
                print(f"✅ Config for {model} created successfully")

        except Exception as e:
            pytest.fail(f"OpenAI embedding config test failed: {e}")

    def test_openai_error_handling(self):
        """Test OpenAI error handling."""
        try:
            from core.embedding_system import OpenAIEmbeddingProvider
            from core.exceptions import APIError, EmbeddingError

            # Test with invalid API key
            with pytest.raises((EmbeddingError, APIError)):
                provider = OpenAIEmbeddingProvider(api_key="invalid_key")
                provider.generate_embeddings(["test"])

            print("✅ OpenAI error handling works correctly")

        except Exception as e:
            pytest.fail(f"OpenAI error handling test failed: {e}")

    def test_openai_performance_characteristics(self):
        """Test OpenAI API performance characteristics."""
        try:
            import time

            from core.embedding_system import OpenAIEmbeddingProvider

            api_key = os.getenv("OPENAI_API_KEY")
            provider = OpenAIEmbeddingProvider(api_key=api_key)

            # Test different batch sizes
            test_cases = [
                (1, ["Single sentence test."]),
                (3, ["First sentence.", "Second sentence.", "Third sentence."]),
                (5, [f"Test sentence number {i}." for i in range(1, 6)]),
            ]

            for expected_count, texts in test_cases:
                start_time = time.time()
                embeddings = provider.generate_embeddings(texts)
                elapsed_time = time.time() - start_time

                assert embeddings.shape[0] == expected_count
                assert elapsed_time < 10.0, f"Should complete in reasonable time, took {elapsed_time:.2f}s"

                print(f"✅ {expected_count} embeddings generated in {elapsed_time:.3f}s")

        except Exception as e:
            pytest.fail(f"OpenAI performance test failed: {e}")


def test_environment_setup_comprehensive():
    """Comprehensive environment setup test."""
    print("🔍 Environment Setup Analysis")
    print("-" * 40)

    # Check .env file
    env_file = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_file):
        print("✅ .env file exists")

        with open(env_file) as f:
            content = f.read()
            if "OPENAI_API_KEY" in content:
                print("✅ OPENAI_API_KEY found in .env file")
            else:
                print("⚠️  OPENAI_API_KEY not found in .env file")
    else:
        print("⚠️  .env file not found")

    # Check environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        if has_valid_openai_key():
            print("✅ Valid OPENAI_API_KEY detected in environment")
            print(f"   Key length: {len(api_key)}")
            print(f"   Key format: {api_key[:10]}...{api_key[-4:]}")
        else:
            print("⚠️  OPENAI_API_KEY present but appears to be placeholder")
    else:
        print("❌ OPENAI_API_KEY not set in environment")

    # Check OpenAI library
    try:
        import openai  # noqa: F401

        print("✅ OpenAI library available")
    except ImportError:
        print("❌ OpenAI library not installed")

    print(
        f"\n🎯 Overall Status: {'✅ Ready for OpenAI integration' if has_valid_openai_key() else '⚠️  OpenAI integration not ready'}"
    )


if __name__ == "__main__":
    # Run environment check
    test_environment_setup_comprehensive()

    # Run pytest if API key is available
    if has_valid_openai_key():
        print("\n" + "=" * 50)
        print("Running OpenAI integration tests...")
        pytest.main([__file__, "-v", "-s"])
    else:
        print("\n⚠️  Skipping OpenAI tests - no valid API key found")
