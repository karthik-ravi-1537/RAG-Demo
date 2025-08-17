#!/usr/bin/env python3
"""
Tests for OpenAI API integration and key validation.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from core.data_models import ChunkingConfig, EmbeddingConfig, RAGConfig
from core.embedding_system import OpenAIEmbeddingProvider
from core.exceptions import APIError, EmbeddingError
from vanilla_rag.vanilla_rag import VanillaRAG


class TestOpenAIIntegration:
    """Test OpenAI API integration and key handling."""

    def test_openai_api_key_from_env(self):
        """Test that OpenAI API key is properly loaded from environment."""
        # Load environment variables
        from dotenv import load_dotenv

        load_dotenv()

        # Check if API key exists in environment
        api_key = os.getenv("OPENAI_API_KEY")

        if api_key and api_key != "test_key_placeholder" and not api_key.startswith("your_"):
            assert len(api_key) > 20, "API key should be substantial length"
            assert api_key.startswith("sk-"), "OpenAI API key should start with 'sk-'"
            print(f"✅ Valid OpenAI API key found: {api_key[:10]}...")
        else:
            pytest.skip("No valid OpenAI API key found in environment")

    def test_openai_embedding_provider_initialization(self):
        """Test OpenAI embedding provider initialization."""
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key or api_key == "test_key_placeholder":
            pytest.skip("No valid OpenAI API key found")

        # Test successful initialization
        provider = OpenAIEmbeddingProvider(api_key=api_key)
        assert provider.model_name == "text-embedding-ada-002"
        assert provider.dimension == 1536
        assert provider.api_key == api_key

    def test_openai_embedding_provider_no_key(self):
        """Test OpenAI embedding provider fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EmbeddingError, match="OpenAI API key not provided"):
                OpenAIEmbeddingProvider()

    def test_openai_embedding_provider_invalid_key(self):
        """Test OpenAI embedding provider with invalid key."""
        # Test with clearly invalid key format
        with pytest.raises((EmbeddingError, APIError)):
            provider = OpenAIEmbeddingProvider(api_key="invalid_key")
            # Try to use it to trigger the error
            provider.generate_embeddings(["test"])

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "test_key_placeholder",
        reason="No valid OpenAI API key available",
    )
    def test_openai_api_call(self):
        """Test actual OpenAI API call (requires valid API key)."""
        try:
            import openai

            api_key = os.getenv("OPENAI_API_KEY")
            client = openai.OpenAI(api_key=api_key)

            # Test a simple API call
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Say 'API test successful' exactly."}],
                max_tokens=10,
                temperature=0,
            )

            result = response.choices[0].message.content.strip()
            assert "API test successful" in result
            print(f"✅ OpenAI API call successful: {result}")

        except ImportError:
            pytest.skip("OpenAI library not installed")
        except Exception as e:
            pytest.fail(f"OpenAI API call failed: {e}")

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "test_key_placeholder",
        reason="No valid OpenAI API key available",
    )
    def test_openai_embedding_generation(self):
        """Test OpenAI embedding generation (requires valid API key)."""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            provider = OpenAIEmbeddingProvider(api_key=api_key)

            # Test embedding generation
            texts = ["This is a test sentence.", "Another test sentence."]
            embeddings = provider.generate_embeddings(texts)

            assert embeddings.shape[0] == 2, "Should generate 2 embeddings"
            assert embeddings.shape[1] == 1536, "Should be 1536-dimensional"
            assert embeddings.dtype == float, "Embeddings should be float type"

            print(f"✅ Generated embeddings: {embeddings.shape}")

        except Exception as e:
            pytest.fail(f"OpenAI embedding generation failed: {e}")

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "test_key_placeholder",
        reason="No valid OpenAI API key available",
    )
    def test_rag_with_openai_embeddings(self):
        """Test RAG system with OpenAI embeddings (requires valid API key)."""
        try:
            # Configure RAG to use OpenAI embeddings
            config = RAGConfig(
                chunking=ChunkingConfig(strategy="semantic", chunk_size=300),
                embedding=EmbeddingConfig(model_name="text-embedding-ada-002", api_key=os.getenv("OPENAI_API_KEY")),
                retrieval_top_k=2,
            )

            rag = VanillaRAG(config)

            # Create test document
            test_content = """
            Artificial Intelligence (AI) is a branch of computer science that aims to create
            intelligent machines. Machine learning is a subset of AI that enables computers
            to learn from data without being explicitly programmed.
            """

            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write(test_content)
                temp_file = f.name

            # Test document addition and querying
            rag.add_documents([temp_file])
            result = rag.query("What is artificial intelligence?")

            assert len(result.retrieved_chunks) > 0, "Should retrieve chunks"
            assert result.confidence_score > 0, "Should have confidence score"
            assert len(result.response) > 0, "Should generate response"

            print(
                f"✅ RAG with OpenAI embeddings: {len(result.retrieved_chunks)} chunks, confidence={result.confidence_score:.3f}"
            )

            # Cleanup
            os.unlink(temp_file)

        except Exception as e:
            pytest.fail(f"RAG with OpenAI embeddings failed: {e}")

    def test_api_key_validation_format(self):
        """Test API key format validation."""
        api_key = os.getenv("OPENAI_API_KEY")

        if api_key and api_key != "test_key_placeholder":
            # Test key format
            assert isinstance(api_key, str), "API key should be string"
            assert len(api_key) > 20, "API key should be substantial length"
            assert api_key.startswith("sk-"), "OpenAI API key should start with 'sk-'"
            assert not api_key.isspace(), "API key should not be whitespace"

            # Test key doesn't contain obvious placeholder text
            placeholder_indicators = ["test", "placeholder", "example", "your_key_here"]
            for indicator in placeholder_indicators:
                assert (
                    indicator.lower() not in api_key.lower()
                ), f"API key appears to contain placeholder text: {indicator}"
        else:
            pytest.skip("No valid OpenAI API key found for format validation")

    def test_embedding_config_with_openai_key(self):
        """Test embedding configuration with OpenAI API key."""
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key or api_key == "test_key_placeholder":
            pytest.skip("No valid OpenAI API key available")

        # Test embedding config creation
        config = EmbeddingConfig(model_name="text-embedding-ada-002", api_key=api_key)

        assert config.model_name == "text-embedding-ada-002"
        assert config.api_key == api_key
        assert config.dimension == 1536
        assert config.batch_size == 100
        assert config.normalize is True

    def test_mock_openai_api_failure(self):
        """Test handling of OpenAI API failures."""
        with patch("openai.OpenAI") as mock_openai:
            # Mock API failure
            mock_client = MagicMock()
            mock_client.embeddings.create.side_effect = Exception("API Error")
            mock_openai.return_value = mock_client

            provider = OpenAIEmbeddingProvider(api_key="test_key")

            with pytest.raises(APIError, match="OpenAI API error"):
                provider.generate_embeddings(["test text"])


def test_environment_setup():
    """Test that environment is properly set up for OpenAI integration."""
    # Check if .env file exists
    env_file = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_file):
        print("✅ .env file found")

        # Read .env file and check for OpenAI key
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
        if api_key == "test_key_placeholder":
            print("⚠️  OPENAI_API_KEY is set to placeholder value")
        else:
            print("✅ OPENAI_API_KEY is set in environment")
    else:
        print("⚠️  OPENAI_API_KEY not set in environment")


if __name__ == "__main__":
    # Run basic environment check
    test_environment_setup()

    # Run pytest
    pytest.main([__file__, "-v"])
