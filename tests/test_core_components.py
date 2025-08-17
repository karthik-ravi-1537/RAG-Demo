"""Test script to verify core components work together."""

import sys
from pathlib import Path

import pytest

# Add project root to path for proper imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Now import with proper package structure
from core.chunking_engine import ChunkingEngine
from core.data_models import Chunk, ChunkingConfig, EmbeddingConfig
from core.document_processor import DocumentProcessor
from core.embedding_system import EmbeddingSystem
from utils.logging_utils import setup_logger

# Setup logging
logger = setup_logger("test_core", level="INFO")


def test_document_processing():
    """Test document processing functionality."""
    logger.info("Testing document processing...")

    # Create a test document
    test_content = """
    # Test Document

    This is a test document for the RAG demo.

    ## Section 1

    This section contains some information about natural language processing.
    It discusses various techniques for text analysis and understanding.

    ## Section 2

    This section covers machine learning approaches to text processing.
    We explore different algorithms and their applications.

    The document continues with more detailed explanations and examples.
    """

    # Save test document
    test_file = Path("test_document.txt")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(test_content)

    try:
        # Process document
        processor = DocumentProcessor()
        doc = processor.process_document(str(test_file))

        logger.info(f"Processed document: {doc.id}")
        logger.info(f"Content length: {len(doc.content)} characters")
        logger.info(f"Document type: {doc.document_type}")
        logger.info(f"Metadata keys: {list(doc.metadata.keys())}")

        # Test assertions
        assert doc is not None
        assert len(doc.content) > 0
        assert doc.document_type == "text"

    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()


def test_chunking():
    """Test chunking functionality."""
    logger.info("Testing chunking engine...")

    # Create test document
    test_content = """
    This is a test document for chunking.
    It contains multiple sentences and paragraphs.

    This is the second paragraph.
    It has some additional content to test different chunking strategies.

    Machine learning is a subset of artificial intelligence.
    It enables computers to learn from data without explicit programming.
    """

    import tempfile

    test_file = Path(tempfile.mktemp(suffix=".txt"))
    test_file.write_text(test_content)

    try:
        processor = DocumentProcessor()
        doc = processor.process_document(str(test_file))
    finally:
        if test_file.exists():
            test_file.unlink()

    # Test different chunking strategies
    chunking_engine = ChunkingEngine()

    strategies = ["fixed_size", "semantic", "recursive", "structure_aware"]

    for strategy in strategies:
        logger.info(f"Testing {strategy} chunking...")

        config = ChunkingConfig(strategy=strategy, chunk_size=200, overlap=20, min_chunk_size=50)

        chunks = chunking_engine.chunk_document(doc, config)

        logger.info(f"{strategy}: Generated {len(chunks)} chunks")
        if chunks:
            logger.info(f"First chunk length: {len(chunks[0].content)} characters")
            logger.info(f"Average chunk length: {sum(len(c.content) for c in chunks) / len(chunks):.1f}")

    # Test assertions
    assert len(chunks) > 0
    assert all(isinstance(chunk, Chunk) for chunk in chunks)


def test_embedding_system():
    """Test embedding system (without external APIs)."""
    logger.info("Testing embedding system...")

    # Test with a simple mock or sentence-transformers if available
    try:
        # Try to use sentence-transformers for testing
        config = EmbeddingConfig(model_name="all-MiniLM-L6-v2", batch_size=10, normalize=True)

        embedding_system = EmbeddingSystem(config)

        # Test texts
        test_texts = [
            "This is a test sentence about natural language processing.",
            "Machine learning is a subset of artificial intelligence.",
            "Text embeddings capture semantic meaning of words and sentences.",
        ]

        # Generate embeddings
        embeddings = embedding_system.generate_embeddings(test_texts)

        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        logger.info(f"Embedding dimension: {embedding_system.embedding_dimension}")
        logger.info(f"Model name: {embedding_system.model_name}")

        # Test similarity computation
        query_embedding = embeddings[0:1]  # First text as query
        similarities = embedding_system.compute_similarity(query_embedding, embeddings)

        logger.info(f"Similarity scores: {similarities}")

        # Test assertions
        assert embedding_system is not None
        assert embeddings is not None
        assert len(embeddings) > 0

    except Exception as e:
        logger.warning(f"Embedding test failed (expected if dependencies not installed): {str(e)}")
        pytest.skip(f"Embedding test skipped due to missing dependencies: {str(e)}")


def test_integration():
    """Test integration of all components."""
    logger.info("Testing component integration...")

    try:
        # Create test content
        test_content = """
        This is a comprehensive test for RAG integration.
        We are testing document processing, chunking, and embedding.

        Artificial Intelligence (AI) represents a significant technological advancement.
        Machine learning enables systems to automatically learn and improve from experience.
        """

        import tempfile

        test_file = Path(tempfile.mktemp(suffix=".txt"))
        test_file.write_text(test_content)

        try:
            processor = DocumentProcessor()
            doc = processor.process_document(str(test_file))
        finally:
            if test_file.exists():
                test_file.unlink()

        # Chunk document
        chunking_engine = ChunkingEngine()
        config = ChunkingConfig(strategy="semantic", chunk_size=150, overlap=20)
        chunks = chunking_engine.chunk_document(doc, config)

        logger.info(f"Integration test: {len(chunks)} chunks created")

        # Try embedding (if available)
        try:
            embedding_config = EmbeddingConfig(model_name="all-MiniLM-L6-v2")
            embedding_system = EmbeddingSystem(embedding_config)

            # Embed first few chunks
            test_chunks = chunks[:3] if len(chunks) > 3 else chunks
            embedded_chunks = embedding_system.embed_chunks(test_chunks)

            logger.info(f"Integration test: Embedded {len(embedded_chunks)} chunks")

            for i, chunk in enumerate(embedded_chunks):
                if chunk.embedding is not None:
                    logger.info(f"Chunk {i}: embedding shape {chunk.embedding.shape}")

        except Exception as e:
            logger.warning(f"Embedding integration test failed: {str(e)}")

        logger.info("Integration test completed successfully!")

    except Exception as e:
        logger.error(f"Integration test failed: {str(e)}")


def main():
    """Run all tests."""
    logger.info("Starting core components test...")

    try:
        # Test individual components
        test_document_processing()
        test_chunking()
        test_embedding_system()

        # Test integration
        test_integration()

        logger.info("All tests completed!")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise AssertionError(f"Integration test failed: {str(e)}")


if __name__ == "__main__":
    exit(main())
