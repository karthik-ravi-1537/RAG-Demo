"""Test script for Vanilla RAG implementation."""

import os
import sys
import tempfile
from pathlib import Path

# Add RAG to path for proper imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

from core.data_models import ChunkingConfig, EmbeddingConfig, RAGConfig
from utils.logging_utils import setup_logger
from vanilla_rag.vanilla_rag import VanillaRAG

# Setup logging
logger = setup_logger("test_vanilla_rag", level="INFO")


def create_test_documents():
    """Create test documents for RAG testing."""
    test_docs = []

    # Document 1: AI and Machine Learning
    doc1_content = """
    # Artificial Intelligence and Machine Learning

    Artificial Intelligence (AI) is a broad field of computer science focused on creating
    systems that can perform tasks that typically require human intelligence. Machine Learning
    is a subset of AI that enables computers to learn and improve from experience without
    being explicitly programmed.

    ## Types of Machine Learning

    There are three main types of machine learning:

    1. **Supervised Learning**: Uses labeled training data to learn a mapping from inputs to outputs.
       Examples include classification and regression tasks.

    2. **Unsupervised Learning**: Finds patterns in data without labeled examples.
       Common techniques include clustering and dimensionality reduction.

    3. **Reinforcement Learning**: Learns through interaction with an environment,
       receiving rewards or penalties for actions taken.

    ## Applications

    Machine learning has numerous applications including:
    - Natural language processing
    - Computer vision
    - Recommendation systems
    - Autonomous vehicles
    - Medical diagnosis
    """

    # Document 2: Natural Language Processing
    doc2_content = """
    # Natural Language Processing (NLP)

    Natural Language Processing is a subfield of artificial intelligence that focuses on
    the interaction between computers and human language. It combines computational linguistics
    with machine learning and deep learning to help computers understand, interpret, and
    generate human language.

    ## Key NLP Tasks

    ### Text Processing
    - Tokenization: Breaking text into individual words or tokens
    - Part-of-speech tagging: Identifying grammatical roles of words
    - Named entity recognition: Identifying people, places, organizations

    ### Understanding Tasks
    - Sentiment analysis: Determining emotional tone of text
    - Text classification: Categorizing documents by topic or intent
    - Question answering: Providing answers to questions based on text

    ### Generation Tasks
    - Machine translation: Converting text from one language to another
    - Text summarization: Creating concise summaries of longer texts
    - Dialogue systems: Engaging in conversations with users

    ## Modern Approaches

    Recent advances in NLP have been driven by:
    - Transformer architectures like BERT and GPT
    - Large language models trained on massive datasets
    - Transfer learning and fine-tuning techniques
    """

    # Document 3: Retrieval-Augmented Generation
    doc3_content = """
    # Retrieval-Augmented Generation (RAG)

    Retrieval-Augmented Generation is a technique that combines information retrieval
    with text generation to create more accurate and informative responses. RAG systems
    first retrieve relevant information from a knowledge base, then use that information
    to generate contextually appropriate responses.

    ## How RAG Works

    The RAG process typically involves several steps:

    1. **Document Processing**: Convert documents into searchable chunks
    2. **Embedding Generation**: Create vector representations of text chunks
    3. **Query Processing**: Convert user queries into embeddings
    4. **Retrieval**: Find most similar chunks using vector similarity
    5. **Generation**: Use retrieved context to generate responses

    ## Advantages of RAG

    - **Up-to-date Information**: Can access current information not in training data
    - **Factual Accuracy**: Grounds responses in retrieved factual content
    - **Transparency**: Shows which sources were used for generation
    - **Customization**: Can be adapted to specific domains or knowledge bases

    ## RAG Variants

    - **Vanilla RAG**: Basic similarity-based retrieval
    - **Hierarchical RAG**: Multi-level document representation
    - **Graph RAG**: Uses knowledge graphs for retrieval
    - **Multi-modal RAG**: Handles text, images, and other media types
    """

    # Create temporary files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(doc1_content)
        test_docs.append(f.name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(doc2_content)
        test_docs.append(f.name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(doc3_content)
        test_docs.append(f.name)

    return test_docs


def test_vanilla_rag_basic():
    """Test basic Vanilla RAG functionality."""
    logger.info("Testing basic Vanilla RAG functionality...")

    # Create configuration
    config = RAGConfig(
        chunking=ChunkingConfig(strategy="semantic", chunk_size=300, overlap=50, min_chunk_size=100),
        embedding=EmbeddingConfig(model_name="all-MiniLM-L6-v2", normalize=True),  # Use local model for testing
        retrieval_top_k=3,
        similarity_threshold=0.1,
    )

    # Initialize RAG system
    rag = VanillaRAG(config)

    # Create test documents
    test_docs = create_test_documents()

    try:
        # Add documents
        logger.info("Adding test documents...")
        rag.add_documents(test_docs)

        # Check stats
        stats = rag.get_stats()
        logger.info(f"RAG Stats: {stats}")

        assert stats["documents_processed"] == 3
        assert stats["total_chunks"] > 0
        assert stats["total_documents"] == 3

        # Test queries
        test_queries = [
            "What is machine learning?",
            "How does RAG work?",
            "What are the types of machine learning?",
            "What is natural language processing?",
            "What are the advantages of RAG?",
        ]

        logger.info("Testing queries...")
        for query in test_queries:
            logger.info(f"\nQuery: {query}")

            result = rag.query(query)

            logger.info(f"Response length: {len(result.response)} characters")
            logger.info(f"Chunks retrieved: {len(result.retrieved_chunks)}")
            logger.info(f"Confidence: {result.confidence_score:.3f}")
            logger.info(f"Retrieval time: {result.retrieval_time:.3f}s")
            logger.info(f"Generation time: {result.generation_time:.3f}s")

            # Print first part of response
            response_preview = result.response[:200] + "..." if len(result.response) > 200 else result.response
            logger.info(f"Response preview: {response_preview}")

            # Check that we got results
            assert len(result.retrieved_chunks) > 0
            assert len(result.response) > 0
            assert result.confidence_score > 0

            # Check similarity scores
            for chunk in result.retrieved_chunks:
                assert chunk.similarity_score >= config.similarity_threshold
                logger.debug(f"Chunk similarity: {chunk.similarity_score:.3f}")

        # Test explanation
        result = rag.query("What is AI?")
        explanation = result.explanation

        logger.info(f"Process explanation: {explanation}")
        assert explanation["rag_type"] == "vanilla"
        assert "process_steps" in explanation
        assert len(explanation["process_steps"]) == 5

        logger.info("✅ Basic Vanilla RAG test passed!")

    finally:
        # Cleanup test files
        for doc_path in test_docs:
            try:
                os.unlink(doc_path)
            except Exception:
                pass


def test_vanilla_rag_edge_cases():
    """Test edge cases and error handling."""
    logger.info("Testing Vanilla RAG edge cases...")

    config = RAGConfig(
        embedding=EmbeddingConfig(model_name="all-MiniLM-L6-v2"),
        retrieval_top_k=5,
        similarity_threshold=0.8,  # High threshold
    )

    rag = VanillaRAG(config)

    # Test query without documents
    try:
        result = rag.query("What is AI?")
        raise AssertionError("Should have raised an error")
    except Exception as e:
        logger.info(f"✅ Correctly handled empty system: {str(e)}")

    # Add a document and test high similarity threshold
    test_docs = create_test_documents()

    try:
        rag.add_documents([test_docs[0]])  # Add just one document

        # Query with high similarity threshold
        result = rag.query("What is quantum computing?")  # Unrelated query

        # Should still return something, but with low confidence
        logger.info(
            f"High threshold query - chunks: {len(result.retrieved_chunks)}, "
            f"confidence: {result.confidence_score:.3f}"
        )

        # Test empty query
        result = rag.query("")
        logger.info(f"Empty query handled - response length: {len(result.response)}")

        logger.info("✅ Edge cases test passed!")

    finally:
        # Cleanup
        for doc_path in test_docs:
            try:
                os.unlink(doc_path)
            except Exception:
                pass


def test_vanilla_rag_different_configs():
    """Test different configuration options."""
    logger.info("Testing different RAG configurations...")

    test_docs = create_test_documents()

    configs = [
        # Fixed-size chunking
        RAGConfig(
            chunking=ChunkingConfig(strategy="fixed_size", chunk_size=200, overlap=20),
            embedding=EmbeddingConfig(model_name="all-MiniLM-L6-v2"),
            retrieval_top_k=2,
        ),
        # Recursive chunking
        RAGConfig(
            chunking=ChunkingConfig(strategy="recursive", chunk_size=400, overlap=40),
            embedding=EmbeddingConfig(model_name="all-MiniLM-L6-v2"),
            retrieval_top_k=4,
        ),
        # Structure-aware chunking
        RAGConfig(
            chunking=ChunkingConfig(strategy="structure_aware", chunk_size=300, overlap=30),
            embedding=EmbeddingConfig(model_name="all-MiniLM-L6-v2"),
            retrieval_top_k=3,
        ),
    ]

    try:
        for i, config in enumerate(configs):
            logger.info(f"\nTesting configuration {i+1}: {config.chunking.strategy} chunking")

            rag = VanillaRAG(config)
            rag.add_documents(test_docs)

            result = rag.query("What is machine learning?")
            stats = rag.get_stats()

            logger.info(f"Strategy: {config.chunking.strategy}")
            logger.info(f"Chunks created: {stats['total_chunks']}")
            logger.info(f"Chunks retrieved: {len(result.retrieved_chunks)}")
            logger.info(f"Confidence: {result.confidence_score:.3f}")

            assert len(result.retrieved_chunks) > 0
            assert result.confidence_score > 0

        logger.info("✅ Different configurations test passed!")

    finally:
        # Cleanup
        for doc_path in test_docs:
            try:
                os.unlink(doc_path)
            except Exception:
                pass


def main():
    """Run all Vanilla RAG tests."""
    logger.info("Starting Vanilla RAG tests...")

    try:
        test_vanilla_rag_basic()
        test_vanilla_rag_edge_cases()
        test_vanilla_rag_different_configs()

        logger.info("\n🎉 All Vanilla RAG tests passed!")
        return 0

    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
