"""Demo script for Vanilla RAG implementation."""

import sys
from pathlib import Path
import tempfile

# Add RAG to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vanilla_rag.vanilla_rag import VanillaRAG
from core.data_models import RAGConfig, ChunkingConfig, EmbeddingConfig
from utils.logging_utils import setup_logger

# Setup logging
logger = setup_logger("vanilla_rag_demo", level="INFO")


def create_sample_documents():
    """Create sample documents for the demo."""
    
    # Sample document about AI
    ai_doc = """
    # Introduction to Artificial Intelligence
    
    Artificial Intelligence (AI) represents one of the most transformative technologies of our time.
    It encompasses the development of computer systems that can perform tasks typically requiring
    human intelligence, such as visual perception, speech recognition, decision-making, and
    language translation.
    
    ## Machine Learning
    
    Machine Learning is a subset of AI that enables computers to learn and improve from experience
    without being explicitly programmed. It uses algorithms to analyze data, identify patterns,
    and make predictions or decisions.
    
    ### Types of Machine Learning:
    - Supervised Learning: Uses labeled data to train models
    - Unsupervised Learning: Finds patterns in unlabeled data  
    - Reinforcement Learning: Learns through trial and error with rewards
    
    ## Applications
    
    AI has numerous real-world applications including:
    - Healthcare: Medical diagnosis and drug discovery
    - Transportation: Autonomous vehicles and traffic optimization
    - Finance: Fraud detection and algorithmic trading
    - Entertainment: Recommendation systems and content generation
    """
    
    # Sample document about RAG
    rag_doc = """
    # Retrieval-Augmented Generation (RAG)
    
    RAG is a powerful technique that combines information retrieval with text generation
    to create more accurate and informative AI responses. Instead of relying solely on
    pre-trained knowledge, RAG systems can access and utilize external information sources.
    
    ## How RAG Works
    
    The RAG process involves several key steps:
    
    1. Document Processing: Convert source documents into searchable chunks
    2. Embedding Generation: Create vector representations of text chunks
    3. Query Processing: Convert user questions into vector embeddings
    4. Similarity Search: Find the most relevant chunks using vector similarity
    5. Context Assembly: Combine retrieved chunks into coherent context
    6. Response Generation: Use the context to generate accurate answers
    
    ## Benefits of RAG
    
    - Access to current information not in training data
    - Improved factual accuracy through grounding in source material
    - Transparency through source attribution
    - Customizable knowledge bases for specific domains
    - Reduced hallucination in AI responses
    
    ## RAG Variants
    
    Different RAG approaches offer various advantages:
    - Vanilla RAG: Simple similarity-based retrieval
    - Hierarchical RAG: Multi-level document understanding
    - Graph RAG: Knowledge graph-based connections
    - Multi-modal RAG: Handling text, images, and other media
    """
    
    # Create temporary files
    temp_files = []
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(ai_doc)
        temp_files.append(f.name)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(rag_doc)
        temp_files.append(f.name)
    
    return temp_files


def main():
    """Run the Vanilla RAG demo."""
    print("🚀 Vanilla RAG Demo")
    print("=" * 50)
    
    # Create configuration
    config = RAGConfig(
        chunking=ChunkingConfig(
            strategy="semantic",
            chunk_size=400,
            overlap=50,
            min_chunk_size=100
        ),
        embedding=EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",  # Local model for demo
            normalize=True
        ),
        retrieval_top_k=3,
        similarity_threshold=0.2
    )
    
    print(f"📋 Configuration:")
    print(f"   Chunking: {config.chunking.strategy} (size: {config.chunking.chunk_size})")
    print(f"   Embedding: {config.embedding.model_name}")
    print(f"   Retrieval: top-{config.retrieval_top_k} chunks")
    print()
    
    # Initialize RAG system
    print("🔧 Initializing Vanilla RAG system...")
    rag = VanillaRAG(config)
    
    # Create and add sample documents
    print("📄 Creating sample documents...")
    temp_files = create_sample_documents()
    
    try:
        print("📚 Adding documents to RAG system...")
        rag.add_documents(temp_files)
        
        # Show system stats
        stats = rag.get_stats()
        print(f"✅ System ready!")
        print(f"   Documents: {stats['total_documents']}")
        print(f"   Chunks: {stats['total_chunks']}")
        print(f"   Embedding model: {stats['embedding_model']}")
        print()
        
        # Demo queries
        demo_queries = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What are the benefits of RAG?",
            "What are the different types of machine learning?",
            "How does RAG improve AI responses?"
        ]
        
        print("🤖 Running demo queries...")
        print("=" * 50)
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n📝 Query {i}: {query}")
            print("-" * 40)
            
            # Process query
            result = rag.query(query)
            
            # Show results
            print(f"⏱️  Processing time: {result.total_time:.3f}s")
            print(f"📊 Confidence: {result.confidence_score:.3f}")
            print(f"📄 Chunks retrieved: {len(result.retrieved_chunks)}")
            
            # Show similarity scores
            similarities = [chunk.similarity_score for chunk in result.retrieved_chunks]
            if similarities:
                print(f"🎯 Similarity scores: {[f'{s:.3f}' for s in similarities]}")
            
            print(f"\n💬 Response:")
            print(result.response)
            
            # Show source information
            sources = list(set(chunk.source_document for chunk in result.retrieved_chunks))
            if sources:
                print(f"\n📚 Sources: {len(sources)} document(s)")
        
        # Show final system statistics
        final_stats = rag.get_stats()
        print("\n" + "=" * 50)
        print("📈 Final System Statistics:")
        print(f"   Queries processed: {final_stats['queries_processed']}")
        print(f"   Average retrieval time: {final_stats['avg_retrieval_time']:.3f}s")
        print(f"   Average generation time: {final_stats['avg_generation_time']:.3f}s")
        print(f"   Total chunks: {final_stats['total_chunks']}")
        
        print("\n✨ Demo completed successfully!")
        print("\n💡 Next steps:")
        print("   - Try your own documents with: rag.add_documents(['your_file.txt'])")
        print("   - Experiment with different chunking strategies")
        print("   - Adjust similarity thresholds and retrieval parameters")
        print("   - Compare with other RAG approaches (coming soon!)")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup temporary files
        import os
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
    
    return 0


if __name__ == "__main__":
    exit(main())