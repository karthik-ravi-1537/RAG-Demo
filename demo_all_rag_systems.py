"""Comprehensive demo showcasing all RAG implementations with comparison."""

import sys
import tempfile
from pathlib import Path

# Add RAG to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_models import ChunkingConfig, EmbeddingConfig, RAGConfig
from evaluation.rag_comparator import ComparisonQuery, RAGComparator
from graph_rag.graph_rag import GraphRAG
from hierarchical_rag.hierarchical_rag import HierarchicalRAG
from utils.logging_utils import setup_logger
from vanilla_rag.vanilla_rag import VanillaRAG

# Setup logging
logger = setup_logger("rag_demo_all", level="INFO")


def create_comprehensive_test_documents() -> list[str]:
    """Create comprehensive test documents for RAG comparison."""

    # Document 1: AI and Machine Learning Overview
    ai_ml_doc = """
    # Artificial Intelligence and Machine Learning: A Comprehensive Overview

    Artificial Intelligence (AI) represents one of the most transformative technologies of the 21st century.
    It encompasses the development of computer systems capable of performing tasks that typically require
    human intelligence, including visual perception, speech recognition, decision-making, and language translation.

    ## Machine Learning Fundamentals

    Machine Learning (ML) is a subset of AI that enables computers to learn and improve from experience
    without being explicitly programmed. ML algorithms analyze data, identify patterns, and make predictions
    or decisions based on the information they process.

    ### Types of Machine Learning

    **Supervised Learning** uses labeled training data to learn a mapping from inputs to outputs.
    Common applications include:
    - Classification tasks (email spam detection, image recognition)
    - Regression problems (price prediction, sales forecasting)
    - Medical diagnosis and treatment recommendation

    **Unsupervised Learning** finds patterns in data without labeled examples.
    Key techniques include:
    - Clustering algorithms for customer segmentation
    - Dimensionality reduction for data visualization
    - Anomaly detection for fraud prevention

    **Reinforcement Learning** learns through interaction with an environment,
    receiving rewards or penalties for actions taken. Applications include:
    - Game playing (Chess, Go, video games)
    - Autonomous vehicle navigation
    - Resource allocation and optimization

    ## Deep Learning Revolution

    Deep Learning, a subset of machine learning, uses artificial neural networks with multiple layers
    to model and understand complex patterns in data. The breakthrough came with:

    - **Convolutional Neural Networks (CNNs)** for image processing
    - **Recurrent Neural Networks (RNNs)** for sequential data
    - **Transformer architectures** for natural language processing

    ## Real-World Applications

    AI and ML have found applications across numerous industries:

    **Healthcare**: Medical imaging analysis, drug discovery, personalized treatment plans
    **Finance**: Algorithmic trading, risk assessment, fraud detection
    **Transportation**: Autonomous vehicles, route optimization, predictive maintenance
    **Technology**: Search engines, recommendation systems, virtual assistants
    **Manufacturing**: Quality control, predictive maintenance, supply chain optimization
    """

    # Document 2: Natural Language Processing and RAG
    nlp_rag_doc = """
    # Natural Language Processing and Retrieval-Augmented Generation

    Natural Language Processing (NLP) is a subfield of artificial intelligence that focuses on
    the interaction between computers and human language. It combines computational linguistics
    with machine learning and deep learning to help computers understand, interpret, and generate
    human language in a valuable way.

    ## Core NLP Tasks and Techniques

    ### Text Processing and Understanding

    **Tokenization** breaks text into individual words, phrases, or other meaningful elements.
    Modern tokenizers like WordPiece and SentencePiece handle subword units for better
    vocabulary coverage and handling of out-of-vocabulary words.

    **Part-of-Speech Tagging** identifies grammatical roles of words in sentences,
    enabling better understanding of sentence structure and meaning.

    **Named Entity Recognition (NER)** identifies and classifies named entities in text,
    such as people, organizations, locations, dates, and other important information.

    **Dependency Parsing** analyzes grammatical structure by identifying relationships
    between words in a sentence, creating a tree structure that represents syntactic dependencies.

    ### Advanced NLP Applications

    **Sentiment Analysis** determines the emotional tone of text, widely used in:
    - Social media monitoring and brand reputation management
    - Customer feedback analysis and product reviews
    - Market research and consumer behavior studies

    **Text Classification** categorizes documents by topic, intent, or other criteria:
    - Email filtering and spam detection
    - News article categorization
    - Legal document classification

    **Question Answering** systems provide direct answers to questions based on text:
    - Search engines with direct answer features
    - Customer service chatbots
    - Educational and research assistance tools

    ## The Rise of Large Language Models

    The development of large language models has revolutionized NLP:

    **BERT (Bidirectional Encoder Representations from Transformers)** introduced
    bidirectional context understanding, significantly improving performance on
    various NLP tasks through pre-training on large text corpora.

    **GPT (Generative Pre-trained Transformer)** series demonstrated the power
    of autoregressive language modeling, leading to impressive text generation
    capabilities and few-shot learning abilities.

    **T5 (Text-to-Text Transfer Transformer)** unified various NLP tasks under
    a single text-to-text framework, simplifying model architecture and training.

    ## Retrieval-Augmented Generation (RAG)

    RAG represents a significant advancement in combining information retrieval with
    text generation to create more accurate and informative AI responses.

    ### How RAG Works

    The RAG process involves several sophisticated steps:

    1. **Document Processing**: Source documents are processed and segmented into
       manageable chunks while preserving semantic coherence and context.

    2. **Embedding Generation**: Text chunks are converted into high-dimensional
       vector representations that capture semantic meaning and relationships.

    3. **Query Processing**: User queries are similarly embedded into the same
       vector space for semantic similarity comparison.

    4. **Similarity Search**: The system finds the most relevant chunks using
       vector similarity measures like cosine similarity or dot product.

    5. **Context Assembly**: Retrieved chunks are assembled into coherent context
       while managing token limits and maintaining relevance.

    6. **Response Generation**: Language models use the assembled context to
       generate accurate, grounded responses that cite specific sources.

    ### Advantages of RAG Systems

    **Access to Current Information**: RAG systems can incorporate up-to-date
    information not present in the model's training data, ensuring responses
    reflect the latest knowledge and developments.

    **Improved Factual Accuracy**: By grounding responses in retrieved factual
    content, RAG systems significantly reduce hallucination and improve
    the reliability of generated information.

    **Source Transparency**: RAG systems can provide citations and source
    attribution, allowing users to verify information and explore topics
    in greater depth.

    **Domain Customization**: RAG systems can be easily adapted to specific
    domains or knowledge bases without requiring expensive model retraining.

    **Reduced Computational Requirements**: Instead of training massive models
    with all knowledge embedded, RAG systems can use smaller models with
    external knowledge retrieval.

    ### RAG Variants and Evolution

    **Vanilla RAG** uses basic similarity-based retrieval with straightforward
    chunk selection and context formatting.

    **Hierarchical RAG** creates multi-level document representations with
    summaries at different granularities, enabling more sophisticated retrieval.

    **Graph RAG** constructs knowledge graphs from documents, enabling
    relationship-aware retrieval and more connected information discovery.

    **Multi-modal RAG** extends beyond text to handle images, tables, and
    other media types in a unified retrieval and generation framework.
    """

    # Document 3: Future of AI and Emerging Technologies
    future_ai_doc = """
    # The Future of Artificial Intelligence: Emerging Technologies and Trends

    As we advance deeper into the 21st century, artificial intelligence continues to evolve
    at an unprecedented pace. The convergence of multiple technological trends is creating
    new possibilities and challenges that will shape the future of human-AI interaction.

    ## Emerging AI Technologies

    ### Multimodal AI Systems

    The next generation of AI systems will seamlessly integrate multiple modalities:

    **Vision-Language Models** like CLIP and DALL-E demonstrate the power of
    connecting visual and textual understanding, enabling:
    - Image generation from text descriptions
    - Visual question answering and image captioning
    - Cross-modal search and retrieval systems

    **Audio-Visual-Text Integration** creates comprehensive understanding systems:
    - Video analysis with natural language descriptions
    - Automatic subtitle generation and translation
    - Multimodal content creation and editing

    ### Neuromorphic Computing

    Inspired by the human brain's architecture, neuromorphic computing promises:
    - Ultra-low power consumption for AI inference
    - Real-time learning and adaptation capabilities
    - Improved handling of temporal and sequential data

    **Spiking Neural Networks** mimic biological neurons more closely:
    - Event-driven processing reduces computational overhead
    - Natural handling of temporal dynamics in data
    - Potential for brain-computer interface applications

    ### Quantum Machine Learning

    The intersection of quantum computing and machine learning opens new frontiers:

    **Quantum Advantage** in specific ML tasks:
    - Exponential speedup for certain optimization problems
    - Enhanced pattern recognition in high-dimensional spaces
    - Novel approaches to unsupervised learning and clustering

    **Hybrid Classical-Quantum Systems** leverage the best of both worlds:
    - Classical preprocessing with quantum optimization
    - Quantum feature maps for enhanced representation learning
    - Distributed quantum-classical training algorithms

    ## Autonomous Systems and Robotics

    The integration of AI with robotics is creating increasingly sophisticated autonomous systems:

    ### Autonomous Vehicles

    Self-driving technology continues to advance through:
    - Improved sensor fusion and perception systems
    - Better handling of edge cases and unusual scenarios
    - Integration with smart city infrastructure

    **Levels of Autonomy** progress from driver assistance to full autonomy:
    - Level 3: Conditional automation with human oversight
    - Level 4: High automation in specific conditions
    - Level 5: Full automation in all conditions

    ### Robotic Process Automation (RPA)

    AI-powered automation is transforming business processes:
    - Intelligent document processing and data extraction
    - Automated customer service and support
    - Supply chain optimization and inventory management

    ## Ethical AI and Responsible Development

    As AI systems become more powerful and pervasive, ethical considerations become paramount:

    ### Bias and Fairness

    Addressing algorithmic bias requires:
    - Diverse and representative training datasets
    - Fairness-aware machine learning algorithms
    - Regular auditing and bias detection systems
    - Inclusive development teams and perspectives

    ### Privacy and Security

    Protecting user privacy while enabling AI innovation:
    - Federated learning for distributed model training
    - Differential privacy for data protection
    - Homomorphic encryption for secure computation
    - Zero-knowledge proofs for verification without disclosure

    ### Explainable AI (XAI)

    Making AI decisions transparent and interpretable:
    - Model interpretability techniques and tools
    - Natural language explanations of AI decisions
    - Visual representations of model reasoning
    - Regulatory compliance and accountability frameworks

    ## Human-AI Collaboration

    The future lies not in AI replacing humans, but in augmenting human capabilities:

    ### Augmented Intelligence

    AI systems that enhance rather than replace human decision-making:
    - Medical diagnosis support systems
    - Creative writing and content generation assistance
    - Scientific research and hypothesis generation
    - Educational personalization and tutoring

    ### Brain-Computer Interfaces

    Direct neural interfaces promise unprecedented human-AI integration:
    - Thought-controlled computing and communication
    - Memory enhancement and cognitive augmentation
    - Treatment of neurological conditions and disabilities
    - New forms of human-computer interaction

    ## Challenges and Considerations

    ### Technical Challenges

    **Scalability**: Building AI systems that can handle massive scale
    **Robustness**: Creating AI that works reliably in diverse conditions
    **Efficiency**: Developing energy-efficient AI for widespread deployment
    **Generalization**: Building AI that can adapt to new domains and tasks

    ### Societal Impact

    **Employment**: Managing workforce transitions and reskilling
    **Education**: Adapting educational systems for an AI-driven world
    **Governance**: Developing appropriate regulatory frameworks
    **Global Cooperation**: Ensuring equitable access to AI benefits

    The future of AI holds immense promise for solving humanity's greatest challenges,
    from climate change and healthcare to education and scientific discovery.
    Success will depend on our ability to develop AI responsibly, ethically,
    and in service of human flourishing.
    """

    # Create temporary files
    temp_files = []

    documents = [
        ("ai_ml_overview.txt", ai_ml_doc),
        ("nlp_and_rag.txt", nlp_rag_doc),
        ("future_of_ai.txt", future_ai_doc),
    ]

    for _filename, content in documents:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            temp_files.append(f.name)

    return temp_files


def create_test_queries() -> list[ComparisonQuery]:
    """Create test queries for RAG comparison."""
    return [
        ComparisonQuery(
            id="q1",
            query="What is machine learning and what are its main types?",
            category="definition",
            difficulty="basic",
        ),
        ComparisonQuery(
            id="q2",
            query="How does RAG work and what are its advantages?",
            category="technical",
            difficulty="intermediate",
        ),
        ComparisonQuery(
            id="q3",
            query="What are the emerging trends in AI and how might they impact the future?",
            category="analysis",
            difficulty="advanced",
        ),
        ComparisonQuery(
            id="q4",
            query="Compare supervised and unsupervised learning approaches",
            category="comparison",
            difficulty="intermediate",
        ),
        ComparisonQuery(
            id="q5", query="What role do transformers play in modern NLP?", category="technical", difficulty="advanced"
        ),
    ]


def main():
    """Run comprehensive RAG comparison demo."""
    print("🚀 Comprehensive RAG Systems Comparison Demo")
    print("=" * 60)

    # Create configuration
    config = RAGConfig(
        chunking=ChunkingConfig(strategy="semantic", chunk_size=400, overlap=50, min_chunk_size=100),
        embedding=EmbeddingConfig(model_name="all-MiniLM-L6-v2", normalize=True),
        retrieval_top_k=3,
        similarity_threshold=0.2,
    )

    print("📋 Configuration:")
    print(f"   Chunking: {config.chunking.strategy} (size: {config.chunking.chunk_size})")
    print(f"   Embedding: {config.embedding.model_name}")
    print(f"   Retrieval: top-{config.retrieval_top_k} chunks")
    print()

    # Create test documents
    print("📄 Creating comprehensive test documents...")
    temp_files = create_comprehensive_test_documents()

    try:
        # Initialize RAG systems
        print("🔧 Initializing RAG systems...")

        vanilla_rag = VanillaRAG(config)
        hierarchical_rag = HierarchicalRAG(config)
        graph_rag = GraphRAG(config)

        # Add documents to all systems
        print("📚 Adding documents to all RAG systems...")

        print("   Adding to Vanilla RAG...")
        vanilla_rag.add_documents(temp_files)

        print("   Adding to Hierarchical RAG...")
        hierarchical_rag.add_documents(temp_files)

        print("   Adding to Graph RAG...")
        graph_rag.add_documents(temp_files)

        # Show system statistics
        print("\n📊 System Statistics:")
        print("-" * 40)

        vanilla_stats = vanilla_rag.get_stats()
        print(f"Vanilla RAG: {vanilla_stats['total_documents']} docs, {vanilla_stats['total_chunks']} chunks")

        hierarchical_stats = hierarchical_rag.get_stats()
        print(
            f"Hierarchical RAG: {hierarchical_stats['documents_processed']} docs, {hierarchical_stats['total_nodes']} nodes"
        )

        graph_stats = graph_rag.get_stats()
        print(
            f"Graph RAG: {graph_stats['documents_processed']} docs, {graph_stats['entities_extracted']} entities, {graph_stats['relationships_extracted']} relationships"
        )

        # Initialize comparator
        print("\n🔬 Setting up RAG Comparator...")
        comparator = RAGComparator()

        # Register RAG systems
        comparator.register_rag_system("Vanilla RAG", vanilla_rag)
        comparator.register_rag_system("Hierarchical RAG", hierarchical_rag)
        comparator.register_rag_system("Graph RAG", graph_rag)

        # Create test queries
        test_queries = create_test_queries()

        print(f"\n🤖 Running comparison on {len(test_queries)} queries...")
        print("=" * 60)

        # Run comparisons
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Query {i}: {query.query}")
            print(f"Category: {query.category}, Difficulty: {query.difficulty}")
            print("-" * 50)

            # Compare all systems
            comparison_result = comparator.compare_single_query(query)

            # Display results
            print(f"🏆 Winner: {comparison_result.winner}")
            print("📈 Performance Summary:")

            for rag_name, metrics in comparison_result.metrics.items():
                if metrics.error:
                    print(f"   {rag_name}: ERROR - {metrics.error}")
                else:
                    print(f"   {rag_name}:")
                    print(f"      Time: {metrics.total_time:.3f}s")
                    print(f"      Confidence: {metrics.confidence_score:.3f}")
                    print(f"      Chunks: {metrics.chunks_retrieved}")
                    print(f"      Response Length: {metrics.response_length} chars")

            # Show comparison summary
            if comparison_result.comparison_summary:
                perf_summary = comparison_result.comparison_summary.get("performance_summary", {})
                if perf_summary:
                    print("\n⚡ Speed Analysis:")
                    print(
                        f"   Fastest: {perf_summary.get('fastest_system')} ({perf_summary.get('fastest_time', 0):.3f}s)"
                    )
                    print(
                        f"   Slowest: {perf_summary.get('slowest_system')} ({perf_summary.get('slowest_time', 0):.3f}s)"
                    )

                quality_summary = comparison_result.comparison_summary.get("quality_summary", {})
                if quality_summary:
                    print(
                        f"   Quality Leader: {quality_summary.get('highest_confidence_system')} ({quality_summary.get('highest_confidence_score', 0):.3f})"
                    )

            # Show sample responses (truncated)
            print("\n💬 Sample Responses:")
            for rag_name, result in comparison_result.results.items():
                response_preview = result.response[:200] + "..." if len(result.response) > 200 else result.response
                print(f"   {rag_name}: {response_preview}")

        # Generate aggregate analysis
        print("\n" + "=" * 60)
        print("📈 AGGREGATE ANALYSIS")
        print("=" * 60)

        aggregate_analysis = comparator.get_aggregate_analysis()

        for rag_name, analysis in aggregate_analysis.items():
            print(f"\n🔍 {rag_name}:")
            print(f"   Win Rate: {analysis['win_rate']:.1%}")
            print(f"   Avg Response Time: {analysis['total_times']['mean']:.3f}s")
            print(f"   Avg Confidence: {analysis['confidence_scores']['mean']:.3f}")
            print(f"   Avg Chunks Retrieved: {analysis['chunks_retrieved']['mean']:.1f}")
            print(f"   Queries/Second: {analysis['queries_per_second']:.2f}")

        # Generate comprehensive report
        print("\n📋 Generating comprehensive comparison report...")
        report = comparator.generate_comparison_report()

        print("📊 Overall Rankings:")
        rankings = report["rankings"]["overall"]
        for rag_name, rank_info in sorted(rankings.items(), key=lambda x: x[1]["rank"]):
            print(f"   #{rank_info['rank']}: {rag_name} (score: {rank_info['score']:.3f})")

        # Show system capabilities summary
        print("\n🎯 System Capabilities Summary:")
        print("   Vanilla RAG: Fast, simple, good baseline performance")
        print("   Hierarchical RAG: Structure-aware, good for complex documents")
        print("   Graph RAG: Relationship-aware, best for connected information")

        print("\n✨ Demo completed successfully!")
        print("\n💡 Key Insights:")
        print("   - Different RAG approaches excel in different scenarios")
        print("   - Graph RAG provides richer context through entity relationships")
        print("   - Hierarchical RAG better handles document structure")
        print("   - Vanilla RAG offers good performance with simplicity")
        print("   - Choice depends on your specific use case and requirements")

        return 0

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
            except Exception:
                pass


if __name__ == "__main__":
    exit(main())
