"""Graph-RAG implementation with knowledge graph-based retrieval."""

import time
from collections import defaultdict
from typing import Any

import numpy as np
from core.chunking_engine import ChunkingEngine
from core.data_models import Chunk, ProcessedDocument, RAGConfig
from core.document_processor import DocumentProcessor
from core.embedding_system import EmbeddingSystem
from core.exceptions import GenerationError, RAGException, RetrievalError
from core.interfaces import BaseRAG
from utils.logging_utils import get_logger

from graph_rag.entity_extractor import EntityExtractor, ExtractionResult
from graph_rag.knowledge_graph import KnowledgeGraph, KnowledgeGraphBuilder

logger = get_logger(__name__)


class GraphRAG(BaseRAG):
    """
    Graph-RAG implementation using knowledge graph-based retrieval.

    This approach:
    1. Extracts entities and relationships from documents
    2. Builds a knowledge graph from extracted information
    3. Identifies query entities and maps them to the graph
    4. Traverses the graph to find connected information
    5. Assembles context from graph paths and related chunks
    6. Generates responses leveraging graph structure
    """

    def __init__(self, config: RAGConfig = None):
        """Initialize Graph-RAG system."""
        self.config = config or RAGConfig()

        # Initialize core components
        self.document_processor = DocumentProcessor()
        self.chunking_engine = ChunkingEngine()
        self.embedding_system = EmbeddingSystem(self.config.embedding)

        # Initialize graph-specific components
        self.entity_extractor = EntityExtractor()
        self.graph_builder = KnowledgeGraphBuilder()
        self.knowledge_graph: KnowledgeGraph | None = None

        # Storage for processed data
        self.documents: list[ProcessedDocument] = []
        self.chunks: list[Chunk] = []
        self.chunk_index: dict[str, Chunk] = {}

        # Graph-specific storage
        self.extraction_results: list[ExtractionResult] = []

        # Performance tracking
        self.stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "entities_extracted": 0,
            "relationships_extracted": 0,
            "graph_nodes": 0,
            "graph_edges": 0,
            "queries_processed": 0,
            "total_retrieval_time": 0.0,
            "total_generation_time": 0.0,
            "avg_graph_traversal_time": 0.0,
        }

        logger.info(f"Initialized GraphRAG with {self.embedding_system.model_name} embeddings")

    def add_documents(self, document_paths: list[str]) -> None:
        """Add documents to the Graph-RAG system."""
        logger.info(f"Adding {len(document_paths)} documents to GraphRAG")

        try:
            # Process documents
            new_documents = []
            for doc_path in document_paths:
                try:
                    doc = self.document_processor.process_document(doc_path)
                    new_documents.append(doc)
                    logger.debug(f"Processed document: {doc.id}")
                except Exception as e:
                    logger.error(f"Failed to process document {doc_path}: {str(e)}")
                    continue

            if not new_documents:
                raise RAGException("No documents were successfully processed")

            # Chunk documents
            new_chunks = []
            for doc in new_documents:
                try:
                    doc_chunks = self.chunking_engine.chunk_document(doc, self.config.chunking)
                    new_chunks.extend(doc_chunks)
                    logger.debug(f"Created {len(doc_chunks)} chunks for document {doc.id}")
                except Exception as e:
                    logger.error(f"Failed to chunk document {doc.id}: {str(e)}")
                    continue

            if not new_chunks:
                raise RAGException("No chunks were successfully created")

            # Generate embeddings for chunks
            embedded_chunks = self.embedding_system.embed_chunks(new_chunks)

            # Extract entities and relationships
            extraction_results = []
            for doc in new_documents:
                doc_chunks = [c for c in embedded_chunks if c.source_document == doc.id]
                try:
                    extraction_result = self.entity_extractor.extract_from_document(doc, doc_chunks)
                    extraction_results.append(extraction_result)
                    logger.debug(
                        f"Extracted {len(extraction_result.entities)} entities and "
                        f"{len(extraction_result.relationships)} relationships from {doc.id}"
                    )
                except Exception as e:
                    logger.error(f"Failed to extract from document {doc.id}: {str(e)}")
                    continue

            # Build knowledge graph
            self._build_knowledge_graph(extraction_results)

            # Update storage
            self.documents.extend(new_documents)
            self.chunks.extend(embedded_chunks)
            self.extraction_results.extend(extraction_results)

            # Update chunk index
            for chunk in embedded_chunks:
                self.chunk_index[chunk.id] = chunk

            # Update stats
            self.stats["documents_processed"] += len(new_documents)
            self.stats["chunks_created"] += len(embedded_chunks)
            self.stats["entities_extracted"] += sum(len(r.entities) for r in extraction_results)
            self.stats["relationships_extracted"] += sum(len(r.relationships) for r in extraction_results)

            if self.knowledge_graph:
                graph_stats = self.knowledge_graph.get_statistics()
                self.stats["graph_nodes"] = graph_stats["nodes"]
                self.stats["graph_edges"] = graph_stats["edges"]

            logger.info(
                f"Successfully added {len(new_documents)} documents to GraphRAG with "
                f"{self.stats['entities_extracted']} entities and "
                f"{self.stats['relationships_extracted']} relationships"
            )

        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise RAGException(f"Document addition failed: {str(e)}")

    def retrieve(self, query: str, top_k: int = None) -> list[Chunk]:
        """Retrieve relevant chunks using graph-based approach."""
        if top_k is None:
            top_k = self.config.retrieval_top_k

        if not self.knowledge_graph:
            raise RetrievalError("Knowledge graph not available")

        if not self.chunks:
            raise RetrievalError("No documents have been added to the system")

        try:
            start_time = time.time()

            # Stage 1: Identify query entities
            query_entities = self._identify_query_entities(query)

            # Stage 2: Graph traversal to find related entities
            related_entities = self._traverse_graph_for_entities(query_entities, query)

            # Stage 3: Retrieve chunks associated with related entities
            candidate_chunks = self._get_entity_chunks(related_entities)

            # Stage 4: Rank chunks using combined similarity and graph relevance
            ranked_chunks = self._rank_chunks_with_graph_context(
                query, candidate_chunks, query_entities, related_entities
            )

            # Stage 5: Select top-k chunks
            selected_chunks = ranked_chunks[:top_k]

            retrieval_time = time.time() - start_time
            self.stats["total_retrieval_time"] += retrieval_time
            self.stats["avg_graph_traversal_time"] = (
                self.stats["avg_graph_traversal_time"] * self.stats["queries_processed"] + retrieval_time
            ) / (self.stats["queries_processed"] + 1)

            logger.info(
                f"Graph-RAG retrieval: {len(query_entities)} query entities -> "
                f"{len(related_entities)} related entities -> {len(selected_chunks)} chunks in {retrieval_time:.3f}s"
            )

            return selected_chunks

        except Exception as e:
            logger.error(f"Graph-RAG retrieval failed: {str(e)}")
            raise RetrievalError(f"Failed to retrieve chunks: {str(e)}")

    def generate_response(self, query: str, context_chunks: list[Chunk]) -> str:
        """Generate response using graph-enhanced context."""
        try:
            start_time = time.time()

            # Format context with graph information
            context = self._format_graph_context(query, context_chunks)

            # Generate response (mock implementation)
            response = self._generate_graph_response(query, context, context_chunks)

            generation_time = time.time() - start_time
            self.stats["total_generation_time"] += generation_time

            logger.info(f"Generated graph-enhanced response in {generation_time:.3f}s")

            return response

        except Exception as e:
            logger.error(f"Graph-RAG response generation failed: {str(e)}")
            raise GenerationError(f"Failed to generate response: {str(e)}")

    def _build_knowledge_graph(self, extraction_results: list[ExtractionResult]) -> None:
        """Build knowledge graph from extraction results."""
        # Combine all entities and relationships
        all_entities = []
        all_relationships = []

        for result in extraction_results:
            all_entities.extend(result.entities)
            all_relationships.extend(result.relationships)

        # Build the graph
        self.knowledge_graph = self.graph_builder.build_from_extractions(all_entities, all_relationships)

        logger.info(
            f"Built knowledge graph with {len(all_entities)} entities and {len(all_relationships)} relationships"
        )

    def _identify_query_entities(self, query: str) -> list[str]:
        """Identify entities mentioned in the query."""
        if not self.knowledge_graph:
            return []

        query_entities = []
        query_lower = query.lower()

        # Direct entity name matching
        for entity_name in self.knowledge_graph.model.entities.keys():
            entity_name_lower = entity_name.lower()

            # Exact match
            if entity_name_lower in query_lower:
                query_entities.append(entity_name)
                continue

            # Partial match for multi-word entities
            entity_words = entity_name_lower.split()
            if len(entity_words) > 1:
                if all(word in query_lower for word in entity_words):
                    query_entities.append(entity_name)
                    continue

            # Check mentions
            entity = self.knowledge_graph.model.entities[entity_name]
            for mention in entity.mentions:
                if mention.lower() in query_lower:
                    query_entities.append(entity_name)
                    break

        # If no direct matches, use embedding similarity
        if not query_entities:
            query_entities = self._find_similar_entities(query, top_k=3)

        logger.debug(f"Identified query entities: {query_entities}")
        return query_entities

    def _find_similar_entities(self, query: str, top_k: int = 3) -> list[str]:
        """Find entities similar to query using embeddings."""
        if not self.knowledge_graph:
            return []

        # Generate query embedding
        query_embedding = self.embedding_system.generate_embeddings([query])[0]

        # Get entity embeddings (use entity names as text)
        entity_names = list(self.knowledge_graph.model.entities.keys())
        if not entity_names:
            return []

        entity_embeddings = self.embedding_system.generate_embeddings(entity_names)

        # Compute similarities
        similarities = self.embedding_system.compute_similarity(query_embedding.reshape(1, -1), entity_embeddings)

        # Get top-k most similar
        top_indices = np.argsort(similarities)[::-1][:top_k]

        similar_entities = []
        for idx in top_indices:
            if similarities[idx] >= 0.3:  # Minimum similarity threshold
                similar_entities.append(entity_names[idx])

        return similar_entities

    def _traverse_graph_for_entities(self, query_entities: list[str], query: str) -> list[tuple[str, float]]:
        """Traverse graph to find entities related to query entities."""
        if not self.knowledge_graph or not query_entities:
            return []

        related_entities = set()
        entity_scores = {}

        # For each query entity, find related entities
        for query_entity in query_entities:
            if query_entity not in self.knowledge_graph.model.entities:
                continue

            # Get directly connected entities
            direct_related = self.knowledge_graph.find_related_entities(query_entity, max_depth=2, relation_types=None)

            # Add to results with scores
            for entity_name, importance in direct_related:
                related_entities.add(entity_name)
                # Combine importance with query relevance
                if entity_name not in entity_scores:
                    entity_scores[entity_name] = 0.0
                entity_scores[entity_name] = max(entity_scores[entity_name], importance * 0.8)

        # Add query entities themselves with high scores
        for query_entity in query_entities:
            if query_entity in self.knowledge_graph.model.entities:
                related_entities.add(query_entity)
                entity_scores[query_entity] = 1.0

        # Sort by score
        scored_entities = [(entity, entity_scores.get(entity, 0.0)) for entity in related_entities]
        scored_entities.sort(key=lambda x: x[1], reverse=True)

        return scored_entities

    def _get_entity_chunks(self, related_entities: list[tuple[str, float]]) -> list[tuple[Chunk, float]]:
        """Get chunks associated with related entities."""
        chunk_scores = defaultdict(float)
        chunk_entity_count = defaultdict(int)

        for entity_name, entity_score in related_entities:
            if entity_name not in self.knowledge_graph.model.entities:
                continue

            entity = self.knowledge_graph.model.entities[entity_name]

            # Get chunks containing this entity
            for chunk_id in entity.chunk_ids:
                if chunk_id in self.chunk_index:
                    chunk = self.chunk_index[chunk_id]
                    # Accumulate scores from multiple entities
                    chunk_scores[chunk.id] += entity_score * entity.confidence
                    chunk_entity_count[chunk.id] += 1

        # Create list of chunks with scores
        candidate_chunks = []
        for chunk_id, score in chunk_scores.items():
            chunk = self.chunk_index[chunk_id]
            # Boost score based on number of entities in chunk
            boosted_score = score * (1 + 0.1 * chunk_entity_count[chunk_id])
            candidate_chunks.append((chunk, boosted_score))

        return candidate_chunks

    def _rank_chunks_with_graph_context(
        self,
        query: str,
        candidate_chunks: list[tuple[Chunk, float]],
        query_entities: list[str],
        related_entities: list[tuple[str, float]],
    ) -> list[Chunk]:
        """Rank chunks using combined similarity and graph relevance."""
        if not candidate_chunks:
            return []

        # Generate query embedding
        query_embedding = self.embedding_system.generate_embeddings([query])[0]

        # Calculate combined scores
        scored_chunks = []

        for chunk, graph_score in candidate_chunks:
            # Semantic similarity score
            if chunk.embedding is not None:
                similarity_score = self.embedding_system.compute_similarity(
                    query_embedding.reshape(1, -1), chunk.embedding.reshape(1, -1)
                )[0]
            else:
                similarity_score = 0.0

            # Graph connectivity score
            connectivity_score = self._calculate_connectivity_score(chunk, query_entities, related_entities)

            # Combined score: 40% similarity, 40% graph relevance, 20% connectivity
            combined_score = 0.4 * similarity_score + 0.4 * graph_score + 0.2 * connectivity_score

            # Store scores in chunk metadata
            chunk.similarity_score = float(combined_score)
            chunk.metadata["semantic_similarity"] = float(similarity_score)
            chunk.metadata["graph_relevance"] = float(graph_score)
            chunk.metadata["connectivity_score"] = float(connectivity_score)

            scored_chunks.append((chunk, combined_score))

        # Sort by combined score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Filter by threshold
        filtered_chunks = []
        for chunk, score in scored_chunks:
            if score >= self.config.similarity_threshold * 0.7:  # Slightly lower threshold for graph-based
                filtered_chunks.append(chunk)

        return filtered_chunks

    def _calculate_connectivity_score(
        self, chunk: Chunk, query_entities: list[str], related_entities: list[tuple[str, float]]
    ) -> float:
        """Calculate how well connected a chunk is to the query entities."""
        if not self.knowledge_graph:
            return 0.0

        # Find entities in this chunk
        chunk_entities = []
        for entity_name, entity in self.knowledge_graph.model.entities.items():
            if chunk.id in entity.chunk_ids:
                chunk_entities.append(entity_name)

        if not chunk_entities:
            return 0.0

        # Calculate connectivity to query entities
        connectivity_scores = []

        for chunk_entity in chunk_entities:
            max_connectivity = 0.0

            for query_entity in query_entities:
                if chunk_entity == query_entity:
                    max_connectivity = 1.0
                    break

                # Find connection paths
                paths = self.knowledge_graph.find_connection_paths(chunk_entity, query_entity, max_length=3)

                if paths:
                    # Score based on shortest path length and relationship confidence
                    shortest_path = min(paths, key=lambda p: p["length"])
                    path_score = 1.0 / (shortest_path["length"] + 1)

                    # Boost by relationship confidence
                    avg_confidence = sum(
                        rel["confidence"] for rel in shortest_path["relationships"] if rel and "confidence" in rel
                    ) / max(len(shortest_path["relationships"]), 1)

                    path_score *= avg_confidence
                    max_connectivity = max(max_connectivity, path_score)

            connectivity_scores.append(max_connectivity)

        # Return average connectivity
        return sum(connectivity_scores) / len(connectivity_scores) if connectivity_scores else 0.0

    def _format_graph_context(self, query: str, chunks: list[Chunk]) -> str:
        """Format context with graph structure information."""
        if not chunks:
            return ""

        context_parts = []

        # Add graph overview
        if self.knowledge_graph:
            query_entities = self._identify_query_entities(query)
            if query_entities:
                context_parts.append(f"Key entities related to your query: {', '.join(query_entities)}")

        # Group chunks by document and add graph context
        doc_chunks = defaultdict(list)
        for chunk in chunks:
            doc_chunks[chunk.source_document].append(chunk)

        for doc_id, doc_chunks_list in doc_chunks.items():
            context_parts.append(f"\nFrom document {doc_id}:")

            for i, chunk in enumerate(doc_chunks_list, 1):
                # Get entities in this chunk
                chunk_entities = self._get_chunk_entities(chunk)

                # Format chunk with entity information
                chunk_info = f"Section {i}"
                if chunk_entities:
                    chunk_info += f" (entities: {', '.join(chunk_entities[:3])})"

                semantic_sim = chunk.metadata.get("semantic_similarity", 0)
                graph_rel = chunk.metadata.get("graph_relevance", 0)

                chunk_info += f" [semantic: {semantic_sim:.3f}, graph: {graph_rel:.3f}]"
                chunk_info += f":\n{chunk.content}"

                context_parts.append(chunk_info)

        return "\n\n".join(context_parts)

    def _get_chunk_entities(self, chunk: Chunk) -> list[str]:
        """Get entities present in a chunk."""
        if not self.knowledge_graph:
            return []

        chunk_entities = []
        for entity_name, entity in self.knowledge_graph.model.entities.items():
            if chunk.id in entity.chunk_ids:
                chunk_entities.append(entity_name)

        return chunk_entities

    def _generate_graph_response(self, query: str, context: str, chunks: list[Chunk]) -> str:
        """Generate response using graph-enhanced context."""
        if not chunks:
            return "I don't have enough information to answer your question."

        # Analyze graph connections
        query_entities = self._identify_query_entities(query)
        entity_connections = self._analyze_entity_connections(query_entities, chunks)

        response_parts = [
            f"Based on knowledge graph analysis of {len({chunk.source_document for chunk in chunks})} document(s), "
            f"here's what I found regarding: '{query}'",
            "",
        ]

        # Add entity analysis
        if query_entities:
            response_parts.append(f"Key entities identified: {', '.join(query_entities)}")

        if entity_connections:
            response_parts.append(f"Related concepts found: {', '.join(entity_connections[:5])}")

        response_parts.append(f"\nI found {len(chunks)} relevant sections through graph traversal:")

        # Present information with graph context
        for i, chunk in enumerate(chunks[:3], 1):  # Show top 3
            snippet = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content

            # Get scores
            semantic_sim = chunk.metadata.get("semantic_similarity", 0)
            graph_rel = chunk.metadata.get("graph_relevance", 0)
            connectivity = chunk.metadata.get("connectivity_score", 0)

            # Get entities in chunk
            chunk_entities = self._get_chunk_entities(chunk)
            entity_info = f" (entities: {', '.join(chunk_entities[:2])})" if chunk_entities else ""

            response_parts.append(
                f"{i}. {snippet}{entity_info} "
                f"[similarity: {semantic_sim:.3f}, graph: {graph_rel:.3f}, connectivity: {connectivity:.3f}]"
            )

        if len(chunks) > 3:
            response_parts.append(f"... and {len(chunks) - 3} more connected sources.")

        response_parts.extend(
            [
                "",
                "This Graph-RAG approach identified entities in your query, traversed the knowledge graph "
                "to find related concepts, and retrieved information based on both semantic similarity "
                "and graph connectivity.",
                "",
                "Note: This is a demonstration response. In a production system, "
                "this would be generated by a language model using the graph-enhanced context.",
            ]
        )

        return "\n".join(response_parts)

    def _analyze_entity_connections(self, query_entities: list[str], chunks: list[Chunk]) -> list[str]:
        """Analyze entity connections in retrieved chunks."""
        if not self.knowledge_graph:
            return []

        connected_entities = set()

        for query_entity in query_entities:
            related = self.knowledge_graph.find_related_entities(query_entity, max_depth=1)
            for entity_name, _ in related[:5]:  # Top 5 related
                connected_entities.add(entity_name)

        # Filter to entities that appear in retrieved chunks
        chunk_entities = set()
        for chunk in chunks:
            chunk_entities.update(self._get_chunk_entities(chunk))

        return list(connected_entities.intersection(chunk_entities))

    def _calculate_confidence(self, chunks: list[Chunk]) -> float:
        """Calculate confidence score for graph-based retrieval."""
        if not chunks:
            return 0.0

        # Consider semantic similarity, graph relevance, and connectivity
        semantic_scores = []
        graph_scores = []
        connectivity_scores = []

        for chunk in chunks:
            semantic_scores.append(chunk.metadata.get("semantic_similarity", 0))
            graph_scores.append(chunk.metadata.get("graph_relevance", 0))
            connectivity_scores.append(chunk.metadata.get("connectivity_score", 0))

        if not semantic_scores:
            return 0.5

        # Weighted combination
        avg_semantic = sum(semantic_scores) / len(semantic_scores)
        avg_graph = sum(graph_scores) / len(graph_scores)
        avg_connectivity = sum(connectivity_scores) / len(connectivity_scores)

        # Factor in chunk count and graph coverage
        chunk_factor = min(len(chunks) / self.config.retrieval_top_k, 1.0)

        confidence = (0.3 * avg_semantic + 0.4 * avg_graph + 0.3 * avg_connectivity) * chunk_factor
        return min(confidence, 1.0)

    def _explain_process(self, query: str, chunks: list[Chunk]) -> dict[str, Any]:
        """Provide explanation of the Graph-RAG process."""
        query_entities = self._identify_query_entities(query)

        # Analyze graph traversal
        entity_analysis = {}
        if self.knowledge_graph:
            for entity in query_entities:
                if entity in self.knowledge_graph.model.entities:
                    related = self.knowledge_graph.find_related_entities(entity, max_depth=2)
                    entity_analysis[entity] = {
                        "type": self.knowledge_graph.model.entities[entity].entity_type,
                        "confidence": self.knowledge_graph.model.entities[entity].confidence,
                        "related_count": len(related),
                        "top_related": [name for name, _ in related[:3]],
                    }

        # Analyze chunk scores
        score_analysis = {
            "avg_semantic_similarity": sum(chunk.metadata.get("semantic_similarity", 0) for chunk in chunks)
            / max(len(chunks), 1),
            "avg_graph_relevance": sum(chunk.metadata.get("graph_relevance", 0) for chunk in chunks)
            / max(len(chunks), 1),
            "avg_connectivity": sum(chunk.metadata.get("connectivity_score", 0) for chunk in chunks)
            / max(len(chunks), 1),
        }

        return {
            "rag_type": "graph",
            "query_length": len(query),
            "chunks_retrieved": len(chunks),
            "query_entities": query_entities,
            "entity_analysis": entity_analysis,
            "score_analysis": score_analysis,
            "graph_stats": self.knowledge_graph.get_statistics() if self.knowledge_graph else {},
            "chunking_strategy": self.config.chunking.strategy,
            "embedding_model": self.embedding_system.model_name,
            "embedding_dimension": self.embedding_system.embedding_dimension,
            "process_steps": [
                "1. Extract entities and relationships from documents",
                "2. Build knowledge graph from extracted information",
                "3. Identify entities mentioned in query",
                "4. Traverse graph to find related entities",
                "5. Retrieve chunks associated with related entities",
                "6. Rank chunks using semantic similarity + graph relevance + connectivity",
                "7. Format context with graph structure information",
                "8. Generate response leveraging graph connections",
            ],
        }

    def get_stats(self) -> dict[str, Any]:
        """Get Graph-RAG statistics."""
        avg_retrieval_time = self.stats["total_retrieval_time"] / max(self.stats["queries_processed"], 1)
        avg_generation_time = self.stats["total_generation_time"] / max(self.stats["queries_processed"], 1)

        base_stats = {
            **self.stats,
            "total_chunks": len(self.chunks),
            "total_documents": len(self.documents),
            "avg_retrieval_time": avg_retrieval_time,
            "avg_generation_time": avg_generation_time,
            "embedding_model": self.embedding_system.model_name,
            "chunking_strategy": self.config.chunking.strategy,
        }

        # Add graph statistics
        if self.knowledge_graph:
            graph_stats = self.knowledge_graph.get_statistics()
            base_stats.update(
                {
                    "graph_density": graph_stats.get("density", 0),
                    "graph_clustering": graph_stats.get("clustering_coefficient", 0),
                    "connected_components": graph_stats.get("connected_components", 0),
                    "entity_types": graph_stats.get("entity_types", {}),
                    "relationship_types": graph_stats.get("relationship_types", {}),
                }
            )

        return base_stats

    def get_knowledge_graph(self) -> KnowledgeGraph | None:
        """Get the knowledge graph for external analysis."""
        return self.knowledge_graph

    def clear(self) -> None:
        """Clear all stored documents, chunks, and graph data."""
        self.documents.clear()
        self.chunks.clear()
        self.chunk_index.clear()
        self.extraction_results.clear()

        if self.knowledge_graph:
            self.knowledge_graph = None

        self.graph_builder = KnowledgeGraphBuilder()
        self.embedding_system.clear_cache()

        # Reset stats
        self.stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "entities_extracted": 0,
            "relationships_extracted": 0,
            "graph_nodes": 0,
            "graph_edges": 0,
            "queries_processed": 0,
            "total_retrieval_time": 0.0,
            "total_generation_time": 0.0,
            "avg_graph_traversal_time": 0.0,
        }

        logger.info("Cleared all documents, chunks, and graph data from GraphRAG")
