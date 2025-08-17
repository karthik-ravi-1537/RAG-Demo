"""Hierarchical RAG implementation with multi-level document representation."""

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from core.chunking_engine import ChunkingEngine
from core.data_models import Chunk, ChunkingConfig, ProcessedDocument, RAGConfig
from core.document_processor import DocumentProcessor
from core.embedding_system import EmbeddingSystem
from core.exceptions import GenerationError, RAGException, RetrievalError
from core.interfaces import BaseRAG
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class HierarchicalNode:
    """Represents a node in the document hierarchy."""

    id: str
    content: str
    level: int  # 0 = document, 1 = section, 2 = chunk
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    embedding: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    summary: str | None = None
    chunk: Chunk | None = None


@dataclass
class HierarchicalStructure:
    """Represents the hierarchical structure of a document."""

    document_id: str
    nodes: dict[str, HierarchicalNode] = field(default_factory=dict)
    levels: dict[int, list[str]] = field(default_factory=dict)  # level -> node_ids

    def add_node(self, node: HierarchicalNode) -> None:
        """Add a node to the hierarchy."""
        self.nodes[node.id] = node
        if node.level not in self.levels:
            self.levels[node.level] = []
        self.levels[node.level].append(node.id)

    def get_children(self, node_id: str) -> list[HierarchicalNode]:
        """Get children of a node."""
        if node_id not in self.nodes:
            return []
        node = self.nodes[node_id]
        return [self.nodes[child_id] for child_id in node.children_ids if child_id in self.nodes]

    def get_parent(self, node_id: str) -> HierarchicalNode | None:
        """Get parent of a node."""
        if node_id not in self.nodes:
            return None
        node = self.nodes[node_id]
        if node.parent_id and node.parent_id in self.nodes:
            return self.nodes[node.parent_id]
        return None


class HierarchicalRAG(BaseRAG):
    """
    Hierarchical RAG implementation with multi-level document representation.

    This approach:
    1. Creates hierarchical document structure (document -> sections -> chunks)
    2. Generates summaries at different levels
    3. Performs two-stage retrieval (summary -> detailed chunks)
    4. Assembles context from multiple hierarchy levels
    """

    def __init__(self, config: RAGConfig = None):
        """Initialize Hierarchical RAG system."""
        self.config = config or RAGConfig()

        # Initialize core components
        self.document_processor = DocumentProcessor()
        self.chunking_engine = ChunkingEngine()
        self.embedding_system = EmbeddingSystem(self.config.embedding)

        # Storage for hierarchical data
        self.documents: list[ProcessedDocument] = []
        self.hierarchies: dict[str, HierarchicalStructure] = {}
        self.all_nodes: dict[str, HierarchicalNode] = {}

        # Configuration for hierarchy
        self.max_levels = 3  # document, section, chunk
        self.summary_levels = [0, 1]  # levels that get summaries

        # Performance tracking
        self.stats = {
            "documents_processed": 0,
            "hierarchies_created": 0,
            "nodes_created": 0,
            "summaries_generated": 0,
            "queries_processed": 0,
            "total_retrieval_time": 0.0,
            "total_generation_time": 0.0,
        }

        logger.info(f"Initialized HierarchicalRAG with {self.embedding_system.model_name} embeddings")

    def add_documents(self, document_paths: list[str]) -> None:
        """Add documents to the hierarchical RAG system."""
        logger.info(f"Adding {len(document_paths)} documents to HierarchicalRAG")

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

            # Build hierarchical structures
            for doc in new_documents:
                try:
                    hierarchy = self._build_hierarchy(doc)
                    self.hierarchies[doc.id] = hierarchy

                    # Add all nodes to global index
                    for node in hierarchy.nodes.values():
                        self.all_nodes[node.id] = node

                    logger.debug(f"Built hierarchy for document {doc.id} with {len(hierarchy.nodes)} nodes")
                except Exception as e:
                    logger.error(f"Failed to build hierarchy for document {doc.id}: {str(e)}")
                    continue

            # Generate embeddings for all nodes
            self._generate_embeddings()

            # Generate summaries
            self._generate_summaries()

            # Update storage
            self.documents.extend(new_documents)

            # Update stats
            self.stats["documents_processed"] += len(new_documents)
            self.stats["hierarchies_created"] += len(list(self.hierarchies.values()))
            self.stats["nodes_created"] += len(self.all_nodes)

            logger.info(f"Successfully added {len(new_documents)} documents with hierarchical structure")

        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise RAGException(f"Document addition failed: {str(e)}")

    def retrieve(self, query: str, top_k: int = None) -> list[Chunk]:
        """Retrieve relevant chunks using hierarchical approach."""
        if top_k is None:
            top_k = self.config.retrieval_top_k

        if not self.all_nodes:
            raise RetrievalError("No documents have been added to the system")

        try:
            start_time = time.time()

            # Stage 1: Retrieve from high-level summaries
            summary_nodes = self._retrieve_summary_nodes(query, top_k * 2)

            # Stage 2: Retrieve detailed chunks from relevant hierarchies
            detailed_chunks = self._retrieve_detailed_chunks(query, summary_nodes, top_k)

            retrieval_time = time.time() - start_time
            self.stats["total_retrieval_time"] += retrieval_time

            logger.info(
                f"Hierarchical retrieval: {len(summary_nodes)} summaries -> {len(detailed_chunks)} chunks in {retrieval_time:.3f}s"
            )

            return detailed_chunks

        except Exception as e:
            logger.error(f"Hierarchical retrieval failed: {str(e)}")
            raise RetrievalError(f"Failed to retrieve chunks: {str(e)}")

    def generate_response(self, query: str, context_chunks: list[Chunk]) -> str:
        """Generate response using hierarchical context."""
        try:
            start_time = time.time()

            # Format hierarchical context
            context = self._format_hierarchical_context(context_chunks)

            # Generate response (mock implementation)
            response = self._generate_hierarchical_response(query, context, context_chunks)

            generation_time = time.time() - start_time
            self.stats["total_generation_time"] += generation_time

            logger.info(f"Generated hierarchical response in {generation_time:.3f}s")

            return response

        except Exception as e:
            logger.error(f"Hierarchical response generation failed: {str(e)}")
            raise GenerationError(f"Failed to generate response: {str(e)}")

    def _build_hierarchy(self, document: ProcessedDocument) -> HierarchicalStructure:
        """Build hierarchical structure for a document."""
        hierarchy = HierarchicalStructure(document_id=document.id)

        # Level 0: Document node
        doc_node = HierarchicalNode(
            id=f"{document.id}_doc",
            content=document.content,
            level=0,
            metadata={
                "document_id": document.id,
                "document_type": document.document_type,
                "file_path": document.file_path,
            },
        )
        hierarchy.add_node(doc_node)

        # Level 1: Section nodes (based on structure-aware chunking)
        section_chunks = self.chunking_engine.chunk_document(
            document, ChunkingConfig(strategy="structure_aware", chunk_size=800, overlap=100, min_chunk_size=200)
        )

        section_nodes = []
        for i, chunk in enumerate(section_chunks):
            section_node = HierarchicalNode(
                id=f"{document.id}_sec_{i}",
                content=chunk.content,
                level=1,
                parent_id=doc_node.id,
                metadata=chunk.metadata.copy(),
                chunk=chunk,
            )
            hierarchy.add_node(section_node)
            section_nodes.append(section_node)
            doc_node.children_ids.append(section_node.id)

        # Level 2: Detailed chunks
        detailed_chunks = self.chunking_engine.chunk_document(document, self.config.chunking)

        # Assign detailed chunks to sections based on content overlap
        for chunk in detailed_chunks:
            best_section = self._find_best_section(chunk, section_nodes)
            if best_section:
                chunk_node = HierarchicalNode(
                    id=f"{document.id}_chunk_{chunk.chunk_index}",
                    content=chunk.content,
                    level=2,
                    parent_id=best_section.id,
                    metadata=chunk.metadata.copy(),
                    chunk=chunk,
                )
                hierarchy.add_node(chunk_node)
                best_section.children_ids.append(chunk_node.id)

        return hierarchy

    def _find_best_section(self, chunk: Chunk, section_nodes: list[HierarchicalNode]) -> HierarchicalNode | None:
        """Find the best section for a chunk based on content overlap."""
        if not section_nodes:
            return None

        best_section = None
        best_overlap = 0

        chunk_words = set(chunk.content.lower().split())

        for section in section_nodes:
            section_words = set(section.content.lower().split())
            overlap = len(chunk_words.intersection(section_words))

            if overlap > best_overlap:
                best_overlap = overlap
                best_section = section

        return best_section if best_overlap > 0 else section_nodes[0]

    def _generate_embeddings(self) -> None:
        """Generate embeddings for all nodes."""
        if not self.all_nodes:
            return

        # Collect all content for batch embedding
        node_ids = []
        contents = []

        for node_id, node in self.all_nodes.items():
            node_ids.append(node_id)
            # Use summary if available, otherwise use content
            content = node.summary if node.summary else node.content
            contents.append(content)

        # Generate embeddings in batch
        embeddings = self.embedding_system.generate_embeddings(contents)

        # Assign embeddings to nodes
        for node_id, embedding in zip(node_ids, embeddings, strict=False):
            self.all_nodes[node_id].embedding = embedding

        logger.info(f"Generated embeddings for {len(node_ids)} hierarchical nodes")

    def _generate_summaries(self) -> None:
        """Generate summaries for nodes at specified levels."""
        summaries_generated = 0

        for hierarchy in self.hierarchies.values():
            for level in self.summary_levels:
                if level in hierarchy.levels:
                    for node_id in hierarchy.levels[level]:
                        node = hierarchy.nodes[node_id]
                        if not node.summary:
                            node.summary = self._create_summary(node)
                            summaries_generated += 1

        self.stats["summaries_generated"] = summaries_generated
        logger.info(f"Generated {summaries_generated} summaries")

    def _create_summary(self, node: HierarchicalNode) -> str:
        """Create a summary for a node."""
        content = node.content

        # Simple extractive summarization (first few sentences)
        sentences = content.split(". ")

        if len(sentences) <= 3:
            return content

        # Take first 2 sentences and add key information
        summary_parts = sentences[:2]

        # Add any headers or important markers
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("#") or line.startswith("**") or line.isupper():
                if line not in " ".join(summary_parts):
                    summary_parts.append(line)
                    break

        summary = ". ".join(summary_parts)
        if not summary.endswith("."):
            summary += "."

        return summary

    def _retrieve_summary_nodes(self, query: str, top_k: int) -> list[HierarchicalNode]:
        """Retrieve relevant summary nodes (Stage 1)."""
        # Get all summary nodes
        summary_nodes = [
            node for node in self.all_nodes.values() if node.level in self.summary_levels and node.embedding is not None
        ]

        if not summary_nodes:
            return []

        # Generate query embedding
        query_embedding = self.embedding_system.generate_embeddings([query])[0]

        # Compute similarities
        embeddings = np.vstack([node.embedding for node in summary_nodes])
        similarities = self.embedding_system.compute_similarity(query_embedding.reshape(1, -1), embeddings)

        # Get top-k most similar
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Filter by threshold
        relevant_nodes = []
        for idx in top_indices:
            if similarities[idx] >= self.config.similarity_threshold:
                node = summary_nodes[idx]
                # Store similarity score in metadata
                node.metadata["similarity_score"] = float(similarities[idx])
                relevant_nodes.append(node)

        return relevant_nodes

    def _retrieve_detailed_chunks(self, query: str, summary_nodes: list[HierarchicalNode], top_k: int) -> list[Chunk]:
        """Retrieve detailed chunks from relevant hierarchies (Stage 2)."""
        if not summary_nodes:
            return []

        # Get all detailed chunks from relevant hierarchies
        candidate_chunks = []

        for summary_node in summary_nodes:
            # Get hierarchy for this summary
            hierarchy = None
            for h in self.hierarchies.values():
                if summary_node.id in h.nodes:
                    hierarchy = h
                    break

            if not hierarchy:
                continue

            # Get all leaf nodes (chunks) from this hierarchy
            for node in hierarchy.nodes.values():
                if node.level == 2 and node.chunk and node.embedding is not None:  # Detailed chunks
                    # Weight by parent summary relevance
                    parent_similarity = summary_node.metadata.get("similarity_score", 0.5)
                    node.metadata["parent_similarity"] = parent_similarity
                    candidate_chunks.append(node)

        if not candidate_chunks:
            return []

        # Generate query embedding
        query_embedding = self.embedding_system.generate_embeddings([query])[0]

        # Compute similarities for detailed chunks
        embeddings = np.vstack([node.embedding for node in candidate_chunks])
        similarities = self.embedding_system.compute_similarity(query_embedding.reshape(1, -1), embeddings)

        # Combine with parent similarities (weighted)
        combined_scores = []
        for i, node in enumerate(candidate_chunks):
            chunk_similarity = similarities[i]
            parent_similarity = node.metadata.get("parent_similarity", 0.5)
            # Weighted combination: 70% chunk similarity, 30% parent similarity
            combined_score = 0.7 * chunk_similarity + 0.3 * parent_similarity
            combined_scores.append(combined_score)

        # Get top-k chunks
        top_indices = np.argsort(combined_scores)[::-1][:top_k]

        result_chunks = []
        for idx in top_indices:
            if combined_scores[idx] >= self.config.similarity_threshold * 0.8:  # Slightly lower threshold
                node = candidate_chunks[idx]
                chunk = node.chunk
                if chunk:
                    chunk.similarity_score = float(combined_scores[idx])
                    chunk.metadata["hierarchical_score"] = float(similarities[idx])
                    chunk.metadata["parent_similarity"] = node.metadata.get("parent_similarity", 0.0)
                    result_chunks.append(chunk)

        return result_chunks

    def _format_hierarchical_context(self, chunks: list[Chunk]) -> str:
        """Format chunks with hierarchical context information."""
        if not chunks:
            return ""

        context_parts = []

        # Group chunks by document/hierarchy
        doc_chunks = {}
        for chunk in chunks:
            doc_id = chunk.source_document
            if doc_id not in doc_chunks:
                doc_chunks[doc_id] = []
            doc_chunks[doc_id].append(chunk)

        for doc_id, doc_chunks_list in doc_chunks.items():
            # Add document context
            hierarchy = self.hierarchies.get(doc_id)
            if hierarchy and 0 in hierarchy.levels:
                doc_node = hierarchy.nodes[hierarchy.levels[0][0]]
                if doc_node.summary:
                    context_parts.append(f"Document Summary: {doc_node.summary}")

            # Add chunks with hierarchical information
            for i, chunk in enumerate(doc_chunks_list, 1):
                # Find parent section
                parent_info = self._get_parent_context(chunk, hierarchy)

                context_part = f"Section {i}"
                if parent_info:
                    context_part += f" (from {parent_info})"
                context_part += f":\n{chunk.content}"

                context_parts.append(context_part)

        return "\n\n".join(context_parts)

    def _get_parent_context(self, chunk: Chunk, hierarchy: HierarchicalStructure | None) -> str | None:
        """Get parent context information for a chunk."""
        if not hierarchy:
            return None

        # Find the chunk node in hierarchy
        chunk_node = None
        for node in hierarchy.nodes.values():
            if node.chunk and node.chunk.id == chunk.id:
                chunk_node = node
                break

        if not chunk_node:
            return None

        # Get parent section
        parent = hierarchy.get_parent(chunk_node.id)
        if parent and parent.summary:
            return f"Section: {parent.summary[:100]}..."

        return None

    def _generate_hierarchical_response(self, query: str, context: str, chunks: list[Chunk]) -> str:
        """Generate response using hierarchical context."""
        if not chunks:
            return "I don't have enough information to answer your question."

        # Create hierarchical response
        response_parts = [
            f"Based on the hierarchical analysis of {len({chunk.source_document for chunk in chunks})} document(s), here's what I found regarding: '{query}'",
            "",
            f"I analyzed information at multiple levels and found {len(chunks)} relevant sections:",
        ]

        # Group and present information hierarchically
        doc_groups = {}
        for chunk in chunks:
            doc_id = chunk.source_document
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(chunk)

        for doc_id, doc_chunks in doc_groups.items():
            # Sort by hierarchical and similarity scores
            doc_chunks.sort(key=lambda c: c.similarity_score, reverse=True)

            response_parts.append(f"\nFrom document {doc_id}:")

            for i, chunk in enumerate(doc_chunks[:2], 1):  # Show top 2 per document
                snippet = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
                hierarchical_score = chunk.metadata.get("hierarchical_score", 0)
                parent_score = chunk.metadata.get("parent_similarity", 0)

                response_parts.append(
                    f"{i}. {snippet} "
                    f"(relevance: {chunk.similarity_score:.3f}, "
                    f"section: {hierarchical_score:.3f}, "
                    f"context: {parent_score:.3f})"
                )

        response_parts.extend(
            [
                "",
                "This hierarchical approach first identified relevant document sections, "
                "then retrieved specific details, providing more contextually accurate information.",
                "",
                "Note: This is a demonstration response. In a production system, "
                "this would be generated by a language model using the hierarchical context.",
            ]
        )

        return "\n".join(response_parts)

    def _calculate_confidence(self, chunks: list[Chunk]) -> float:
        """Calculate confidence score for hierarchical retrieval."""
        if not chunks:
            return 0.0

        # Consider both chunk similarity and hierarchical context
        similarities = []
        hierarchical_scores = []

        for chunk in chunks:
            similarities.append(chunk.similarity_score)
            hierarchical_scores.append(chunk.metadata.get("hierarchical_score", chunk.similarity_score))

        if not similarities:
            return 0.5

        # Weighted combination of direct similarity and hierarchical relevance
        avg_similarity = sum(similarities) / len(similarities)
        avg_hierarchical = sum(hierarchical_scores) / len(hierarchical_scores)

        # Factor in number of chunks and hierarchy coverage
        chunk_factor = min(len(chunks) / self.config.retrieval_top_k, 1.0)
        hierarchy_factor = len({chunk.source_document for chunk in chunks}) / max(len(self.hierarchies), 1)

        confidence = (0.4 * avg_similarity + 0.4 * avg_hierarchical + 0.2 * hierarchy_factor) * chunk_factor
        return min(confidence, 1.0)

    def _explain_process(self, query: str, chunks: list[Chunk]) -> dict[str, Any]:
        """Provide explanation of the hierarchical RAG process."""
        # Analyze hierarchy coverage
        doc_coverage = {}
        for chunk in chunks:
            doc_id = chunk.source_document
            if doc_id not in doc_coverage:
                doc_coverage[doc_id] = {"chunks": 0, "avg_similarity": 0, "avg_hierarchical": 0}
            doc_coverage[doc_id]["chunks"] += 1
            doc_coverage[doc_id]["avg_similarity"] += chunk.similarity_score
            doc_coverage[doc_id]["avg_hierarchical"] += chunk.metadata.get("hierarchical_score", 0)

        # Calculate averages
        for doc_id in doc_coverage:
            count = doc_coverage[doc_id]["chunks"]
            doc_coverage[doc_id]["avg_similarity"] /= count
            doc_coverage[doc_id]["avg_hierarchical"] /= count

        return {
            "rag_type": "hierarchical",
            "query_length": len(query),
            "chunks_retrieved": len(chunks),
            "documents_covered": len(doc_coverage),
            "hierarchy_levels": self.max_levels,
            "summary_levels": self.summary_levels,
            "chunking_strategy": self.config.chunking.strategy,
            "embedding_model": self.embedding_system.model_name,
            "embedding_dimension": self.embedding_system.embedding_dimension,
            "document_coverage": doc_coverage,
            "process_steps": [
                "1. Build hierarchical document structure (document -> sections -> chunks)",
                "2. Generate summaries at multiple levels",
                "3. Stage 1: Retrieve relevant summary nodes",
                "4. Stage 2: Retrieve detailed chunks from relevant hierarchies",
                "5. Combine hierarchical and similarity scores",
                "6. Format context with hierarchical information",
                "7. Generate response using multi-level context",
            ],
        }

    def get_stats(self) -> dict[str, Any]:
        """Get hierarchical RAG statistics."""
        avg_retrieval_time = self.stats["total_retrieval_time"] / max(self.stats["queries_processed"], 1)
        avg_generation_time = self.stats["total_generation_time"] / max(self.stats["queries_processed"], 1)

        return {
            **self.stats,
            "total_nodes": len(self.all_nodes),
            "total_hierarchies": len(self.hierarchies),
            "avg_retrieval_time": avg_retrieval_time,
            "avg_generation_time": avg_generation_time,
            "embedding_model": self.embedding_system.model_name,
            "chunking_strategy": self.config.chunking.strategy,
            "hierarchy_levels": self.max_levels,
        }

    def clear(self) -> None:
        """Clear all stored documents and hierarchies."""
        self.documents.clear()
        self.hierarchies.clear()
        self.all_nodes.clear()
        self.embedding_system.clear_cache()

        # Reset stats
        self.stats = {
            "documents_processed": 0,
            "hierarchies_created": 0,
            "nodes_created": 0,
            "summaries_generated": 0,
            "queries_processed": 0,
            "total_retrieval_time": 0.0,
            "total_generation_time": 0.0,
        }

        logger.info("Cleared all documents and hierarchies from HierarchicalRAG")
