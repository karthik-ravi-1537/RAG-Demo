"""Data models and structures for RAG components."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class Chunk:
    """Represents a chunk of text with metadata."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None
    source_document: str = ""
    chunk_index: int = 0
    parent_chunk: str | None = None
    similarity_score: float = 0.0

    def __post_init__(self):
        """Ensure metadata has required fields."""
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = datetime.now().isoformat()
        if "chunk_type" not in self.metadata:
            self.metadata["chunk_type"] = "text"


@dataclass
class ProcessedDocument:
    """Represents a processed document with content and metadata."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    file_path: str = ""
    document_type: str = "text"
    structure: dict[str, Any] | None = None

    def __post_init__(self):
        """Ensure metadata has required fields."""
        if "processed_at" not in self.metadata:
            self.metadata["processed_at"] = datetime.now().isoformat()
        if "word_count" not in self.metadata:
            self.metadata["word_count"] = len(self.content.split())


@dataclass
class Entity:
    """Represents an extracted entity."""

    name: str
    entity_type: str
    mentions: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class Relationship:
    """Represents a relationship between entities."""

    source_entity: str
    target_entity: str
    relation_type: str
    confidence: float = 1.0
    evidence_chunks: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeGraph:
    """Represents a knowledge graph structure."""

    entities: dict[str, Entity] = field(default_factory=dict)
    relationships: list[Relationship] = field(default_factory=list)
    adjacency_matrix: np.ndarray | None = None

    def add_entity(self, entity: Entity) -> None:
        """Add entity to the graph."""
        self.entities[entity.name] = entity

    def add_relationship(self, relationship: Relationship) -> None:
        """Add relationship to the graph."""
        self.relationships.append(relationship)

    def get_connected_entities(self, entity_name: str, max_depth: int = 2) -> list[str]:
        """Get entities connected to the given entity within max_depth."""
        connected = set()
        current_level = {entity_name}

        for _ in range(max_depth):
            next_level = set()
            for entity in current_level:
                for rel in self.relationships:
                    if rel.source_entity == entity and rel.target_entity not in connected:
                        next_level.add(rel.target_entity)
                    elif rel.target_entity == entity and rel.source_entity not in connected:
                        next_level.add(rel.source_entity)
            connected.update(next_level)
            current_level = next_level
            if not current_level:
                break

        return list(connected)


@dataclass
class RAGResult:
    """Represents the result of a RAG query."""

    query: str
    retrieved_chunks: list[Chunk]
    response: str
    retrieval_time: float
    generation_time: float
    confidence_score: float
    explanation: dict[str, Any] = field(default_factory=dict)

    @property
    def total_time(self) -> float:
        """Total time for retrieval and generation."""
        return self.retrieval_time + self.generation_time

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "query": self.query,
            "response": self.response,
            "retrieval_time": self.retrieval_time,
            "generation_time": self.generation_time,
            "total_time": self.total_time,
            "confidence_score": self.confidence_score,
            "num_chunks_retrieved": len(self.retrieved_chunks),
            "explanation": self.explanation,
        }


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies."""

    strategy: str = "fixed_size"
    chunk_size: int = 512
    overlap: int = 50
    separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " "])
    preserve_structure: bool = True
    min_chunk_size: int = 100


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 100
    normalize: bool = True
    api_key: str | None = None

    def __post_init__(self):
        """Auto-detect dimension for known OpenAI models."""
        openai_model_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }

        if self.model_name in openai_model_dimensions:
            self.dimension = openai_model_dimensions[self.model_name]


@dataclass
class RAGConfig:
    """Configuration for RAG systems."""

    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval_top_k: int = 5
    similarity_threshold: float = 0.7
    context_template: str = "Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    max_context_tokens: int = 4000

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "RAGConfig":
        """Create RAGConfig from dictionary."""
        chunking_config = ChunkingConfig(**config_dict.get("chunking", {}))
        embedding_config = EmbeddingConfig(**config_dict.get("embedding", {}))

        return cls(
            chunking=chunking_config,
            embedding=embedding_config,
            retrieval_top_k=config_dict.get("retrieval_top_k", 5),
            similarity_threshold=config_dict.get("similarity_threshold", 0.7),
            context_template=config_dict.get("context_template", cls().context_template),
            max_context_tokens=config_dict.get("max_context_tokens", 4000),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert RAGConfig to dictionary."""
        return {
            "chunking": {
                "strategy": self.chunking.strategy,
                "chunk_size": self.chunking.chunk_size,
                "overlap": self.chunking.overlap,
                "separators": self.chunking.separators,
                "preserve_structure": self.chunking.preserve_structure,
                "min_chunk_size": self.chunking.min_chunk_size,
            },
            "embedding": {
                "model_name": self.embedding.model_name,
                "dimension": self.embedding.dimension,
                "batch_size": self.embedding.batch_size,
                "normalize": self.embedding.normalize,
            },
            "retrieval_top_k": self.retrieval_top_k,
            "similarity_threshold": self.similarity_threshold,
            "context_template": self.context_template,
            "max_context_tokens": self.max_context_tokens,
        }


@dataclass
class MultiModalChunk(Chunk):
    """Extended chunk for multi-modal content."""

    modality: str = "text"  # text, image, table, etc.
    image_path: str | None = None
    image_embedding: np.ndarray | None = None
    structured_data: dict[str, Any] | None = None


@dataclass
class MultiModalResponse:
    """Response that can include multiple modalities."""

    text_response: str
    image_references: list[str] = field(default_factory=list)
    structured_data: dict[str, Any] | None = None
    confidence_by_modality: dict[str, float] = field(default_factory=dict)
