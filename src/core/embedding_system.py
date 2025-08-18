"""Embedding system with multiple model support and vector operations."""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from utils.logging_utils import get_logger

from core.data_models import Chunk, EmbeddingConfig
from core.exceptions import APIError, EmbeddingError, ModelNotFoundError
from core.interfaces import BaseEmbedder

logger = get_logger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""

    embeddings: np.ndarray
    model_name: str
    dimension: int
    processing_time: float
    token_count: int


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for list of texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model name."""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(self, model_name: str = "text-embedding-ada-002", api_key: str | None = None):
        self._model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise EmbeddingError("OpenAI API key not provided")

        try:
            import openai

            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ModelNotFoundError("OpenAI library not installed. Install with: pip install openai")

        self._dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        if not texts:
            return np.array([])

        try:
            response = self.client.embeddings.create(model=self._model_name, input=texts)

            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)

        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {str(e)}")
            raise APIError(f"OpenAI API error: {str(e)}")

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimensions.get(self._model_name, 1536)

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._model_name


class SentenceBERTProvider(EmbeddingProvider):
    """Sentence-BERT embedding provider."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self._dimension = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ModelNotFoundError(
                "sentence-transformers library not installed. Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load model {model_name}: {str(e)}")

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings using Sentence-BERT."""
        if not texts:
            return np.array([])

        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Sentence-BERT embedding generation failed: {str(e)}")
            raise EmbeddingError(f"Sentence-BERT error: {str(e)}")

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._model_name


class HuggingFaceProvider(EmbeddingProvider):
    """Hugging Face transformers embedding provider."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model_name = model_name

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

            # Get dimension from model config
            self._dimension = self.model.config.hidden_size

        except ImportError:
            raise ModelNotFoundError("transformers library not installed. Install with: pip install transformers torch")
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load model {model_name}: {str(e)}")

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings using Hugging Face transformers."""
        if not texts:
            return np.array([])

        try:
            import torch

            embeddings = []

            for text in texts:
                # Tokenize and encode
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use mean pooling of last hidden states
                    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    embeddings.append(embedding[0])

            return np.array(embeddings)

        except Exception as e:
            logger.error(f"Hugging Face embedding generation failed: {str(e)}")
            raise EmbeddingError(f"Hugging Face error: {str(e)}")

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._model_name


class EmbeddingSystem(BaseEmbedder):
    """Main embedding system supporting multiple providers."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.provider = self._create_provider()
        self._cache: dict[str, np.ndarray] = {}

    def _create_provider(self) -> EmbeddingProvider:
        """Create embedding provider based on configuration."""
        model_name = self.config.model_name.lower()

        if "ada-002" in model_name or "text-embedding" in model_name:
            return OpenAIEmbeddingProvider(self.config.model_name, self.config.api_key)
        elif "sentence-transformers" in model_name or model_name in ["all-minilm-l6-v2", "all-mpnet-base-v2"]:
            return SentenceBERTProvider(self.config.model_name)
        else:
            # Try Hugging Face as fallback
            return HuggingFaceProvider(self.config.model_name)

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for list of texts with batching."""
        if not texts:
            return np.array([])

        start_time = time.time()

        try:
            # Process in batches
            all_embeddings = []
            batch_size = self.config.batch_size

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_embeddings = self.provider.generate_embeddings(batch_texts)
                all_embeddings.append(batch_embeddings)

            # Combine all batches
            embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])

            # Normalize if requested
            if self.config.normalize and embeddings.size > 0:
                embeddings = self._normalize_embeddings(embeddings)

            processing_time = time.time() - start_time
            logger.info(f"Generated {len(embeddings)} embeddings in {processing_time:.2f}s")

            return embeddings

        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")

    def generate_embeddings_with_cache(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings with caching support."""
        if not texts:
            return np.array([])

        # Check cache for existing embeddings
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            text_hash = str(hash(text))
            if text_hash in self._cache:
                cached_embeddings[i] = self._cache[text_hash]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Generate embeddings for uncached texts
        if uncached_texts:
            new_embeddings = self.generate_embeddings(uncached_texts)

            # Cache new embeddings
            for i, embedding in enumerate(new_embeddings):
                text_hash = str(hash(uncached_texts[i]))
                self._cache[text_hash] = embedding
                cached_embeddings[uncached_indices[i]] = embedding

        # Reconstruct full embedding array
        result_embeddings = []
        for i in range(len(texts)):
            result_embeddings.append(cached_embeddings[i])

        return np.array(result_embeddings)

    def compute_similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """Compute similarity scores between query and document embeddings."""
        if query_embedding.size == 0 or doc_embeddings.size == 0:
            return np.array([])

        try:
            # Ensure proper dimensions
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            if doc_embeddings.ndim == 1:
                doc_embeddings = doc_embeddings.reshape(1, -1)

            # Compute cosine similarity
            similarities = self._cosine_similarity(query_embedding, doc_embeddings)
            return similarities.flatten()

        except Exception as e:
            logger.error(f"Similarity computation failed: {str(e)}")
            raise EmbeddingError(f"Failed to compute similarity: {str(e)}")

    def compute_pairwise_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise similarity matrix."""
        if embeddings.size == 0:
            return np.array([])

        try:
            return self._cosine_similarity(embeddings, embeddings)
        except Exception as e:
            logger.error(f"Pairwise similarity computation failed: {str(e)}")
            raise EmbeddingError(f"Failed to compute pairwise similarity: {str(e)}")

    def find_most_similar(
        self, query_embedding: np.ndarray, doc_embeddings: np.ndarray, top_k: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find most similar embeddings and return indices and scores."""
        similarities = self.compute_similarity(query_embedding, doc_embeddings)

        if len(similarities) == 0:
            return np.array([]), np.array([])

        # Get top-k most similar
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]

        return top_indices, top_scores

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Generate embeddings for chunks and update them in-place."""
        if not chunks:
            return chunks

        try:
            # Extract text content
            texts = [chunk.content for chunk in chunks]

            # Generate embeddings
            embeddings = self.generate_embeddings_with_cache(texts)

            # Update chunks with embeddings
            for chunk, embedding in zip(chunks, embeddings, strict=False):
                chunk.embedding = embedding
                chunk.metadata["embedding_model"] = self.provider.model_name
                chunk.metadata["embedding_dimension"] = len(embedding)

            logger.info(f"Generated embeddings for {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Chunk embedding failed: {str(e)}")
            raise EmbeddingError(f"Failed to embed chunks: {str(e)}")

    @property
    def embedding_dimension(self) -> int:
        """Return the dimension of embeddings produced by this embedder."""
        return self.provider.dimension

    @property
    def model_name(self) -> str:
        """Return the name of the embedding model."""
        return self.provider.model_name

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit length."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between two sets of vectors."""
        # Normalize vectors
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)

        # Compute dot product
        return np.dot(a_norm, b_norm.T)

    def _dot_product_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute dot product similarity."""
        return np.dot(a, b.T)

    def _euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance (converted to similarity)."""
        distances = np.linalg.norm(a[:, np.newaxis] - b, axis=2)
        # Convert distance to similarity (higher is more similar)
        return 1 / (1 + distances)

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "memory_usage_mb": sum(embedding.nbytes for embedding in self._cache.values()) / (1024 * 1024),
        }

    def save_embeddings(self, embeddings: np.ndarray, file_path: str) -> None:
        """Save embeddings to file."""
        try:
            np.save(file_path, embeddings)
            logger.info(f"Saved embeddings to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save embeddings: {str(e)}")
            raise EmbeddingError(f"Failed to save embeddings: {str(e)}")

    def load_embeddings(self, file_path: str) -> np.ndarray:
        """Load embeddings from file."""
        try:
            embeddings = np.load(file_path)
            logger.info(f"Loaded embeddings from {file_path}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to load embeddings: {str(e)}")
            raise EmbeddingError(f"Failed to load embeddings: {str(e)}")


class EmbeddingBenchmark:
    """Benchmark different embedding models."""

    def __init__(self):
        self.results: list[dict[str, Any]] = []

    def benchmark_models(self, models: list[str], test_texts: list[str]) -> dict[str, Any]:
        """Benchmark multiple embedding models."""
        results = {}

        for model_name in models:
            try:
                config = EmbeddingConfig(model_name=model_name)
                embedding_system = EmbeddingSystem(config)

                # Measure performance
                start_time = time.time()
                embeddings = embedding_system.generate_embeddings(test_texts)
                processing_time = time.time() - start_time

                results[model_name] = {
                    "dimension": embedding_system.embedding_dimension,
                    "processing_time": processing_time,
                    "texts_per_second": len(test_texts) / processing_time if processing_time > 0 else 0,
                    "embedding_shape": embeddings.shape,
                    "memory_usage_mb": embeddings.nbytes / (1024 * 1024),
                }

                logger.info(f"Benchmarked {model_name}: {processing_time:.2f}s for {len(test_texts)} texts")

            except Exception as e:
                logger.error(f"Failed to benchmark {model_name}: {str(e)}")
                results[model_name] = {"error": str(e)}

        return results
