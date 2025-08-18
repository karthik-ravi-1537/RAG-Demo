"""Custom exceptions for RAG components."""


class RAGException(Exception):
    """Base exception for RAG-related errors."""

    pass


class ProcessingError(RAGException):
    """Exception raised during document processing."""

    pass


class ChunkingError(RAGException):
    """Exception raised during text chunking."""

    pass


class EmbeddingError(RAGException):
    """Exception raised during embedding generation."""

    pass


class RetrievalError(RAGException):
    """Exception raised during information retrieval."""

    pass


class GenerationError(RAGException):
    """Exception raised during response generation."""

    pass


class ConfigurationError(RAGException):
    """Exception raised for configuration-related issues."""

    pass


class ModelNotFoundError(RAGException):
    """Exception raised when a required model is not available."""

    pass


class APIError(RAGException):
    """Exception raised for external API-related errors."""

    pass
