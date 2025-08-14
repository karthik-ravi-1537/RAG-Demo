"""Base interfaces and abstract classes for RAG components."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np

from core.data_models import Chunk, ProcessedDocument, RAGResult


class BaseDocumentProcessor(ABC):
    """Abstract base class for document processing."""
    
    @abstractmethod
    def process_document(self, file_path: str, doc_type: str) -> ProcessedDocument:
        """Process a document and return structured representation."""
        pass
    
    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """Extract text content from document."""
        pass
    
    @abstractmethod
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from document."""
        pass


class BaseChunker(ABC):
    """Abstract base class for text chunking strategies."""
    
    @abstractmethod
    def chunk_text(self, text: str, **kwargs) -> List[Chunk]:
        """Split text into chunks using specific strategy."""
        pass
    
    @abstractmethod
    def chunk_document(self, document: ProcessedDocument, **kwargs) -> List[Chunk]:
        """Split document into chunks preserving structure."""
        pass


class BaseEmbedder(ABC):
    """Abstract base class for embedding generation."""
    
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for list of texts."""
        pass
    
    @abstractmethod
    def compute_similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """Compute similarity scores between query and document embeddings."""
        pass
    
    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the dimension of embeddings produced by this embedder."""
        pass


class BaseRAG(ABC):
    """Abstract base class for RAG implementations."""
    
    def __init__(self, embedder: BaseEmbedder, chunker: BaseChunker):
        self.embedder = embedder
        self.chunker = chunker
        self.chunks: List[Chunk] = []
        self.chunk_embeddings: Optional[np.ndarray] = None
    
    @abstractmethod
    def add_documents(self, documents: List[ProcessedDocument]) -> None:
        """Add documents to the RAG system."""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        """Retrieve relevant chunks for a query."""
        pass
    
    @abstractmethod
    def generate_response(self, query: str, context_chunks: List[Chunk]) -> str:
        """Generate response using retrieved context."""
        pass
    
    def query(self, query: str, top_k: int = 5) -> RAGResult:
        """Complete RAG pipeline: retrieve and generate."""
        import time
        
        # Retrieve relevant chunks
        start_time = time.time()
        retrieved_chunks = self.retrieve(query, top_k)
        retrieval_time = time.time() - start_time
        
        # Generate response
        start_time = time.time()
        response = self.generate_response(query, retrieved_chunks)
        generation_time = time.time() - start_time
        
        return RAGResult(
            query=query,
            retrieved_chunks=retrieved_chunks,
            response=response,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            confidence_score=self._calculate_confidence(retrieved_chunks),
            explanation=self._explain_process(query, retrieved_chunks)
        )
    
    @abstractmethod
    def _calculate_confidence(self, chunks: List[Chunk]) -> float:
        """Calculate confidence score for retrieved chunks."""
        pass
    
    @abstractmethod
    def _explain_process(self, query: str, chunks: List[Chunk]) -> Dict[str, Any]:
        """Provide explanation of the RAG process."""
        pass


class BaseContextEngineer(ABC):
    """Abstract base class for context engineering."""
    
    @abstractmethod
    def format_context(self, chunks: List[Chunk], template: str) -> str:
        """Format chunks into context using template."""
        pass
    
    @abstractmethod
    def rank_by_relevance(self, chunks: List[Chunk], query: str) -> List[Chunk]:
        """Rank chunks by relevance to query."""
        pass
    
    @abstractmethod
    def compress_context(self, context: str, max_tokens: int) -> str:
        """Compress context to fit within token limits."""
        pass


class BaseEvaluator(ABC):
    """Abstract base class for RAG evaluation."""
    
    @abstractmethod
    def evaluate_retrieval(self, queries: List[str], ground_truth: List[List[str]]) -> Dict[str, float]:
        """Evaluate retrieval quality."""
        pass
    
    @abstractmethod
    def evaluate_generation(self, responses: List[str], references: List[str]) -> Dict[str, float]:
        """Evaluate generation quality."""
        pass
    
    @abstractmethod
    def compare_rag_systems(self, rag_systems: List[BaseRAG], queries: List[str]) -> Dict[str, Any]:
        """Compare multiple RAG systems."""
        pass