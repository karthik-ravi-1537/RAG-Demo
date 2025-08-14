"""Vanilla RAG implementation - the foundational RAG approach."""

import time
from typing import List, Dict, Any, Optional
import numpy as np

from core.interfaces import BaseRAG
from core.data_models import (
    ProcessedDocument, Chunk, RAGResult, RAGConfig, 
    ChunkingConfig, EmbeddingConfig
)
from core.document_processor import DocumentProcessor
from core.chunking_engine import ChunkingEngine
from core.embedding_system import EmbeddingSystem
from core.exceptions import RAGException, RetrievalError, GenerationError
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class VanillaRAG(BaseRAG):
    """
    Vanilla RAG implementation using traditional similarity-based retrieval.
    
    This is the foundational RAG approach that:
    1. Processes documents into chunks
    2. Generates embeddings for all chunks
    3. Retrieves most similar chunks for a query
    4. Formats context and generates response
    """
    
    def __init__(self, config: RAGConfig = None):
        """Initialize Vanilla RAG system."""
        self.config = config or RAGConfig()
        
        # Initialize core components
        self.document_processor = DocumentProcessor()
        self.chunking_engine = ChunkingEngine()
        self.embedding_system = EmbeddingSystem(self.config.embedding)
        
        # Storage for processed data
        self.documents: List[ProcessedDocument] = []
        self.chunks: List[Chunk] = []
        self.chunk_embeddings: Optional[np.ndarray] = None
        
        # Performance tracking
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'queries_processed': 0,
            'total_retrieval_time': 0.0,
            'total_generation_time': 0.0
        }
        
        logger.info(f"Initialized VanillaRAG with {self.embedding_system.model_name} embeddings")
    
    def add_documents(self, document_paths: List[str]) -> None:
        """Add documents to the RAG system."""
        logger.info(f"Adding {len(document_paths)} documents to VanillaRAG")
        
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
            
            # Generate embeddings for new chunks
            embedded_chunks = self.embedding_system.embed_chunks(new_chunks)
            
            # Update storage
            self.documents.extend(new_documents)
            self.chunks.extend(embedded_chunks)
            
            # Rebuild embedding matrix
            self._rebuild_embedding_matrix()
            
            # Update stats
            self.stats['documents_processed'] += len(new_documents)
            self.stats['chunks_created'] += len(embedded_chunks)
            
            logger.info(f"Successfully added {len(new_documents)} documents, "
                       f"created {len(embedded_chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise RAGException(f"Document addition failed: {str(e)}")
    
    def retrieve(self, query: str, top_k: int = None) -> List[Chunk]:
        """Retrieve relevant chunks for a query using similarity search."""
        if top_k is None:
            top_k = self.config.retrieval_top_k
        
        if not self.chunks:
            raise RetrievalError("No documents have been added to the system")
        
        if self.chunk_embeddings is None:
            raise RetrievalError("Chunk embeddings not available")
        
        try:
            start_time = time.time()
            
            # Generate query embedding
            query_embedding = self.embedding_system.generate_embeddings([query])
            if query_embedding.size == 0:
                raise RetrievalError("Failed to generate query embedding")
            
            # Compute similarities
            similarities = self.embedding_system.compute_similarity(
                query_embedding, self.chunk_embeddings
            )
            
            # Get top-k most similar chunks
            top_indices, top_scores = self.embedding_system.find_most_similar(
                query_embedding, self.chunk_embeddings, top_k
            )
            
            # Filter by similarity threshold if configured
            if self.config.similarity_threshold > 0:
                valid_indices = top_scores >= self.config.similarity_threshold
                top_indices = top_indices[valid_indices]
                top_scores = top_scores[valid_indices]
            
            # Create result chunks with similarity scores
            retrieved_chunks = []
            for idx, score in zip(top_indices, top_scores):
                chunk = self.chunks[idx]
                # Create a copy with similarity score
                retrieved_chunk = Chunk(
                    id=chunk.id,
                    content=chunk.content,
                    metadata=chunk.metadata.copy(),
                    embedding=chunk.embedding,
                    source_document=chunk.source_document,
                    chunk_index=chunk.chunk_index,
                    parent_chunk=chunk.parent_chunk,
                    similarity_score=float(score)
                )
                retrieved_chunks.append(retrieved_chunk)
            
            retrieval_time = time.time() - start_time
            self.stats['total_retrieval_time'] += retrieval_time
            
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks in {retrieval_time:.3f}s "
                       f"(similarity threshold: {self.config.similarity_threshold})")
            
            return retrieved_chunks
            
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            raise RetrievalError(f"Failed to retrieve chunks: {str(e)}")
    
    def generate_response(self, query: str, context_chunks: List[Chunk]) -> str:
        """Generate response using retrieved context (mock implementation)."""
        try:
            start_time = time.time()
            
            # Format context from chunks
            context = self._format_context(context_chunks)
            
            # For now, return a formatted response without actual LLM call
            # This can be extended to use OpenAI API or other LLMs
            response = self._generate_mock_response(query, context, context_chunks)
            
            generation_time = time.time() - start_time
            self.stats['total_generation_time'] += generation_time
            
            logger.info(f"Generated response in {generation_time:.3f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            raise GenerationError(f"Failed to generate response: {str(e)}")
    
    def query(self, query: str, top_k: int = None) -> RAGResult:
        """Complete RAG pipeline: retrieve and generate."""
        logger.info(f"Processing query: {query[:100]}...")
        
        try:
            # Retrieve relevant chunks
            start_time = time.time()
            retrieved_chunks = self.retrieve(query, top_k)
            retrieval_time = time.time() - start_time
            
            # Generate response
            start_time = time.time()
            response = self.generate_response(query, retrieved_chunks)
            generation_time = time.time() - start_time
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(retrieved_chunks)
            
            # Create explanation
            explanation = self._explain_process(query, retrieved_chunks)
            
            # Update stats
            self.stats['queries_processed'] += 1
            
            result = RAGResult(
                query=query,
                retrieved_chunks=retrieved_chunks,
                response=response,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                confidence_score=confidence_score,
                explanation=explanation
            )
            
            logger.info(f"Query processed successfully in {result.total_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            raise RAGException(f"Query failed: {str(e)}")
    
    def _rebuild_embedding_matrix(self) -> None:
        """Rebuild the embedding matrix from all chunks."""
        if not self.chunks:
            self.chunk_embeddings = None
            return
        
        embeddings = []
        for chunk in self.chunks:
            if chunk.embedding is not None:
                embeddings.append(chunk.embedding)
            else:
                logger.warning(f"Chunk {chunk.id} has no embedding")
        
        if embeddings:
            self.chunk_embeddings = np.vstack(embeddings)
            logger.debug(f"Rebuilt embedding matrix: {self.chunk_embeddings.shape}")
        else:
            self.chunk_embeddings = None
            logger.warning("No valid embeddings found for chunks")
    
    def _format_context(self, chunks: List[Chunk]) -> str:
        """Format retrieved chunks into context string."""
        if not chunks:
            return ""
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            # Add chunk with source information
            source_info = f"[Source: {chunk.source_document}]" if chunk.source_document else ""
            context_parts.append(f"Context {i} {source_info}:\n{chunk.content}")
        
        return "\n\n".join(context_parts)
    
    def _generate_mock_response(self, query: str, context: str, chunks: List[Chunk]) -> str:
        """Generate a mock response for demonstration purposes."""
        # This is a placeholder implementation
        # In a real system, this would call an LLM API
        
        if not chunks:
            return "I don't have enough information to answer your question."
        
        # Create a simple response based on the context
        response_parts = [
            f"Based on the provided context, here's what I found regarding your query: '{query}'",
            "",
            f"I found {len(chunks)} relevant pieces of information:",
        ]
        
        # Add key information from top chunks
        for i, chunk in enumerate(chunks[:3], 1):  # Show top 3 chunks
            snippet = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            similarity = f"(similarity: {chunk.similarity_score:.3f})" if chunk.similarity_score > 0 else ""
            response_parts.append(f"{i}. {snippet} {similarity}")
        
        if len(chunks) > 3:
            response_parts.append(f"... and {len(chunks) - 3} more relevant sources.")
        
        response_parts.extend([
            "",
            "Note: This is a demonstration response. In a production system, "
            "this would be generated by a language model using the retrieved context."
        ])
        
        return "\n".join(response_parts)
    
    def _calculate_confidence(self, chunks: List[Chunk]) -> float:
        """Calculate confidence score based on retrieved chunks."""
        if not chunks:
            return 0.0
        
        # Simple confidence calculation based on similarity scores
        similarities = [chunk.similarity_score for chunk in chunks if chunk.similarity_score > 0]
        
        if not similarities:
            return 0.5  # Default confidence when no similarity scores
        
        # Average similarity with some weighting for number of chunks
        avg_similarity = sum(similarities) / len(similarities)
        chunk_factor = min(len(chunks) / self.config.retrieval_top_k, 1.0)
        
        confidence = avg_similarity * chunk_factor
        return min(confidence, 1.0)
    
    def _explain_process(self, query: str, chunks: List[Chunk]) -> Dict[str, Any]:
        """Provide explanation of the RAG process."""
        return {
            'rag_type': 'vanilla',
            'query_length': len(query),
            'chunks_retrieved': len(chunks),
            'similarity_threshold': self.config.similarity_threshold,
            'chunking_strategy': self.config.chunking.strategy,
            'embedding_model': self.embedding_system.model_name,
            'embedding_dimension': self.embedding_system.embedding_dimension,
            'top_similarities': [
                chunk.similarity_score for chunk in chunks[:5] 
                if chunk.similarity_score > 0
            ],
            'source_documents': list(set(
                chunk.source_document for chunk in chunks 
                if chunk.source_document
            )),
            'process_steps': [
                '1. Query embedding generation',
                '2. Similarity computation with all chunks',
                '3. Top-k retrieval based on cosine similarity',
                '4. Context formatting from retrieved chunks',
                '5. Response generation (mock implementation)'
            ]
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        avg_retrieval_time = (
            self.stats['total_retrieval_time'] / max(self.stats['queries_processed'], 1)
        )
        avg_generation_time = (
            self.stats['total_generation_time'] / max(self.stats['queries_processed'], 1)
        )
        
        return {
            **self.stats,
            'total_chunks': len(self.chunks),
            'total_documents': len(self.documents),
            'avg_retrieval_time': avg_retrieval_time,
            'avg_generation_time': avg_generation_time,
            'embedding_model': self.embedding_system.model_name,
            'chunking_strategy': self.config.chunking.strategy
        }
    
    def clear(self) -> None:
        """Clear all stored documents and chunks."""
        self.documents.clear()
        self.chunks.clear()
        self.chunk_embeddings = None
        self.embedding_system.clear_cache()
        
        # Reset stats
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'queries_processed': 0,
            'total_retrieval_time': 0.0,
            'total_generation_time': 0.0
        }
        
        logger.info("Cleared all documents and chunks from VanillaRAG")