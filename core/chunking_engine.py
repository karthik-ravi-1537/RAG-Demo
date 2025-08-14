"""Text chunking for RAG."""

import re
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from core.interfaces import BaseChunker
from core.data_models import Chunk, ProcessedDocument, ChunkingConfig
from core.exceptions import ChunkingError
from utils.text_utils import split_text, count_tokens, extract_sentences
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk(self, text: str, config: ChunkingConfig, **kwargs) -> List[str]:
        """Split text into chunks using specific strategy."""
        pass


class FixedSizeChunker(ChunkingStrategy):
    """Fixed-size chunking with configurable overlap."""
    
    def chunk(self, text: str, config: ChunkingConfig, **kwargs) -> List[str]:
        """Split text into fixed-size chunks with overlap."""
        if not text.strip():
            return []
        
        chunk_size = config.chunk_size
        overlap = config.overlap
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            if end < len(text) and not text[end].isspace():
                last_space = chunk.rfind(' ')
                if last_space > chunk_size // 2:
                    chunk = chunk[:last_space]
                    end = start + last_space
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            # Move start position with overlap
            start = max(start + chunk_size - overlap, end)
            
            if start >= len(text):
                break
        
        return chunks


class SemanticChunker(ChunkingStrategy):
    """Semantic chunking using sentence boundaries and similarity."""
    
    def chunk(self, text: str, config: ChunkingConfig, **kwargs) -> List[str]:
        """Split text into semantically coherent chunks."""
        if not text.strip():
            return []
        
        sentences = extract_sentences(text)
        if not sentences:
            return FixedSizeChunker().chunk(text, config)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > config.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.strip()) >= config.min_chunk_size:
                    chunks.append(chunk_text.strip())
                if config.overlap > 0 and len(current_chunk) > 1:
                    overlap_sentences = []
                    overlap_length = 0
                    for sent in reversed(current_chunk):
                        if overlap_length + len(sent) <= config.overlap:
                            overlap_sentences.insert(0, sent)
                            overlap_length += len(sent)
                        else:
                            break
                    current_chunk = overlap_sentences
                    current_length = overlap_length
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.strip()) >= config.min_chunk_size:
                chunks.append(chunk_text.strip())
        
        return chunks


class RecursiveChunker(ChunkingStrategy):
    """Recursive chunking that respects document structure."""
    
    def chunk(self, text: str, config: ChunkingConfig, **kwargs) -> List[str]:
        """Split text recursively using hierarchical separators."""
        if not text.strip():
            return []
        
        separators = config.separators or ["\n\n", "\n", ". ", " "]
        return self._recursive_split(text, separators, config.chunk_size, config.min_chunk_size)
    
    def _recursive_split(self, text: str, separators: List[str], max_size: int, min_size: int) -> List[str]:
        """Recursively split text using separators."""
        if len(text) <= max_size:
            return [text] if len(text) >= min_size else []
        
        if not separators:
            # No more separators, use character-based splitting
            chunks = []
            for i in range(0, len(text), max_size):
                chunk = text[i:i + max_size]
                if len(chunk) >= min_size:
                    chunks.append(chunk)
            return chunks
        
        # Try current separator
        separator = separators[0]
        remaining_separators = separators[1:]
        
        splits = text.split(separator)
        chunks = []
        current_chunk = ""
        
        for split in splits:
            test_chunk = current_chunk + (separator if current_chunk else "") + split
            
            if len(test_chunk) <= max_size:
                current_chunk = test_chunk
            else:
                if current_chunk and len(current_chunk) >= min_size:
                    chunks.append(current_chunk)
                
                if len(split) > max_size:
                    sub_chunks = self._recursive_split(split, remaining_separators, max_size, min_size)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= min_size:
            chunks.append(current_chunk)
        
        return chunks


class StructureAwareChunker(ChunkingStrategy):
    """Structure-aware chunking that preserves document hierarchy."""
    
    def chunk(self, text: str, config: ChunkingConfig, **kwargs) -> List[str]:
        """Split text while preserving document structure."""
        if not text.strip():
            return []
        
        structure_info = self._analyze_structure(text)
        
        if structure_info['has_headers']:
            return self._chunk_by_headers(text, config, structure_info)
        elif structure_info['has_paragraphs']:
            return self._chunk_by_paragraphs(text, config)
        else:
            return SemanticChunker().chunk(text, config)
    
    def _analyze_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure."""
        lines = text.split('\n')
        
        header_pattern = re.compile(r'^#{1,6}\s+')
        headers = [line for line in lines if header_pattern.match(line)]
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        return {
            'has_headers': len(headers) > 0,
            'header_count': len(headers),
            'has_paragraphs': len(paragraphs) > 1,
            'paragraph_count': len(paragraphs),
            'total_lines': len(lines)
        }
    
    def _chunk_by_headers(self, text: str, config: ChunkingConfig, structure_info: Dict) -> List[str]:
        """Chunk text by header sections."""
        lines = text.split('\n')
        chunks = []
        current_section = []
        current_length = 0
        
        header_pattern = re.compile(r'^#{1,6}\s+')
        
        for line in lines:
            line_length = len(line)
            
            if header_pattern.match(line):
                # New header found
                if current_section and current_length >= config.min_chunk_size:
                    chunks.append('\n'.join(current_section))
                
                # Start new section
                current_section = [line]
                current_length = line_length
            else:
                # Check if adding this line would exceed chunk size
                if current_length + line_length > config.chunk_size and current_section:
                    # Finalize current chunk
                    if current_length >= config.min_chunk_size:
                        chunks.append('\n'.join(current_section))
                    
                    # Start new chunk (keep header if present)
                    header_lines = []
                    for sect_line in current_section:
                        if header_pattern.match(sect_line):
                            header_lines.append(sect_line)
                        else:
                            break
                    
                    current_section = header_lines + [line]
                    current_length = sum(len(l) for l in current_section)
                else:
                    current_section.append(line)
                    current_length += line_length
        
        # Add final section
        if current_section and current_length >= config.min_chunk_size:
            chunks.append('\n'.join(current_section))
        
        return chunks
    
    def _chunk_by_paragraphs(self, text: str, config: ChunkingConfig) -> List[str]:
        """Chunk text by paragraph boundaries."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            para_length = len(paragraph)
            
            # Check if adding this paragraph would exceed chunk size
            if current_length + para_length > config.chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = '\n\n'.join(current_chunk)
                if len(chunk_text) >= config.min_chunk_size:
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap
                if config.overlap > 0 and len(current_chunk) > 1:
                    # Keep last paragraph for overlap
                    current_chunk = [current_chunk[-1]]
                    current_length = len(current_chunk[0])
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(paragraph)
            current_length += para_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if len(chunk_text) >= config.min_chunk_size:
                chunks.append(chunk_text)
        
        return chunks


class ChunkingEngine(BaseChunker):
    """Main chunking engine that supports multiple strategies."""
    
    def __init__(self):
        self.strategies = {
            'fixed_size': FixedSizeChunker(),
            'semantic': SemanticChunker(),
            'recursive': RecursiveChunker(),
            'structure_aware': StructureAwareChunker()
        }
    
    def chunk_text(self, text: str, config: ChunkingConfig = None, **kwargs) -> List[Chunk]:
        """Split text into chunks using specified strategy."""
        if config is None:
            config = ChunkingConfig()
        
        if not text.strip():
            return []
        
        try:
            # Get chunking strategy
            strategy = self.strategies.get(config.strategy)
            if not strategy:
                logger.warning(f"Unknown chunking strategy: {config.strategy}, using fixed_size")
                strategy = self.strategies['fixed_size']
            
            # Perform chunking
            chunk_texts = strategy.chunk(text, config, **kwargs)
            
            # Create Chunk objects
            chunks = []
            for i, chunk_text in enumerate(chunk_texts):
                chunk = Chunk(
                    id=str(uuid.uuid4()),
                    content=chunk_text,
                    chunk_index=i,
                    metadata={
                        'chunking_strategy': config.strategy,
                        'chunk_size': len(chunk_text),
                        'token_count': count_tokens(chunk_text),
                        'chunk_position': i / len(chunk_texts) if chunk_texts else 0
                    }
                )
                chunks.append(chunk)
            
            logger.info(f"Chunked text into {len(chunks)} chunks using {config.strategy} strategy")
            return chunks
            
        except Exception as e:
            logger.error(f"Chunking failed: {str(e)}")
            raise ChunkingError(f"Failed to chunk text: {str(e)}")
    
    def chunk_document(self, document: ProcessedDocument, config: ChunkingConfig = None, **kwargs) -> List[Chunk]:
        """Split document into chunks preserving structure and metadata."""
        if config is None:
            config = ChunkingConfig()
        
        try:
            # Chunk the document content
            chunks = self.chunk_text(document.content, config, **kwargs)
            
            # Add document metadata to chunks
            for chunk in chunks:
                chunk.source_document = document.id
                chunk.metadata.update({
                    'source_file': document.file_path,
                    'document_type': document.document_type,
                    'source_metadata': document.metadata
                })
            
            logger.info(f"Chunked document {document.id} into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Document chunking failed for {document.id}: {str(e)}")
            raise ChunkingError(f"Failed to chunk document: {str(e)}")
    
    def chunk_documents_batch(self, documents: List[ProcessedDocument], config: ChunkingConfig = None) -> List[Chunk]:
        """Chunk multiple documents in batch."""
        if config is None:
            config = ChunkingConfig()
        
        all_chunks = []
        
        for document in documents:
            try:
                chunks = self.chunk_document(document, config)
                all_chunks.extend(chunks)
            except ChunkingError as e:
                logger.error(f"Failed to chunk document {document.id}: {str(e)}")
                continue
        
        logger.info(f"Batch chunking completed: {len(all_chunks)} total chunks from {len(documents)} documents")
        return all_chunks
    
    def get_chunking_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get statistics about chunked content."""
        if not chunks:
            return {}
        
        chunk_sizes = [len(chunk.content) for chunk in chunks]
        token_counts = [chunk.metadata.get('token_count', 0) for chunk in chunks]
        
        stats = {
            'total_chunks': len(chunks),
            'total_characters': sum(chunk_sizes),
            'total_tokens': sum(token_counts),
            'avg_chunk_size': sum(chunk_sizes) / len(chunks),
            'avg_token_count': sum(token_counts) / len(chunks) if token_counts else 0,
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'strategies_used': list(set(chunk.metadata.get('chunking_strategy', 'unknown') for chunk in chunks))
        }
        
        return stats
    
    def add_chunking_strategy(self, name: str, strategy: ChunkingStrategy) -> None:
        """Add a custom chunking strategy."""
        self.strategies[name] = strategy
        logger.info(f"Added custom chunking strategy: {name}")
    
    def list_strategies(self) -> List[str]:
        """List available chunking strategies."""
        return list(self.strategies.keys())