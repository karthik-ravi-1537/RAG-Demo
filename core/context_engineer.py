"""Context engineering module for optimizing RAG context preparation."""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from core.interfaces import BaseContextEngineer
from core.data_models import Chunk
from core.exceptions import ProcessingError
from utils.text_utils import count_tokens, truncate_text
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ContextTemplate:
    """Template for formatting context."""
    name: str
    template: str
    description: str
    max_tokens: int = 4000


class ContextEngineer(BaseContextEngineer):
    """Context engineering module for optimizing context preparation and formatting."""
    
    def __init__(self):
        """Initialize the context engineer."""
        self.templates = {
            "default": ContextTemplate(
                name="default",
                template="Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
                description="Simple context and question format"
            ),
            "detailed": ContextTemplate(
                name="detailed", 
                template="Based on the following information:\n\n{context}\n\nQuestion: {query}\n\nAnswer:",
                description="Detailed format with instructions"
            )
        }
        logger.info("Initialized ContextEngineer")
    
    def format_context(self, chunks: List[Chunk], template: str = "default", **kwargs) -> str:
        """Format chunks into context using specified template."""
        if not chunks:
            return ""
        
        # Prepare context content
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source_info = f" [Source: {chunk.source_document}]" if chunk.source_document else ""
            context_parts.append(f"{i}.{source_info}\n{chunk.content}")
        
        context_content = "\n\n".join(context_parts)
        
        # Get template
        context_template = self.templates.get(template, self.templates["default"])
        
        # Format using template
        return context_template.template.format(
            context=context_content,
            query=kwargs.get('query', ''),
            **kwargs
        )
    
    def rank_by_relevance(self, chunks: List[Chunk], query: str) -> List[Chunk]:
        """Rank chunks by relevance to query."""
        return sorted(chunks, key=lambda x: x.similarity_score, reverse=True)
    
    def compress_context(self, context: str, max_tokens: int) -> str:
        """Compress context to fit within token limits."""
        current_tokens = count_tokens(context)
        if current_tokens <= max_tokens:
            return context
        return truncate_text(context, max_tokens)