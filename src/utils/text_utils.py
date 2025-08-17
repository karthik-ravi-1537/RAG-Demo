"""Text processing utilities."""

import re

import tiktoken


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove special characters but keep punctuation
    text = re.sub(r"[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/]", "", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def split_text(text: str, separators: list[str] = None, max_length: int | None = None) -> list[str]:
    """Split text using hierarchical separators."""
    if not text:
        return []

    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    chunks = [text]

    for separator in separators:
        new_chunks = []
        for chunk in chunks:
            if max_length and len(chunk) <= max_length:
                new_chunks.append(chunk)
            else:
                split_chunks = chunk.split(separator)
                new_chunks.extend([c.strip() for c in split_chunks if c.strip()])
        chunks = new_chunks

    # Filter out empty chunks
    return [chunk for chunk in chunks if chunk.strip()]


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to rough estimation (4 chars per token)
        return len(text) // 4


def truncate_text(text: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> str:
    """Truncate text to fit within token limit."""
    if count_tokens(text, model) <= max_tokens:
        return text

    try:
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    except Exception:
        # Fallback to character-based truncation
        estimated_chars = max_tokens * 4
        return text[:estimated_chars]


def extract_sentences(text: str) -> list[str]:
    """Extract sentences from text."""
    # Simple sentence splitting - could be improved with NLTK
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity using word overlap."""
    if not text1 or not text2:
        return 0.0

    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    if not union:
        return 0.0

    return len(intersection) / len(union)
