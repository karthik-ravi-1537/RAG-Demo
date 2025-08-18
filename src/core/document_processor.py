"""Document processing for RAG."""

import hashlib
import json
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.logging_utils import get_logger
from utils.text_utils import clean_text

from core.data_models import ProcessedDocument
from core.exceptions import ProcessingError
from core.interfaces import BaseDocumentProcessor

logger = get_logger(__name__)


class DocumentProcessor(BaseDocumentProcessor):
    """Processes documents from various formats into structured representations."""

    def __init__(self):
        self.supported_formats = {
            ".txt": self._process_text,
            ".md": self._process_markdown,
            ".json": self._process_json,
            ".csv": self._process_csv,
            # PDF and DOCX will be added when dependencies are available
        }

    def process_document(self, file_path: str, doc_type: str = None) -> ProcessedDocument:
        """Process a document and return structured representation."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise ProcessingError(f"File not found: {file_path}")

        logger.info(f"Processing document: {file_path}")

        try:
            # Determine document type
            if doc_type is None:
                doc_type = self._detect_document_type(file_path)

            # Extract text content
            content = self.extract_text(str(file_path))

            # Extract metadata
            metadata = self.extract_metadata(str(file_path))

            # Generate document ID
            doc_id = self._generate_document_id(file_path, content)

            # Create processed document
            processed_doc = ProcessedDocument(
                id=doc_id, content=content, metadata=metadata, file_path=str(file_path), document_type=doc_type
            )

            logger.info(f"Successfully processed document: {file_path} ({len(content)} chars)")
            return processed_doc

        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {str(e)}")
            raise ProcessingError(f"Document processing failed: {str(e)}")

    def extract_text(self, file_path: str) -> str:
        """Extract text content from document."""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        if extension not in self.supported_formats:
            # Try to read as plain text
            try:
                return self._process_text(file_path)
            except Exception:
                raise ProcessingError(f"Unsupported file format: {extension}")

        processor = self.supported_formats[extension]
        return processor(file_path)

    def extract_metadata(self, file_path: str) -> dict[str, Any]:
        """Extract metadata from document."""
        file_path = Path(file_path)

        try:
            stat = file_path.stat()
            mime_type, _ = mimetypes.guess_type(str(file_path))

            metadata = {
                "filename": file_path.name,
                "file_size": stat.st_size,
                "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "mime_type": mime_type,
                "extension": file_path.suffix.lower(),
                "processing_time": datetime.now().isoformat(),
            }

            # Add format-specific metadata
            extension = file_path.suffix.lower()
            if extension == ".json":
                metadata.update(self._extract_json_metadata(file_path))
            elif extension == ".csv":
                metadata.update(self._extract_csv_metadata(file_path))

            return metadata

        except Exception as e:
            logger.warning(f"Failed to extract metadata from {file_path}: {str(e)}")
            return {
                "filename": file_path.name,
                "processing_time": datetime.now().isoformat(),
                "metadata_extraction_error": str(e),
            }

    def process_batch(self, file_paths: list[str | Path]) -> list[ProcessedDocument]:
        """Process multiple documents in batch."""
        processed_docs = []

        for file_path in file_paths:
            try:
                doc = self.process_document(str(file_path))
                processed_docs.append(doc)
            except ProcessingError as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                continue

        logger.info(f"Batch processing completed: {len(processed_docs)}/{len(file_paths)} documents processed")
        return processed_docs

    def _detect_document_type(self, file_path: Path) -> str:
        """Detect document type based on file extension and content."""
        extension = file_path.suffix.lower()

        type_mapping = {
            ".txt": "text",
            ".md": "markdown",
            ".json": "json",
            ".csv": "csv",
            ".pdf": "pdf",
            ".docx": "docx",
            ".doc": "doc",
        }

        return type_mapping.get(extension, "unknown")

    def _generate_document_id(self, file_path: Path, content: str) -> str:
        """Generate unique document ID based on path and content."""
        # Use file path and content hash for unique ID
        content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()[:8]
        path_hash = hashlib.md5(str(file_path).encode("utf-8")).hexdigest()[:8]
        return f"doc_{path_hash}_{content_hash}"

    def _process_text(self, file_path: Path) -> str:
        """Process plain text files."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, encoding="latin-1") as f:
                content = f.read()

        return clean_text(content)

    def _process_markdown(self, file_path: Path) -> str:
        """Process Markdown files."""
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # For now, treat as plain text
        # Could add markdown parsing later
        return clean_text(content)

    def _process_json(self, file_path: Path) -> str:
        """Process JSON files."""
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        # Convert JSON to readable text
        if isinstance(data, dict):
            text_parts = []
            for key, value in data.items():
                if isinstance(value, str | int | float):
                    text_parts.append(f"{key}: {value}")
                elif isinstance(value, list):
                    text_parts.append(f"{key}: {', '.join(map(str, value))}")
                else:
                    text_parts.append(f"{key}: {json.dumps(value)}")
            content = "\n".join(text_parts)
        else:
            content = json.dumps(data, indent=2)

        return clean_text(content)

    def _process_csv(self, file_path: Path) -> str:
        """Process CSV files."""
        try:
            import pandas as pd

            df = pd.read_csv(file_path)

            # Convert DataFrame to readable text
            text_parts = []
            text_parts.append(f"CSV with {len(df)} rows and {len(df.columns)} columns")
            text_parts.append(f"Columns: {', '.join(df.columns)}")

            # Add sample rows as text
            for idx, row in df.head(10).iterrows():
                row_text = ", ".join([f"{col}: {val}" for col, val in row.items()])
                text_parts.append(f"Row {idx}: {row_text}")

            if len(df) > 10:
                text_parts.append(f"... and {len(df) - 10} more rows")

            content = "\n".join(text_parts)

        except ImportError:
            # Fallback to basic CSV processing
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()

            content = f"CSV file with {len(lines)} lines\n"
            content += "".join(lines[:20])  # First 20 lines
            if len(lines) > 20:
                content += f"\n... and {len(lines) - 20} more lines"

        return clean_text(content)

    def _extract_json_metadata(self, file_path: Path) -> dict[str, Any]:
        """Extract metadata specific to JSON files."""
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            metadata = {
                "json_type": type(data).__name__,
                "json_keys": list(data.keys()) if isinstance(data, dict) else None,
                "json_length": len(data) if isinstance(data, list | dict) else None,
            }

            return metadata
        except Exception:
            return {}

    def _extract_csv_metadata(self, file_path: Path) -> dict[str, Any]:
        """Extract metadata specific to CSV files."""
        try:
            # Try with pandas first
            try:
                import pandas as pd

                df = pd.read_csv(file_path)
                return {
                    "csv_rows": len(df),
                    "csv_columns": len(df.columns),
                    "csv_column_names": list(df.columns),
                    "csv_dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                }
            except ImportError:
                # Fallback to basic CSV analysis
                with open(file_path, encoding="utf-8") as f:
                    lines = f.readlines()

                if lines:
                    header = lines[0].strip().split(",")
                    return {
                        "csv_rows": len(lines) - 1,  # Excluding header
                        "csv_columns": len(header),
                        "csv_column_names": header,
                    }

        except Exception:
            pass

        return {}
