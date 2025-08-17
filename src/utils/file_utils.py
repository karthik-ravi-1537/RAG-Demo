"""File handling utilities."""

import json
import mimetypes
import pickle
from pathlib import Path
from typing import Any

try:
    from core.exceptions import ProcessingError
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from core.exceptions import ProcessingError


def ensure_directory(path: str | Path) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_document(file_path: str | Path) -> str:
    """Load document content from file."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise ProcessingError(f"File not found: {file_path}")

    try:
        # Determine file type
        mime_type, _ = mimetypes.guess_type(str(file_path))

        if file_path.suffix.lower() == ".txt":
            with open(file_path, encoding="utf-8") as f:
                return f.read()

        elif file_path.suffix.lower() in [".md", ".markdown"]:
            with open(file_path, encoding="utf-8") as f:
                return f.read()

        elif file_path.suffix.lower() == ".json":
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
                # Convert JSON to readable text
                return json.dumps(data, indent=2)

        else:
            # Try to read as text
            try:
                with open(file_path, encoding="utf-8") as f:
                    return f.read()
            except UnicodeDecodeError:
                # Try with different encoding
                with open(file_path, encoding="latin-1") as f:
                    return f.read()

    except Exception as e:
        raise ProcessingError(f"Failed to load document {file_path}: {str(e)}")


def save_results(results: dict[str, Any] | list[dict[str, Any]], output_path: str | Path, format: str = "json") -> None:
    """Save results to file in specified format."""
    output_path = Path(output_path)
    ensure_directory(output_path.parent)

    try:
        if format.lower() == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)

        elif format.lower() == "pickle":
            with open(output_path, "wb") as f:
                pickle.dump(results, f)

        else:
            raise ProcessingError(f"Unsupported output format: {format}")

    except Exception as e:
        raise ProcessingError(f"Failed to save results to {output_path}: {str(e)}")


def load_results(file_path: str | Path) -> dict[str, Any] | list[dict[str, Any]]:
    """Load results from file."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise ProcessingError(f"Results file not found: {file_path}")

    try:
        if file_path.suffix.lower() == ".json":
            with open(file_path, encoding="utf-8") as f:
                return json.load(f)

        elif file_path.suffix.lower() in [".pkl", ".pickle"]:
            with open(file_path, "rb") as f:
                return pickle.load(f)

        else:
            raise ProcessingError(f"Unsupported file format: {file_path.suffix}")

    except Exception as e:
        raise ProcessingError(f"Failed to load results from {file_path}: {str(e)}")


def get_file_info(file_path: str | Path) -> dict[str, Any]:
    """Get file information including size, type, and modification time."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise ProcessingError(f"File not found: {file_path}")

    stat = file_path.stat()
    mime_type, _ = mimetypes.guess_type(str(file_path))

    return {
        "path": str(file_path),
        "name": file_path.name,
        "size": stat.st_size,
        "modified": stat.st_mtime,
        "mime_type": mime_type,
        "extension": file_path.suffix.lower(),
    }


def list_files(directory: str | Path, extensions: list[str] | None = None, recursive: bool = False) -> list[Path]:
    """List files in directory with optional filtering."""
    directory = Path(directory)

    if not directory.exists():
        raise ProcessingError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise ProcessingError(f"Path is not a directory: {directory}")

    pattern = "**/*" if recursive else "*"
    files = []

    for file_path in directory.glob(pattern):
        if file_path.is_file():
            if extensions is None or file_path.suffix.lower() in extensions:
                files.append(file_path)

    return sorted(files)
