# RAG Demo

Evolution of Retrieval-Augmented Generation from vanilla to sophisticated Graph-RAG implementations.

## Overview

Four distinct RAG approaches:

1. **Vanilla RAG**: Traditional similarity-based retrieval
2. **Hierarchical RAG**: Multi-level document representation  
3. **Graph-RAG**: Knowledge graph construction with entity-relationship retrieval
4. **Multi-modal RAG**: Cross-modal retrieval (text, images, structured data)

## Features

- Multiple chunking strategies (fixed-size, semantic, recursive, structure-aware)
- Advanced context engineering with relevance ranking
- Conversation management and multi-turn context
- Comprehensive evaluation and comparison framework
- Interactive visualizations and process explanations

## Installation

### Prerequisites
- [Homebrew](https://brew.sh/) for installing uv
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Method 1: uv

```bash
brew install uv

git clone https://github.com/karthik-ravi-1537/RAG-Demo.git
cd RAG-Demo

uv sync

source .venv/bin/activate
```

### Method 2: conda

```bash
git clone https://github.com/karthik-ravi-1537/RAG-Demo.git
cd RAG-Demo

conda env create -f environment.yml
conda activate rag-demo
```

### Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Verification
```bash
python test_setup.py
```

## Quick Start

```bash
# Run vanilla RAG demo
python demo_vanilla_rag.py

# Try comprehensive demo
python demo_all_rag_systems.py

# Run tests
python -m pytest tests/ -v
```

## Project Structure

```
RAG-Demo/
├── src/
│   ├── core/             # Core RAG components
│   ├── utils/            # Utility modules
│   ├── vanilla_rag/      # Vanilla RAG implementation
│   ├── hierarchical_rag/ # Hierarchical RAG
│   ├── graph_rag/        # Graph-RAG implementation
│   └── evaluation/       # Evaluation and metrics
├── tests/                # Comprehensive test suite
├── config/               # Configuration files
└── demo_*.py             # Demonstration scripts
```

## Configuration

System uses YAML configuration files. See `src/config/default_config.yaml`:

```yaml
chunking:
  strategy: "semantic"
  chunk_size: 512
  overlap: 50

embedding:
  model_name: "all-MiniLM-L6-v2"
  
retrieval_top_k: 5
similarity_threshold: 0.7
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Test core components
python tests/test_core_components.py

# Verify setup
python test_setup.py
```

## Contributing

### Development Setup

```bash
# Clone and install with dev dependencies
uv sync --dev
source .venv/bin/activate

# Install development tools
uv pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run formatting and linting
pre-commit run --all-files
```

### Development Tools

- **ruff**: Fast Python linter and formatter
- **black**: Code formatter
- **pre-commit**: Git hooks for code quality
- **pytest**: Testing framework