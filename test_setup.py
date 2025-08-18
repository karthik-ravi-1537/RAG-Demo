#!/usr/bin/env python3
"""
Simple test script to verify the RAG Demo setup and imports work correctly.
"""
import os
import sys
from pathlib import Path

# Add project root to path for proper imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")

    try:
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import scipy  # noqa: F401

        print("✅ Core scientific packages imported successfully")

        import nltk  # noqa: F401
        import tiktoken  # noqa: F401

        print("✅ NLP packages imported successfully")

        import sentence_transformers  # noqa: F401
        import sklearn  # noqa: F401
        import transformers  # noqa: F401

        print("✅ ML packages imported successfully")

        import openai  # noqa: F401

        print("✅ API packages imported successfully")

        import networkx  # noqa: F401

        print("✅ Graph processing packages imported successfully")
        sys.path.insert(0, os.getcwd())
        from core.chunking_engine import ChunkingEngine  # noqa: F401
        from core.document_processor import DocumentProcessor  # noqa: F401
        from core.embedding_system import EmbeddingSystem  # noqa: F401

        print("✅ RAG core components imported successfully")

        from utils.config_manager import ConfigManager  # noqa: F401
        from utils.logging_utils import setup_logger  # noqa: F401

        print("✅ RAG utilities imported successfully")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        raise AssertionError(f"Import error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise AssertionError(f"Unexpected error: {e}")


def test_configuration():
    """Test configuration loading."""
    print("\n⚙️  Testing configuration...")

    try:
        from utils.config_manager import ConfigManager

        config_manager = ConfigManager()
        config = config_manager.load_config("config/default_config.yaml")

        print("✅ Default config loaded successfully")
        print(f"✅ Chunking strategy: {config.chunking.strategy}")
        print(f"✅ Embedding model: {config.embedding.model_name}")
        print(f"✅ Retrieval top-k: {config.retrieval_top_k}")

    except Exception as e:
        print(f"❌ Configuration error: {e}")
        raise AssertionError(f"Configuration error: {e}")


def test_basic_functionality():
    """Test basic functionality without requiring API keys."""
    print("\n🔧 Testing basic functionality...")

    try:
        from core.document_processor import DocumentProcessor

        DocumentProcessor()
        print("✅ Document processor initialized successfully")

        from core.chunking_engine import ChunkingEngine

        ChunkingEngine()
        print("✅ Chunking engine initialized successfully")

        from core.data_models import EmbeddingConfig
        from core.embedding_system import EmbeddingSystem

        config = EmbeddingConfig(model_name="all-MiniLM-L6-v2")
        EmbeddingSystem(config)
        print("✅ Embedding system initialized successfully")

    except Exception as e:
        print(f"❌ Functionality error: {e}")
        raise AssertionError(f"Functionality error: {e}")


def test_project_structure():
    """Test that all expected files and directories exist."""
    print("\n📁 Testing project structure...")

    expected_files = [
        "README.md",
        "requirements.txt",
        "environment.yml",
        ".env.example",
        ".gitignore",
        "src/config/default_config.yaml",
        "src/core/__init__.py",
        "src/core/document_processor.py",
        "src/core/chunking_engine.py",
        "src/core/embedding_system.py",
        "src/utils/__init__.py",
        "src/utils/config_manager.py",
        "tests/test_core_components.py",
        "tests/verify_structure.py",
    ]

    missing_files = []
    for file_path in expected_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        raise AssertionError(f"Missing files: {missing_files}")

    print("✅ All expected files found")


def test_environment_variables():
    """Test environment variable setup."""
    print("\n🌍 Testing environment variables...")

    try:
        from dotenv import load_dotenv

        load_dotenv()

        if os.path.exists(".env"):
            print("✅ .env file found")
        else:
            print("⚠️  .env file not found (expected for fresh setup)")
            print("   💡 Copy .env.example to .env and add your API keys")

        if os.path.exists(".env.example"):
            print("✅ .env.example template found")
        else:
            print("❌ .env.example template missing")
            raise AssertionError(".env.example template missing")

    except Exception as e:
        print(f"❌ Environment variable error: {e}")
        raise AssertionError(f"Environment variable error: {e}")


def main():
    """Run all tests."""
    print("🚀 RAG Demo - Setup Test")
    print("=" * 50)

    tests = [
        ("Project Structure", test_project_structure),
        ("Environment Variables", test_environment_variables),
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Basic Functionality", test_basic_functionality),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            test_func()
            results.append((test_name, True))
        except AssertionError as e:
            print(f"❌ {test_name} test failed: {e}")
            results.append((test_name, False))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)

    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🎉 All tests passed! Your RAG Demo setup is ready.")
        print("\n💡 Next steps:")
        print("1. Add your real API keys to the .env file")
        print("2. Run: python demo_vanilla_rag.py")
        print("3. Explore different RAG implementations")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
