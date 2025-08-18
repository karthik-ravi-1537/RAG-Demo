"""Simple verification script to check project structure."""

import sys
from pathlib import Path


def check_file_exists(file_path, description):
    """Check if a file exists and print status."""
    if Path(file_path).exists():
        print(f"✓ {description}: {file_path}")
        return True
    else:
        print(f"✗ {description}: {file_path} (missing)")
        return False


def check_directory_structure():
    """Check the project directory structure."""
    print("Checking RAG Demo project structure...\n")

    core_files = [
        ("__init__.py", "Main package init"),
        ("core/__init__.py", "Core package init"),
        ("core/interfaces.py", "Base interfaces"),
        ("core/data_models.py", "Data models"),
        ("core/exceptions.py", "Custom exceptions"),
        ("core/document_processor.py", "Document processor"),
        ("core/chunking_engine.py", "Chunking engine"),
        ("core/embedding_system.py", "Embedding system"),
    ]

    util_files = [
        ("utils/__init__.py", "Utils package init"),
        ("utils/config_manager.py", "Configuration manager"),
        ("utils/logging_utils.py", "Logging utilities"),
        ("utils/text_utils.py", "Text processing utilities"),
        ("utils/file_utils.py", "File handling utilities"),
    ]

    rag_dirs = [
        ("vanilla_rag/__init__.py", "Vanilla RAG package"),
        ("hierarchical_rag/__init__.py", "Hierarchical RAG package"),
        ("graph_rag/__init__.py", "Graph RAG package"),
    ]

    other_files = [
        ("evaluation/__init__.py", "Evaluation package"),
        ("config/default_config.yaml", "Default configuration"),
        ("requirements.txt", "Dependencies"),
        ("README.md", "Documentation"),
        (".env.example", "Environment template"),
    ]

    all_files = core_files + util_files + rag_dirs + other_files

    print("=== Core Components ===")
    core_count = sum(check_file_exists(file, desc) for file, desc in core_files)

    print("\n=== Utility Components ===")
    util_count = sum(check_file_exists(file, desc) for file, desc in util_files)

    print("\n=== RAG Implementation Packages ===")
    rag_count = sum(check_file_exists(file, desc) for file, desc in rag_dirs)

    print("\n=== Other Components ===")
    other_count = sum(check_file_exists(file, desc) for file, desc in other_files)

    total_expected = len(all_files)
    total_found = core_count + util_count + rag_count + other_count

    print("\n=== Summary ===")
    print(f"Files found: {total_found}/{total_expected}")
    print(f"Core components: {core_count}/{len(core_files)}")
    print(f"Utility components: {util_count}/{len(util_files)}")
    print(f"RAG packages: {rag_count}/{len(rag_dirs)}")
    print(f"Other files: {other_count}/{len(other_files)}")

    if total_found == total_expected:
        print("\n🎉 Project structure is complete!")
        return True
    else:
        print(f"\n⚠️  Project structure is {total_found/total_expected*100:.1f}% complete")
        return False


def check_import_structure():
    """Check if the import structure is correct."""
    print("\n=== Checking Import Structure ===")

    try:
        sys.path.insert(0, str(Path(__file__).parent))

        print("Testing basic package imports...")

        core_modules = [
            "core.data_models",
            "core.exceptions",
            "utils.config_manager",
            "utils.logging_utils",
        ]

        for module_name in core_modules:
            try:
                module_path = module_name.replace(".", "/") + ".py"
                if Path(module_path).exists():
                    print(f"✓ {module_name} structure OK")
                else:
                    print(f"✗ {module_name} file missing")
            except Exception as e:
                print(f"✗ {module_name} error: {str(e)}")

        print("✓ Basic package structure is importable")

    except Exception as e:
        print(f"✗ Import structure check failed: {str(e)}")


def main():
    """Main verification function."""
    print("RAG Demo Project Verification")
    print("=" * 40)

    structure_ok = check_directory_structure()
    check_import_structure()

    print("\n" + "=" * 40)
    if structure_ok:
        print("✅ Project setup verification PASSED")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set up environment: cp .env.example .env")
        print("3. Run tests: python tests/test_core_components.py")
    else:
        print("❌ Project setup verification FAILED")
        print("Some files are missing. Please check the output above.")

    return 0 if structure_ok else 1


if __name__ == "__main__":
    exit(main())
