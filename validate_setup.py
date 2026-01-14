"""
Validation script to check if all dependencies and modules are properly installed
"""

import sys

def check_imports():
    """Check if all required modules can be imported"""
    errors = []
    
    required_modules = [
        ("openai", "OpenAI"),
        ("sentence_transformers", "SentenceTransformer"),
        ("faiss", "FAISS"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML"),
        ("dotenv", "python-dotenv"),
        ("github", "PyGithub"),
        ("scholarly", "scholarly"),
        ("bs4", "BeautifulSoup4"),
    ]
    
    print("Checking required modules...")
    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError as e:
            print(f"  ✗ {display_name} - NOT FOUND")
            errors.append(f"{display_name}: {str(e)}")
    
    # Check our custom modules
    print("\nChecking custom modules...")
    custom_modules = [
        "profile_ingester",
        "embedding_store",
        "job_parser",
        "resume_parser",
        "alignment_engine",
        "rewrite_engine",
        "github_integration",
        "main"
    ]
    
    for module_name in custom_modules:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name}")
        except ImportError as e:
            print(f"  ✗ {module_name} - ERROR: {str(e)}")
            errors.append(f"{module_name}: {str(e)}")
    
    return errors

def check_config():
    """Check if config file exists"""
    import os
    if os.path.exists("config.yaml"):
        print("\n✓ config.yaml found")
        return True
    else:
        print("\n✗ config.yaml not found - copy from config.yaml.example")
        return False

def check_directories():
    """Check if required directories exist"""
    import os
    dirs = ["embeddings_db", "output"]
    all_exist = True
    
    print("\nChecking directories...")
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            print(f"  ✓ Created {dir_name}/")
        else:
            print(f"  ✓ {dir_name}/ exists")
    
    return all_exist

def main():
    print("="*60)
    print("ATS Resume Optimizer - Setup Validation")
    print("="*60)
    
    errors = check_imports()
    config_ok = check_config()
    check_directories()
    
    print("\n" + "="*60)
    if errors:
        print("❌ VALIDATION FAILED")
        print("\nErrors found:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    elif not config_ok:
        print("⚠️  VALIDATION INCOMPLETE")
        print("\nPlease create config.yaml from config.yaml.example")
        sys.exit(1)
    else:
        print("✅ VALIDATION PASSED")
        print("\nSetup looks good! You can now run the optimizer.")
        sys.exit(0)

if __name__ == "__main__":
    main()

