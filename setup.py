"""
Setup Script
Handles initial setup and configuration
"""

import subprocess
import sys
import os
from pathlib import Path


def install_requirements():
    """Install Python requirements"""
    print("Installing Python requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False


def setup_huggingface_token():
    """Setup HuggingFace token (optional)"""
    print("\nHuggingFace Token Setup (Optional)")
    print("If you have a HuggingFace token, you can set it up now.")
    print("This is optional but may help with model downloads.")
    
    token = input("Enter your HuggingFace token (or press Enter to skip): ").strip()
    
    if token:
        try:
            from huggingface_hub import login
            login(token=token)
            print("HuggingFace token configured successfully!")
        except Exception as e:
            print(f"Error configuring HuggingFace token: {e}")
    else:
        print("Skipping HuggingFace token setup.")


def check_system_requirements():
    """Check system requirements"""
    print("Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        return False
    
    print(f"✓ Python {sys.version}")
    
    # Check available space (basic check)
    try:
        stat = os.statvfs('.')
        free_space_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        print(f"✓ Available space: {free_space_gb:.1f} GB")
        
        if free_space_gb < 5:
            print("Warning: Less than 5GB available space. This may not be enough for models and embeddings.")
    except:
        print("✓ Could not check disk space")
    
    return True


def create_directories():
    """Create necessary directories"""
    print("Creating directories...")
    
    directories = [
        "./recipe_chroma_db",
        "./models",
        "./data"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(exist_ok=True)
        print(f"✓ Created/verified directory: {dir_path}")


def main():
    """Main setup function"""
    print("="*60)
    print("RECIPE GENERATION SYSTEM - SETUP")
    print("="*60)
    
    # Check system requirements
    if not check_system_requirements():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("Setup failed: Could not install requirements")
        sys.exit(1)
    
    # Setup HuggingFace token
    setup_huggingface_token()
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("Next steps:")
    print("1. Place your recipe CSV file in the ./data/ directory")
    print("2. Run: python main.py --csv ./data/your_recipe_file.csv")
    print("3. The system will automatically:")
    print("   - Install and start Ollama")
    print("   - Download the AI model")
    print("   - Build the vector database")
    print("   - Start interactive recipe generation")
    print("="*60)


if __name__ == "__main__":
    main()