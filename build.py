#!/usr/bin/env python3
"""
Build script for Render deployment
"""
import os
import sys
import subprocess

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"{'='*50}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: {description} failed with code {result.returncode}")
        sys.exit(1)
    print(f"✓ {description} completed successfully")

def main():
    """Main build process"""
    print("Starting Render Build Process...")
    
    # Install dependencies
    run_command(
        "pip install --upgrade pip && pip install -r requirements.txt",
        "Installing dependencies"
    )
    
    # Download NLTK data
    run_command(
        "python -m nltk.downloader -d /root/nltk_data punkt stopwords",
        "Downloading NLTK data"
    )
    
    # Create directories
    print("\nCreating required directories...")
    for directory in ['models', 'uploads', 'data', 'logs']:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ Created {directory}/")
    
    # Train model
    run_command(
        "python train.py",
        "Training ML model"
    )
    
    print("\n" + "="*50)
    print("✓ Build process completed successfully!")
    print("="*50)

if __name__ == "__main__":
    main()
