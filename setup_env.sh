#!/bin/bash
# Setup script for ATS Resume Optimizer

echo "Setting up ATS Resume Optimizer..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p embeddings_db
mkdir -p output

# Copy config template if config doesn't exist
if [ ! -f config.yaml ]; then
    cp config.yaml.example config.yaml
    echo "Created config.yaml from template. Please edit it with your API keys."
fi

echo "Setup complete!"
echo "1. Edit config.yaml with your API keys"
echo "2. Run: python main.py <job_description.txt> <resume.tex>"

