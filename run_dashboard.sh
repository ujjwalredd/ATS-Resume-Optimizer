#!/bin/bash
# Script to run the Streamlit dashboard

echo "Starting ATS Resume Optimizer Dashboard..."
echo ""

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Run streamlit
streamlit run dashboard.py --server.port 8501 --server.address localhost

