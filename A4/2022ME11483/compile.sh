#!/bin/bash

# Compile script for the Bayesian Network learning solution
# For Python, we don't need compilation, but we'll check dependencies

echo "Checking dependencies..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    exit 1
fi

# Check if numpy is installed
python3 -c "import numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing numpy..."
    pip3 install numpy
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install numpy. Please install it manually."
        exit 1
    fi
fi

echo "Dependencies checked successfully."
echo "Ready to run the solution."