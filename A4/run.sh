#!/bin/bash

# Run script for the Bayesian Network learning solution
# Usage: ./run.sh hailfinder.bif <sample_data>.dat

# Check if correct number of arguments provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <hailfinder.bif> <records.dat>"
    exit 1
fi

# Assign arguments to variables
BIF_FILE=$1
DATA_FILE=$2

# Check if input files exist
if [ ! -f "$BIF_FILE" ]; then
    echo "Error: $BIF_FILE not found."
    exit 1
fi

if [ ! -f "$DATA_FILE" ]; then
    echo "Error: $DATA_FILE not found."
    exit 1
fi

# Run the Python script
echo "Running EM algorithm to learn Bayesian Network parameters..."
python3 solve_parth.py "$BIF_FILE" "$DATA_FILE"

# Check if output file was created
if [ -f "solved_hailfinder.bif" ]; then
    echo "Learning completed successfully. Output saved to solved_hailfinder.bif"
else
    echo "Error: Failed to generate output file."
    exit 1
fi