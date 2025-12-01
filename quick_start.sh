#!/bin/bash

# Quick Start Script for Object Removal
# This script helps you get started with processing your image sequences

set -e  # Exit on error

echo "=================================="
echo "Object Removal - Quick Start"
echo "=================================="
echo ""

# Check if we're in the correct directory
if [ ! -f "batch_object_removal_with_flow.py" ]; then
    echo "Error: Please run this script from the CityScapeGS directory"
    exit 1
fi

# Check Python
echo "Checking Python..."
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.8+"
    exit 1
fi
echo "✓ Python found: $(python --version)"
echo ""

# Check CUDA
echo "Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ CUDA available:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ CUDA not detected. Processing will be slow on CPU."
fi
echo ""

# Check if models need to be downloaded
echo "Note: First run will download models (~2-3 GB)"
echo "Models will be cached for future use."
echo ""

# Ask user what to do
echo "What would you like to do?"
echo ""
echo "1) Run test (RECOMMENDED - processes 5 images)"
echo "2) Process first 10 sequences"
echo "3) Process all sequences (317 sequences)"
echo "4) Process custom range"
echo "5) Exit"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo ""
        echo "Running test on 5 images..."
        python test_object_removal.py
        ;;
    2)
        echo ""
        echo "Processing first 10 sequences..."
        read -p "Continue? [y/N]: " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            python batch_object_removal_with_flow.py --start-seq 0 --end-seq 10
        fi
        ;;
    3)
        echo ""
        echo "WARNING: This will process ALL 317 sequences!"
        echo "Estimated time: 5-12 hours"
        read -p "Are you sure? [y/N]: " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            python batch_object_removal_with_flow.py
        fi
        ;;
    4)
        echo ""
        read -p "Start sequence number: " start_seq
        read -p "End sequence number: " end_seq
        echo "Processing sequences $start_seq to $end_seq..."
        read -p "Continue? [y/N]: " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            python batch_object_removal_with_flow.py --start-seq $start_seq --end-seq $end_seq
        fi
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=================================="
echo "Processing complete!"
echo "=================================="
echo ""
echo "Check your results in the filtered_camera_0 folders"
echo ""
echo "For more options, see:"
echo "  - OBJECT_REMOVAL_SUMMARY.md"
echo "  - OBJECT_REMOVAL_README.md"
echo ""
