#!/bin/bash

# Setup script for Mapillary Data Downloader

echo "Setting up Mapillary Data Downloader..."
echo "======================================"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is required but not installed."
    exit 1
fi

echo "✓ pip3 found"

# Install requirements
echo ""
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

# Make scripts executable
chmod +x mapillary_downloader.py
chmod +x test_config.py

echo ""
echo "Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Get your Mapillary access token from: https://mapillary.com/dashboard/developers"
echo "2. Test your configuration:"
echo "   python3 test_config.py --bbox \"west,south,east,north\" --token \"your_token\""
echo "3. Run the downloader:"
echo "   python3 mapillary_downloader.py --bbox \"west,south,east,north\" --token \"your_token\""
echo ""
echo "Example bounding boxes (small areas only!):"
echo "  Miami:  \"-80.134,25.773,-80.126,25.789\""
echo "  Paris:  \"2.294,48.857,2.296,48.859\""
echo "  London: \"-0.0899,51.5044,-0.0859,51.5074\""
echo ""
echo "Remember: Bounding box area must be < 0.01 square degrees!"
