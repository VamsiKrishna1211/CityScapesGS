#!/bin/bash

# Script to add .jpg extension to all files in images folders
# Usage: ./add_jpg_extension.sh [parent_directory]

# Use provided directory or current directory
PARENT_DIR="${1:-.}"

echo "Processing images folders in: $PARENT_DIR"

# Find all 'images' directories and process files within them
find "$PARENT_DIR" -type d -name "images" | while read -r images_dir; do
    echo "Processing: $images_dir"
    
    # Loop through all files in the images directory
    find "$images_dir" -maxdepth 1 -type f | while read -r file; do
        # Get the filename
        filename=$(basename "$file")
        
        # Check if file already has .jpg extension
        if [[ ! "$filename" =~ \.jpg$ ]]; then
            # Get the directory path
            dir=$(dirname "$file")
            
            # Rename the file by adding .jpg extension
            mv "$file" "${file}.jpg"
            echo "  Renamed: $filename -> ${filename}.jpg"
        fi
    done
done

echo "Done!"
