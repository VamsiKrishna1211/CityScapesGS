import cv2
from pathlib import Path
from PIL import Image
import os
import numpy as np
import argparse
import shutil

def video_to_images(video_path: Path, output_folder: Path, frame_rate=15):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_count = 0
    saved_feame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_count % frame_rate == 0:
            # Save the current frame as an image
            image_path = f"{output_folder}/frame_{frame_count:04d}.jpg"
            cv2.imwrite(image_path, frame)
            saved_feame_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_feame_count} frames to {output_folder}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert video to images.")
    parser.add_argument("--video_path", type=str, help="Path to the input video file.")
    parser.add_argument("--output_folder", type=str, help="Path to the output folder for images.")
    parser.add_argument("--frame_rate", type=int, default=15, help="Frame rate for extracting images (default: 15).")
    
    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    output_folder = Path(args.output_folder)

    
    if not output_folder.exists():
        os.makedirs(output_folder)
    
    video_to_images(video_path, output_folder, args.frame_rate)
