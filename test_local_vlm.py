#!/usr/bin/env python3
"""
Test script for Local VLM Auto-Selection and Inference.
"""
import sys
import os
import cv2
import numpy as np
import logging
import torch

# Add current dir to path
sys.path.append(os.getcwd())

from local_describer import LocalDescriber

# Setup Simple Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_vlm")

def create_dummy_image():
    # Create a simple image (e.g., a red circle on white background)
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:] = (255, 255, 255) # White
    cv2.circle(img, (150, 150), 50, (0, 0, 255), -1) # Red circle (BGR)
    return img

def main():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"Total VRAM: {props.total_memory / 1024**3:.2f} GB")
    
    print("\n--- Initializing LocalDescriber ---")
    describer = LocalDescriber()
    
    print(f"\nSelected Model: {describer.model_id}")
    print(f"Model Type: {describer.model_type}")
    print(f"Target Device: {describer.device}")
    
    # Force load (usually lazy)
    print("\nLoading model (this may take time)...")
    describer.ensure_loaded()
    
    # Wait for load in a simple loop
    import time
    while not describer.loaded:
        time.sleep(1)
        print(".", end="", flush=True)
    print("\nModel Loaded!")
    
    print("\n--- Running Inference ---")
    img = create_dummy_image()
    # Prompt relevant to the dummy image
    prompt = "Describe this image in detail."
    
    result = describer.analyze_image(img, prompt=prompt)
    print(f"\nResult: {result}")
    
    if result:
        print("\nPASS: Valid response received.")
    else:
        print("\nFAIL: No response.")

if __name__ == "__main__":
    main()
