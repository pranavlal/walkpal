import os
import cv2
import numpy as np
import time
from dotenv import load_dotenv
from scene_describer import SceneDescriber
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    load_dotenv()
    key = os.getenv("open_router_api_key")
    if not key:
        print("ERROR: open_router_api_key not found in .env")
        return

    print(f"Key found: {key[:5]}...")

    describer = SceneDescriber(api_key=key)
    
    # Create a dummy image (random noise or black)
    # Let's make something slightly structured so it's not partial black
    # 640x480 random noise
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Draw a circle to make it "something"
    cv2.circle(img, (320, 240), 100, (0, 0, 255), -1)
    
    print("Testing SceneDescriber process...")
    # First call will trigger a task (background thread) but return None immediately
    res = describer.process(img)
    if res is None:
        print("Initial process() returned None (expected, async start).")
        
    # Wait for result
    print("Waiting for result...")
    start = time.time()
    while time.time() - start < 15.0:
        # We need to call process() again or access internal state? 
        # The design of SceneDescriber.process needs to be called repeatedly to retrieve result
        res = describer.process(img)
        if res:
            print(f"\nSUCCESS! Response received:\n{res}\n")
            break
        time.sleep(0.1)
    else:
        print("\nTIMEOUT: No response within 15 seconds.")

if __name__ == "__main__":
    main()
