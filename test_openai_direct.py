import os
import cv2
import numpy as np
import logging
from dotenv import load_dotenv
from scene_describer import SceneDescriber

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_openai")

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        logger.error("OPENAI_API_KEY not found in .env")
        return

    logger.info(f"Key found: {api_key[:8]}...")

    # Test with default model
    describer = SceneDescriber(openai_api_key=api_key)
    
    # Create a dummy image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "TEST IMAGE", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Manually trigger the OpenAI call
    # 1. Encode image
    _, buffer = cv2.imencode('.jpg', img)
    import base64
    b64_image = base64.b64encode(buffer).decode('utf-8')
    
    logger.info("Testing _call_openai (describe)...")
    res = describer._call_openai(b64_image, task="describe")
    print(f"\nOpenAI Description Result: {res}\n")

    logger.info("Testing _call_openai (navigate)...")
    res_nav = describer._call_openai(b64_image, task="navigate")
    print(f"\nOpenAI Navigation Result (JSON): {res_nav}\n")

if __name__ == "__main__":
    main()
