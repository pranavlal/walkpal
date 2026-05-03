import os
import sys
import logging
import base64
import numpy as np
import cv2
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.getcwd())

from scene_describer import SceneDescriber

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("walkingpal.test")

def create_dummy_image():
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(img, "Test", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img

def test_gemini_priority():
    print("--- Test 1: Gemini Priority ---")
    
    # Mock GenAI client
    mock_gemini = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Gemini Description"
    mock_gemini.models.generate_content.return_value = mock_response
    
    with patch('google.genai.Client', return_value=mock_gemini):
        describer = SceneDescriber(api_key="fake_or_key", google_api_key="fake_google_key")
        
        # Inject mock client if init didn't work due to real import (but we patched Client so it should work)
        describer.gemini_client = mock_gemini 
        
        # Create dummy frame
        frame = create_dummy_image()
        
        # We need to bypass the thread/cooldown for testing or just test the method directly
        # Let's test _analyze_scene_task logic indirectly or just call the methods
        
        # Prepare b64
        _, buffer = cv2.imencode('.jpg', frame)
        b64 = base64.b64encode(buffer).decode('utf-8')
        
        res = describer._call_google_gemini(b64)
        print(f"Gemini Result: {res}")
        
        if res == "Gemini Description":
            print("PASS: Gemini was called and returned result.")
        else:
            print(f"FAIL: Expected 'Gemini Description', got {res}")

def test_fallback_to_openrouter():
    print("\n--- Test 2: Fallback to OpenRouter ---")
    
    # Mock GenAI to fail
    mock_gemini = MagicMock()
    mock_gemini.models.generate_content.side_effect = Exception("Gemini Down")
    
    with patch('google.genai.Client', return_value=mock_gemini):
        with patch('requests.post') as mock_post:
            # Mock OpenRouter response
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                'choices': [{'message': {'content': 'OpenRouter Description'}}]
            }
            mock_post.return_value = mock_resp
            
            describer = SceneDescriber(api_key="fake_or_key", google_api_key="fake_google_key")
            describer.gemini_client = mock_gemini
            
            # Manually trigger the logic that is inside _analyze_scene_task
            frame = create_dummy_image()
            _, buffer = cv2.imencode('.jpg', frame)
            b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Simulate what happens in _analyze_scene_task
            # 1. Gemini
            res = describer._call_google_gemini(b64)
            print(f"Gemini Result (Expected None): {res}")
            
            # 2. OpenRouter
            if not res:
                res = describer._call_openrouter("model", b64)
                print(f"OpenRouter Result: {res}")
                
            if res == "OpenRouter Description":
                print("PASS: Fallback to OpenRouter successful.")
            else:
                print(f"FAIL: Expected 'OpenRouter Description', got {res}")

if __name__ == "__main__":
    test_gemini_priority()
    test_fallback_to_openrouter()
