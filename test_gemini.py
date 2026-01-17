import os
from dotenv import load_dotenv
from google import genai

# 1. Load variables from .env
load_dotenv()
api_key = os.getenv("gemini_api_key")

if not api_key:
    print("Error: 'gemini_api_key' not found in .env file.")
else:
    # 2. Initialize the modern client
    # This automatically detects if you are using AI Studio or Vertex AI
    client = genai.Client(api_key=api_key)

    print("Testing connection to Gemini using google-genai...")
    
    try:
        # 3. Generate content
        # Note: In the new SDK, 'gemini-2.0-flash' or 'gemini-1.5-flash' are standard
        response = client.models.generate_content(
            model='gemini-2.0-flash', 
            contents="Say 'Connection Successful!' if you can hear me."
        )
        
        print("-" * 20)
        print(response.text.strip())
        print("-" * 20)
        
    except Exception as e:
        print(f"\nOops! Something went wrong:")
        print(f"Error details: {e}")
