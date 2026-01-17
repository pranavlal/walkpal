import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import threading
import logging
import time
import numpy as np
import cv2

logger = logging.getLogger("walkingpal.local_vlm")

class LocalDescriber:
    def __init__(self, model_id="vikhyatk/moondream2", device=None):
        self.model_id = model_id
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = None
        self.tokenizer = None
        self.lock = threading.Lock()
        self.loaded = False
        self.loading_started = False
        
        logger.info(f"LocalDescriber initialized. Device: {self.device} (Model not loaded yet)")

    def ensure_loaded(self):
        """Lazy load the model in background to avoid blocking main thread."""
        if self.loaded:
            return

        with self.lock:
            # Check again
            if self.loaded or self.loading_started:
                return
            
            self.loading_started = True
            threading.Thread(target=self._load_worker, daemon=True).start()
            logger.info("Background loading of Local VLM started...")

    def _load_worker(self):
        try:
            logger.info(f"Loading local VLM: {self.model_id}...")
            start = time.time()
            
            # Moondream requires trust_remote_code=True
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                trust_remote_code=True,
                revision="2024-08-26",
                torch_dtype=torch.float16
            ).to(self.device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision="2024-08-26")
            
            self.model.eval()
            self.loaded = True
            logger.info(f"Local VLM loaded in {time.time() - start:.1f}s")
        except Exception as e:
            logger.error(f"Failed to load Local VLM: {e}")
            self.loading_started = False # Allow retry

    def analyze_image(self, frame_bgr: np.ndarray, prompt: str = "Describe the single main object in front of the camera in 2-4 words.") -> str:
        """
        Run inference on a BGR OpenCV frame.
        """
        if not self.loaded:
            self.ensure_loaded()
            return None # Fallback to next tier while loading

        try:
            # Convert BGR -> RGB -> PIL
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            enc_image = self.model.encode_image(image)
            
            # Moondream specific generation
            # It usually uses .answer_question(enc_image, prompt, tokenizer)
            # or .generate()
            
            # Checking huggingface model card usage:
            # answer = model.answer_question(enc_image, "Describe this image.", tokenizer)
            
            with torch.no_grad():
                answer = self.model.answer_question(enc_image, prompt, self.tokenizer)
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Local VLM inference failed: {e}")
            return None
