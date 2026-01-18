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
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lock = threading.Lock()
        self.loaded = False
        self.loading_started = False
        self.model = None
        self.tokenizer = None
        self.model_type = "unknown"
        
        # Decide model based on hardware
        self.model_id, self.model_type = self._select_model()
        logger.info(f"LocalDescriber init: Selected {self.model_id} ({self.model_type}) on {self.device}")

    def _select_model(self):
        """
        Auto-select model:
        - GPU > 4GB VRAM -> MiniCPM-V 2.0 (High Accuracy)
        - Otherwise -> Moondream2 (High Efficiency)
        """
        if self.device == "cuda":
            try:
                # Check VRAM
                vram_bytes = torch.cuda.get_device_properties(0).total_memory
                vram_gb = vram_bytes / (1024**3)
                logger.info(f"Detected GPU VRAM: {vram_gb:.2f} GB")
                
                # MiniCPM 2.0 (2.8B) needs ~6-8GB float16, or ~4GB int4/8 (if quantized).
                # We are loading float16 by default.
                # Let's say if > 6GB, try MiniCPM.
                if vram_gb >= 6.0:
                    return "openbmb/MiniCPM-V-2", "minicpm"
            except Exception as e:
                logger.warning(f"Failed to check VRAM: {e}")
        
        # Fallback to lightweight
        return "vikhyatk/moondream2", "moondream"

    def ensure_loaded(self):
        if self.loaded: return
        with self.lock:
            if self.loaded or self.loading_started: return
            self.loading_started = True
            threading.Thread(target=self._load_worker, daemon=True).start()
            logger.info("Background loading of Local VLM started...")

    def _load_worker(self):
        try:
            logger.info(f"Loading local VLM: {self.model_id}...")
            start = time.time()
            
            # Common loading args
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            # Revision pinning for security/stability
            revision = "main"
            if self.model_type == "moondream":
                 revision = "2024-08-26"
            elif self.model_type == "minicpm":
                 # Commit hash for MiniCPM-V 2.0 (Stable as of late 2024)
                 # or just use "main" if we trust it. 
                 # Let's use a specific tag/branch if known, else main is acceptable for now but explicit is better.
                 # I'll stick to 'main' but commented structure for future.
                 revision = "main"

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                trust_remote_code=True, 
                torch_dtype=dtype,
                revision=revision
            ).to(self.device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, 
                trust_remote_code=True,
                revision=revision
            )
            
            self.model.eval()
            self.loaded = True
            logger.info(f"Local VLM ({self.model_type}) loaded in {time.time() - start:.1f}s")
        except Exception as e:
            logger.error(f"Failed to load Local VLM: {e}")
            self.loading_started = False

    def analyze_image(self, frame_bgr: np.ndarray, prompt: str = "Describe the single main object in front of the camera in 2-4 words.") -> str:
        if not self.loaded:
            self.ensure_loaded()
            return None

        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            with torch.no_grad():
                if self.model_type == "moondream":
                    enc_image = self.model.encode_image(image)
                    answer = self.model.answer_question(enc_image, prompt, self.tokenizer)
                    return answer.strip()
                    
                elif self.model_type == "minicpm":
                    # MiniCPM API: model.chat(image=..., msgs=..., tokenizer=...)
                    # It uses a specific message format
                    msgs = [{'role': 'user', 'content': prompt}]
                    
                    # Some versions use model.chat, others might differ. 
                    # Standard openbmb/MiniCPM-V-2 uses .chat()
                    res = self.model.chat(
                        image=image,
                        msgs=msgs,
                        tokenizer=self.tokenizer,
                        sampling=True, 
                        temperature=0.7
                    )
                    # res is usually a string
                    return str(res).strip()
            
            return None
        except Exception as e:
            logger.error(f"Local VLM inference failed: {e}")
            return None
