import os
import time
import logging
import base64
import cv2
import numpy as np
import requests
from threading import Thread, Event, Lock
from typing import Optional, Dict

logger = logging.getLogger("walkingpal.scene")

class SceneChangeMonitor:
    def __init__(self, change_threshold: float = 15.0):
        self.change_threshold = change_threshold
        self.last_frame_small = None

    def _preprocess_for_diff(self, frame: np.ndarray) -> np.ndarray:
        """Resize and grayscale for fast diffing."""
        small = cv2.resize(frame, (64, 64))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return gray

    def detect_change(self, frame: np.ndarray) -> bool:
        """Check if scene has changed significantly."""
        if frame is None:
            return False
            
        cur_small = self._preprocess_for_diff(frame)
        
        if self.last_frame_small is None:
            self.last_frame_small = cur_small
            return True # First frame always triggers
            
        # Calculate Mean Absolute Difference
        diff = cv2.absdiff(cur_small, self.last_frame_small)
        mean_diff = np.mean(diff)
        
        is_changed = mean_diff > self.change_threshold
        
        if is_changed:
            self.last_frame_small = cur_small
            
        return is_changed

class SceneDescriber:
    def __init__(self, api_key: str, 
                 cheap_model: str = "google/gemini-2.0-flash-exp:free", 
                 expensive_model: str = "google/gemini-2.0-flash-exp:free",
                 change_threshold: float = 15.0, # Pixel intensity difference
                 cooldown_s: float = 3.0):
        
        self.api_key = api_key
        self.cheap_model = cheap_model
        self.expensive_model = expensive_model
        
        # Fallback Strategy: Primary -> Thinking -> Llama -> Qwen
        self.fallback_models = [
            cheap_model,
            "google/gemini-2.0-flash-exp:free",
            "qwen/qwen-2-vl-7b-instruct:free" 
        ]
        
        self.monitor = SceneChangeMonitor(change_threshold)
        self.cooldown_s = cooldown_s
        
        self.last_trigger_time = 0
        self.last_desc_time = 0
        
        self.current_task = None
        self.latest_result = None
        self._lock = Lock()
        
    def detect_change(self, frame: np.ndarray) -> bool:
        return self.monitor.detect_change(frame)

    def process(self, frame: np.ndarray):
        """Main entry point. Call this every frame (or every N frames)."""
        now = time.time()
        
        # 0. Check if we are already running a request
        if self.current_task is not None and self.current_task.is_alive():
            return None # Busy
            
        # 1. Check cooldown
        if (now - self.last_trigger_time) < self.cooldown_s:
            return None
            
        # 2. Check for scene change
        if self.detect_change(frame):
            logger.info("Scene Change Detected! Triggering Analysis.")
            self.last_trigger_time = now
            
            # Start conversion in main thread (fast) or worker? 
            # Better to copy frame and start worker
            frame_copy = frame.copy()
            self.current_task = Thread(target=self._analyze_scene_task, args=(frame_copy,))
            self.current_task.start()
            
        # 3. Check for results
        with self._lock:
            if self.latest_result:
                res = self.latest_result
                self.latest_result = None
                return res
        return None

    def _analyze_scene_task(self, frame: np.ndarray):
        """Worker thread for API calls."""
        try:
            # Optimize: Resize larger images to 320x240 to reduce bandwidth/latency
            # It's sufficient for scene description and much faster to upload.
            h, w = frame.shape[:2]
            if w > 320:
                frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)

            # Encode image
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            b64_image = base64.b64encode(buffer).decode('utf-8')
            
            # 1. Try Cheap Model
            description = self._call_openrouter(self.cheap_model, b64_image)
            
            # 2. Check confidence (heuristic: length, or specific keywords if we ask for confidence)
            # For now, let's assume if it returns a short/vague answer or error, we might escalate.
            # But the user said: "if the model is not confident".
            # LLMs don't always give a confidence score unless asked.
            # We can ask it to output "UNCERTAIN" if it's not sure.
            
            if self._is_uncertain(description):
                logger.info("Cheap model uncertain. Escalating to expensive model...")
                description = self._call_openrouter(self.expensive_model, b64_image)
            
            if description:
                with self._lock:
                    self.latest_result = description
                    
        except Exception as e:
            logger.error(f"Scene analysis failed: {e}")

    def _is_uncertain(self, text: str) -> bool:
        """Heuristic to check if response implies uncertainty."""
        if not text: return True
        lower = text.lower()
        unc_keywords = ["unsure", "unclear", "cannot determine", "too blurry", "difficult to see", "uncertain"]
        # Also simple check: if text is very short?
        if len(text) < 10: return True
        
        for k in unc_keywords:
            if k in lower:
                return True
        return False

    def _call_openrouter(self, initial_model: str, b64_image: str) -> Optional[str]:
        if not self.api_key:
            logger.warning("No OpenRouter API key provided.")
            return "Camera connected, but online analysis disabled."

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/pranavlal/walkpal", # Required by OpenRouter
            "X-Title": "WalkingPal"
        }
        
        # Prompt engineering
        prompt_text = (
            "You are a visual assistant for a blind person. "
            "Describe the scene briefly. Focus on safety hazards, people, and major changes. "
            "If the image is too blurry or dark, say 'UNCERTAIN'. "
            "Keep it under 20 words."
        )

        try:
            # Build list of models to try
            # Start with requested model, then unique fallbacks
            models_to_try = [initial_model]
            for m in self.fallback_models:
                if m != initial_model:
                    models_to_try.append(m)
            
            for model in models_to_try:
                data = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                                }
                            ]
                        }
                    ]
                }
                
                try:
                    resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=10)
                    
                    if resp.status_code == 200:
                        result = resp.json()
                        if 'choices' in result and len(result['choices']) > 0:
                            content = result['choices'][0]['message']['content']
                            logger.debug(f"Model {model} response: {content}")
                            return content
                    
                    # Check for non-retryable
                    if resp.status_code in (400, 401):
                        logger.error(f"Non-retryable error ({model}): {resp.status_code} - {resp.text}")
                        return None
                        
                    # Retryable (429, 5xx)
                    logger.warning(f"Model {model} failed ({resp.status_code}). Trying next fallback...")
                    
                except Exception as e:
                    logger.warning(f"Request failed for {model}: {e}. Retrying...")
                    
            logger.error("All fallback models failed.")
            return None
            
            
        except Exception as e:
            logger.error(f"Critical error in _call_openrouter: {e}")
            return None

    def analyze_navigation(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Analyze frame for specific navigation hazards in JSON format.
        Returns: {'label': 'Wet Floor Sign', 'hazard_type': 'warning'} or None
        """
        if not self.api_key:
            return None

        # Resize and encode
        h, w = frame.shape[:2]
        if w > 320:
            frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
        
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        b64_image = base64.b64encode(buffer).decode('utf-8')

        # Precise prompt for JSON
        prompt_text = (
            "Identify the single most prominent obstacle or hazard directly in the path. "
            "Return valid JSON only: {\"label\": \"<short_name>\", \"hazard_type\": \"<warning|info>\"}. "
            "Example: {\"label\": \"Wet Floor Sign\", \"hazard_type\": \"warning\"}. "
            "If path is clear, return null."
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/pranavlal/walkpal",
            "X-Title": "WalkingPal"
        }

        # Prepare list of models to try
        models_to_try = [self.cheap_model]
        for m in self.fallback_models:
            if m != self.cheap_model:
                models_to_try.append(m)

        for model in models_to_try:
            data = {
                "model": model,
                "messages": [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                        ]
                    }
                ],
                "response_format": {"type": "json_object"}
            }

            try:
                resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=8)
                if resp.status_code == 200:
                    res = resp.json()
                    if 'choices' in res and len(res['choices']) > 0:
                        content = res['choices'][0]['message']['content']
                        # Parse JSON
                        import json
                        try:
                            obj = json.loads(content)
                            if obj and 'label' in obj:
                                return obj
                        except json.JSONDecodeError:
                            logger.warning(f"Nav JSON parse failed for {model}: {content[:50]}...")
                            return None
                            
                # Check for non-retryable
                if resp.status_code in (400, 401):
                    logger.error(f"Nav non-retryable error ({model}): {resp.status_code}")
                    return None
                    
                # Retryable (429, 5xx)
                logger.warning(f"Nav model {model} failed ({resp.status_code}). Trying next fallback...")
                
            except Exception as e:
                logger.warning(f"Nav analysis error for {model}: {e}. Retrying...")
            
        return None
