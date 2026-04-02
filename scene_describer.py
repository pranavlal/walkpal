import os
import time
import logging
import base64
import json
import cv2
import numpy as np
import requests
from google import genai
from google.genai import types
from openai import OpenAI
from threading import Thread, Event, Lock
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any

logger = logging.getLogger("walkingpal.scene")

class SceneChangeMonitor:
    def __init__(self, change_threshold: float = 15.0):
        self.change_threshold = change_threshold
        self.last_frame_small = None

    def _preprocess_for_diff(self, frame: np.ndarray) -> np.ndarray:
        """Resize and grayscale for fast diffing."""
        assert frame is not None and frame.ndim == 3, "Invalid frame for preprocess"
        small = cv2.resize(frame, (64, 64))
        assert small.shape == (64, 64, 3), "Resize failed"
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
    def __init__(self, api_key: Optional[str] = None, 
                 google_api_key: Optional[str] = None,
                 openai_api_key: Optional[str] = None,
                 anthropic_api_key: Optional[str] = None,
                 local_describer: Any = None,
                 cheap_model: str = "google/gemini-2.0-flash-exp:free", 
                 expensive_model: str = "google/gemini-2.0-flash-exp:free",
                 openai_model: str = "gpt-4o-mini",
                 change_threshold: float = 15.0, # Pixel intensity difference
                 cooldown_s: float = 3.0):
        
        self.api_key = api_key # OpenRouter key
        self.google_api_key = google_api_key
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.local_describer = local_describer
        self.gemini_client = None
        
        if self.google_api_key:
            try:
                self.gemini_client = genai.Client(api_key=self.google_api_key)
                logger.info("Google Gemini Client initialized.")
            except Exception as e:
                logger.error(f"Failed to init Gemini Client: {e}")

        self.openai_client = None
        self.openai_model = openai_model
        if self.openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                logger.info(f"OpenAI Client initialized (model={self.openai_model}).")
            except Exception as e:
                logger.error(f"Failed to init OpenAI Client: {e}")

        self.cheap_model = cheap_model
        self.expensive_model = expensive_model
        
        # Fallback Strategy: Primary -> Thinking -> Llama -> Qwen
        self.fallback_models = [
            cheap_model,
            "google/gemini-2.0-flash-exp:free",
            "qwen/qwen-2-vl-7b-instruct:free" 
        ]
        
        # Cooldowns for failed APIs (Rule 07)
        self.cooldowns = {'openai': 0.0, 'gemini': 0.0, 'openrouter': 0.0}
        self.cooldown_duration = 60.0 # 1 minute skip on failure
        
        self.monitor = SceneChangeMonitor(change_threshold)
        self.cooldown_s = cooldown_s
        
        self.last_trigger_time = 0
        self.latest_result = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="SceneDescriber")
        self.current_future = None
        self._lock = Lock()
        
    def detect_change(self, frame: np.ndarray) -> bool:
        assert frame is not None, "Frame cannot be None"
        assert isinstance(frame, np.ndarray), "Frame must be a numpy array"
        return self.monitor.detect_change(frame)

    def shutdown(self):
        """Cleanly shutdown the executor."""
        if self._executor:
            logger.debug("Shutting down SceneDescriber executor...")
            self._executor.shutdown(wait=False)
            self._executor = None

    def process(self, frame: np.ndarray):
        """Main entry point. Call this every frame (or every N frames)."""
        assert frame is not None, "Frame cannot be None"
        now = time.time()
        
        # 0. Check if we are already running a request
        if self.current_future is not None and not self.current_future.done():
            return None # Busy
            
        # 1. Check cooldown
        if (now - self.last_trigger_time) < self.cooldown_s:
            return None
            
        # 2. Check for scene change
        if self.detect_change(frame):
            logger.info("Scene Change Detected! Triggering Analysis.")
            self.last_trigger_time = now
            
            # Start conversion in worker
            frame_copy = frame.copy()
            assert frame_copy.shape == frame.shape, "Frame copy failed"
            
            # Submit to executor
            if self._executor:
                self.current_future = self._executor.submit(self._analyze_scene_task, frame_copy)
                assert self.current_future is not None, "Executor submission failed"
            
        # 3. Check for results
        with self._lock:
            if self.latest_result:
                res = self.latest_result
                self.latest_result = None
                return res
        return None

    def _analyze_scene_task(self, frame: np.ndarray):
        """Worker thread for API calls with Single-Pass Encoding."""
        assert frame is not None and frame.size > 0, "Invalid frame for analysis"
        now = time.time()
        try:
            # 1. Optimize: Resize once
            h, w = frame.shape[:2]
            if w > 480:
                frame = cv2.resize(frame, (480, int(h * 480 / w)), interpolation=cv2.INTER_AREA)

            # 2. Encode Image ONCE (Rule 03)
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            b64_image = base64.b64encode(buffer).decode('utf-8')
            
            description = None
            
            # 3. Hierarchy with Adaptive Fallback
            if self.openai_api_key and now > self.cooldowns['openai']:
                description = self._call_openai(b64_image, task="describe")
                if description: 
                    logger.info("OpenAI successful.")
                else:
                    self.cooldowns['openai'] = now + self.cooldown_duration

            if not description and self.gemini_client and now > self.cooldowns['gemini']:
                description = self._call_google_gemini(b64_image, task="describe")
                if description: 
                    logger.info("Gemini successful.")
                else:
                    self.cooldowns['gemini'] = now + self.cooldown_duration
            
            if not description and self.api_key and now > self.cooldowns['openrouter']:
                description = self._call_openrouter(self.cheap_model, b64_image)
                if description: 
                    logger.info("OpenRouter successful.")
                else:
                    self.cooldowns['openrouter'] = now + self.cooldown_duration

            # 4. Fallback to Local (Tier 4)
            if not description and self.local_describer:
                logger.info("Online APIs failed. Using Local VLM for description...")
                description = self.local_describer.analyze_image(frame, prompt="Describe the scene briefly for a blind person. Focus on safety. Under 20 words.")
            
            # Validation / Escalation (Only if online succeeded but is uncertain)
            if description and self._is_uncertain(description):
                logger.info("Model uncertain. Escalating...")
                # If we had a better OpenAI model or Gemini Pro, we'd use it.
                # For now, try OpenRouter expensive model if available.
                if self.api_key:
                    desc_expensive = self._call_openrouter(self.expensive_model, b64_image)
                    if desc_expensive:
                        description = desc_expensive

            if description:
                with self._lock:
                    self.latest_result = description
                    
        except Exception as e:
            logger.error(f"Scene analysis task failed: {e}")

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

    GEMINI_MODEL_ID = "gemini-2.0-flash"

    def _call_google_gemini(self, b64_image: str, task: str = "describe") -> Optional[str]:
        """Direct call to Google Gemini 2.0 Flash."""
        assert b64_image, "Image data missing"
        if not self.gemini_client:
            return None
            
        try:
            model_id = self.GEMINI_MODEL_ID
            
            if task == "describe":
                prompt = "You are a visual assistant for a blind person. Describe the scene briefly. Focus on safety hazards, people, and major changes. If uncertain, say 'UNCERTAIN'. Keep it under 20 words."
                
                response = self.gemini_client.models.generate_content(
                    model=model_id,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_text(text=prompt),
                                types.Part.from_bytes(data=base64.b64decode(b64_image), mime_type="image/jpeg")
                            ]
                        )
                    ]
                )
                if response.text:
                    return response.text.strip()
                    
            elif task == "navigate":
                prompt = (
                    "Identify the single most prominent obstacle or hazard directly in the path. "
                    "Return valid JSON only: {\"label\": \"<short_name>\", \"hazard_type\": \"<warning|info>\"}. "
                    "Example: {\"label\": \"Wet Floor Sign\", \"hazard_type\": \"warning\"}. "
                    "If path is clear, return null."
                )
                response = self.gemini_client.models.generate_content(
                    model=model_id,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_text(text=prompt),
                                types.Part.from_bytes(data=base64.b64decode(b64_image), mime_type="image/jpeg")
                            ]
                        )
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                if response.text:
                    return response.text.strip()

        except Exception as e:
            logger.warning(f"Gemini Direct call failed ({task}): {e}")
            
        return None

    def _call_openai(self, b64_image: str, task: str = "describe") -> Optional[str]:
        """Direct call to OpenAI using the official SDK."""
        assert b64_image, "Image data missing"
        if not self.openai_client:
            return None
            
        try:
            if task == "describe":
                prompt = "You are a visual assistant for a blind person. Describe the scene briefly. Focus on safety hazards, people, and major changes. If uncertain, say 'UNCERTAIN'. Keep it under 20 words."
            else:
                prompt = (
                    "Identify the single most prominent obstacle or hazard directly in the path. "
                    "Return valid JSON only: {\"label\": \"<short_name>\", \"hazard_type\": \"<warning|info>\"}. "
                    "Example: {\"label\": \"Wet Floor Sign\", \"hazard_type\": \"warning\"}. "
                    "If path is clear, return null."
                )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                        }
                    ]
                }
            ]
            
            response_format = {"type": "text"}
            if task == "navigate":
                response_format = {"type": "json_object"}

            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                max_tokens=100,
                response_format=response_format
            )
            
            if response.choices:
                return response.choices[0].message.content.strip()
                
        except Exception as e:
            logger.error(f"OpenAI SDK Exception ({task}): {e}")
            
        return None

    def _call_anthropic(self, b64_image: str, task: str = "describe") -> Optional[str]:
        """Direct call to Anthropic Claude 3.5 Sonnet."""
        if not self.anthropic_api_key:
            return None
            
        try:
            # Model: claude-3-5-sonnet-20241022
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            system_prompt = "You are a visual assistant for a blind person."
            user_text = ""
            
            if task == "describe":
                user_text = "Describe the scene briefly. Focus on safety hazards, people, and major changes. If uncertain, say 'UNCERTAIN'. Keep it under 20 words."
            elif task == "navigate":
                user_text = (
                    "Identify the single most prominent obstacle or hazard directly in the path. "
                    "Return valid JSON only: {\"label\": \"<short_name>\", \"hazard_type\": \"<warning|info>\"}. "
                    "Example: {\"label\": \"Wet Floor Sign\", \"hazard_type\": \"warning\"}. "
                    "If path is clear, return null."
                )

            data = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 150,
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": b64_image
                                }
                            },
                            {
                                "type": "text",
                                "text": user_text
                            }
                        ]
                    }
                ]
            }
            
            resp = requests.post(url, headers=headers, json=data, timeout=10)
            
            if resp.status_code == 200:
                result = resp.json()
                if 'content' in result and len(result['content']) > 0:
                    text_content = result['content'][0]['text']
                    return text_content.strip()
            else:
                logger.warning(f"Claude API failed: {resp.status_code} - {resp.text}")
                
        except Exception as e:
            logger.error(f"Claude API Exception: {e}")
            
        return None

    def _call_openrouter(self, initial_model: str, b64_image: str) -> Optional[str]:
        if not self.api_key:
            # logger.warning("No OpenRouter API key provided.") 
            # Only warn if Gemini also failed/missing (handled by caller logic usually)
            return None

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
                    resp = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions", 
                        headers=headers, json=data, timeout=12, verify=True
                    )
                    
                    if resp.status_code == 200:
                        result = resp.json()
                        if 'choices' in result and len(result['choices']) > 0:
                            content = result['choices'][0]['message']['content']
                            # Privacy: Truncate log to avoid leaking sensitive text
                            log_content = (content[:50] + '...') if len(content) > 50 else content
                            logger.debug(f"Model {model} response: {log_content}")
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
        Hierarchy: OpenAI -> Gemini -> OpenRouter -> Local
        """
        assert frame is not None, "Frame cannot be None"
        # Resize and encode
        h, w = frame.shape[:2]
        if w > 320:
            frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
        
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        b64_image = base64.b64encode(buffer).decode('utf-8')

        # 1. OpenAI
        if self.openai_client:
             json_str = self._call_openai(b64_image, task="navigate")
             if json_str:
                 try:
                     obj = json.loads(strip_markdown_json(json_str))
                     if obj and 'label' in obj: return obj
                 except Exception: pass

        # 2. Gemini
        if self.gemini_client:
             json_str = self._call_google_gemini(b64_image, task="navigate")
             if json_str:
                 try:
                     obj = json.loads(strip_markdown_json(json_str))
                     if obj and 'label' in obj: return obj
                 except Exception: pass

        # 3. OpenRouter
        if self.api_key:
            # Precise prompt for JSON
            prompt_text = (
                "Identify the single most prominent obstacle or hazard directly in the path. "
                "Return valid JSON only: {\"label\": \"<short_name>\", \"hazard_type\": \"<warning|info>\"}. "
                "If path is clear, return null."
            )
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/pranavlal/walkpal",
                "X-Title": "WalkingPal"
            }
            models_to_try = [self.cheap_model]
            for m in self.fallback_models:
                if m != self.cheap_model: models_to_try.append(m)

            for model in models_to_try:
                data = {
                    "model": model,
                    "messages": [{"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}]}],
                    "response_format": {"type": "json_object"}
                }
                try:
                    resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=8)
                    if resp.status_code == 200:
                        res = resp.json()
                        if 'choices' in res and len(res['choices']) > 0:
                            obj = json.loads(strip_markdown_json(res['choices'][0]['message']['content']))
                            if obj and 'label' in obj: return obj
                except Exception: continue

        # 4. Local Fallback
        if self.local_describer:
            try:
                txt = self.local_describer.analyze_image(frame, prompt="Identify the main obstacle or hazard ahead in 2-3 words.")
                if txt:
                    txt = txt.strip().rstrip('.')
                    return {'label': txt}
            except Exception: pass

        return None
def strip_markdown_json(text: str) -> str:
    """Robust extractor for JSON wrapped in markdown ticks."""
    if not text: return ""
    # Look for ```json ... ``` or just ``` ... ```
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()
