#!/usr/bin/env python3
"""
WalkingPal - OAK-D Blind Navigation Assistant
DepthAI v3 (depthai>=3.2.1), OAK-D camera

Features:
- Depth-first navigation: obstacles (L/C/R), drop-offs, stairs, potholes
- Optional YOLOv6-nano hazards on CAM_A
- Optional HIGH-RES OCR from CAM_A
- Positional spatial audio (stereo panning)
- Internationalization (English, Hindi)

Safety: not safety-certified. Depth can fail on black/reflective/glass surfaces.
"""

from __future__ import annotations

import argparse
import time
import threading
import queue
import signal
import logging
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from typing import Optional, List, Tuple, Deque, Dict, Any, Callable
from collections import deque
import re
import sys
import uuid
import pygame
try:
    import pygame._sdl2.audio as sdl2_audio
except ImportError:
    sdl2_audio = None

import numpy as np
import cv2
import math
import depthai as dai
import yaml
from depth_processor import DepthProcessor

def load_config(config_path: str) -> Dict[str, Any]:
    """Load config from yaml file if exists, else return empty."""
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to load config file: {e}")
        return {}

def flatten_config(cfg: Dict[str, Any], prefix='') -> Dict[str, Any]:
    """Flatten nested config dict to match cli args."""
    out = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            out.update(flatten_config(v, prefix))
        else:
            # Simple mapping attempt:
            # We assume config keys map to arg names directly if unique, 
            # but hierarchy is for readability.
            # actually, my cli args are flat.
            # let's just map leaf keys to arg names for simplicity 
            # and allow specific overrides.
            out[k] = v
    return out


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians


# Global logger
logger = logging.getLogger("walkingpal")


# -----------------------------
# Logging Setup
# -----------------------------
def setup_logging(debug: bool = False, log_file: Optional[str] = None):
    """Configure structured logging with optional file output."""
    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%H:%M:%S"
    
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
        )
        handlers.append(file_handler)
    
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)
    logger.setLevel(level)


# -----------------------------
# Internationalization (i18n)
# -----------------------------
MESSAGES: Dict[str, Dict[str, str]] = {
    'en': {
        'nav_started': 'Navigation started.',
        'nav_stopped': 'Navigation stopped.',
        'clear': 'Clear ahead.',
        'uncertain': 'Uncertain path. Slow down.',
        'stop': 'Stop. No safe path.',
        'dropoff': 'Warning. Drop off ahead.',
        'pothole': 'Warning. Pothole or uneven ground ahead.',
        'stairs_up': 'Stairs going up ahead.',
        'stairs_down': 'Stairs going down ahead.',
        'stairs': 'Stairs ahead.',
        'obstacle_left': 'Obstacle on left.',
        'obstacle_center': 'Obstacle ahead.',
        'obstacle_right': 'Obstacle on right.',
        'go_left': 'Go left.',
        'go_right': 'Go right.',
        'go_left_or_right': 'Go left or right.',
        'go_left_center_right': 'Clear. You can go left, center, or right.',
        'obstacle_ahead_go': 'Obstacle {} ahead. You can go {} or {}.',
        'obstacle_go': 'Obstacle {} ahead. Go {}.',
        'hazard': 'Warning. {} ahead.',
        'sign_reads': 'Sign reads: {}',
        'self_test': 'System check.',
        'self_test_pass': 'Self test passed.',
        'device_disconnect': 'Device disconnected. Reconnecting.',
        'device_reconnect_fail': 'Failed to reconnect. Shutting down.',
        'low_light': 'Warning. Too dark for navigation.',
        # Direction helpers
        'dir_left': 'left',
        'dir_center': 'center',

        'dir_right': 'right',
        'step': 'step',
        'steps': 'steps',
    },
    'hi': {
        'nav_started': 'नेविगेशन शुरू।',
        'nav_stopped': 'नेविगेशन बंद।',
        'clear': 'रास्ता साफ है।',
        'uncertain': 'अनिश्चित रास्ता। धीरे चलें।',
        'stop': 'रुकें। कोई सुरक्षित रास्ता नहीं।',
        'dropoff': 'चेतावनी। आगे गड्ढा।',
        'pothole': 'चेतावनी। आगे गड्ढा या असमान जमीन।',
        'stairs_up': 'आगे सीढ़ियाँ ऊपर जा रही हैं।',
        'stairs_down': 'आगे सीढ़ियाँ नीचे जा रही हैं।',
        'stairs': 'आगे सीढ़ियाँ।',
        'obstacle_left': 'बाईं ओर बाधा।',
        'obstacle_center': 'आगे बाधा।',
        'obstacle_right': 'दाईं ओर बाधा।',
        'go_left': 'बाएं जाएं।',
        'go_right': 'दाएं जाएं।',
        'go_left_or_right': 'बाएं या दाएं जाएं।',
        'go_left_center_right': 'रास्ता साफ है। आप बाएं, बीच में या दाएं जा सकते हैं।',
        'obstacle_ahead_go': 'आगे {} बाधा है। आप {} या {} जा सकते हैं।',
        'obstacle_go': 'आगे {} बाधा है। {} जाएं।',
        'hazard': 'चेतावनी। आगे {}।',
        'sign_reads': 'साइन पढ़ता है: {}',
        'self_test': 'सिस्टम जांच।',
        'self_test_pass': 'स्व परीक्षण पास।',
        'device_disconnect': 'डिवाइस डिस्कनेक्ट। पुनः कनेक्ट हो रहा है।',
        'device_reconnect_fail': 'पुनः कनेक्ट विफल। बंद हो रहा है।',
        'low_light': 'चेतावनी। नेविगेशन के लिए बहुत अंधेरा है।',
        # Direction helpers
        'dir_left': 'बाएं',
        'dir_center': 'सीधा',
        'dir_right': 'दाएं',
        'step': 'कदम',
        'steps': 'कदम',
    }
}

_current_lang: str = 'en'

def set_language(lang: str):
    """Set the current language for messages."""
    global _current_lang
    _current_lang = lang if lang in MESSAGES else 'en'

def i18n(key: str, *args) -> str:
    """Get localized message, with optional format arguments."""
    template = MESSAGES.get(_current_lang, MESSAGES['en']).get(key, key)
    if args:
        return template.format(*args)
    return template


# -----------------------------
# Signal Handling
# -----------------------------
_shutdown_requested = threading.Event()

def signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    logger.info("Received %s, initiating shutdown...", sig_name)
    _shutdown_requested.set()
    raise KeyboardInterrupt


def install_signal_handlers():
    """Install handlers for graceful shutdown signals."""
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    logger.debug("Signal handlers installed.")


# -----------------------------
# Watchdog Timer
# -----------------------------
class Watchdog:
    """Detects if main loop stalls beyond timeout."""
    
    def __init__(self, timeout_s: float = 5.0, callback: Optional[Callable] = None):
        self.timeout = timeout_s
        self.callback = callback or self._default_callback
        self._timer: Optional[threading.Timer] = None
        self._active = False
    
    def _default_callback(self):
        logger.error("WATCHDOG: Main loop stalled for %.1fs!", self.timeout)
    
    def start(self):
        """Start the watchdog."""
        self._active = True
        self.reset()
    
    def reset(self):
        """Reset the watchdog timer (call this in main loop)."""
        if not self._active:
            return
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(self.timeout, self.callback)
        self._timer.daemon = True
        self._timer.start()
    
    def stop(self):
        """Stop the watchdog."""
        self._active = False
        if self._timer:
            self._timer.cancel()
            self._timer = None


# -----------------------------
# Text sanitization for TTS
# -----------------------------
def sanitize_tts_text(text: str) -> str:
    """Remove control chars and limit length for safe TTS."""
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    return text[:500]  # Prevent excessively long speech


# -----------------------------
# Offline TTS (Windows + Linux) - Async
# -----------------------------
import tempfile
import os
import uuid

# -----------------------------
# Unified Audio Controller (Positional TTS + Tones)
# -----------------------------
import shutil

class AudioController:
    """
    Handles both TTS and Warning Tones using pygame mixer for spatial (panned) audio.
    TTS is generated to a temp file and played as a Sound object.
    
    Processing is asynchronous to prevent blocking the main navigation loop.
    """
    
    TONES = {
        'dropoff': {'freq': 180, 'duration': 0.4, 'pulses': 2},
        'obstacle': {'freq': 600, 'duration': 0.15, 'pulses': 3},
        'pothole': {'freq': 280, 'duration': 0.3, 'pulses': 2},
        'stairs_up': {'freq': 400, 'duration': 0.5, 'pulses': 1, 'sweep': 1.5},
        'stairs_down': {'freq': 600, 'duration': 0.5, 'pulses': 1, 'sweep': 0.67},
        'stairs': {'freq': 500, 'duration': 0.3, 'pulses': 2},
        'hazard': {'freq': 800, 'duration': 0.1, 'pulses': 4},
    }

    def __init__(self, tts_rate: int = 175, volume: float = 1.0):
        self.enabled = True
        self.volume = max(0.0, min(1.0, volume))
        self._pygame = None
        self._tts_engine = None
        self._sounds: Dict[str, Any] = {}
        self._temp_dir = tempfile.mkdtemp(prefix="walkpal_tts_")
        
        # Async setup
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = None
        
        # Init Pygame Mixer
        try:
            import pygame
            # standard 44.1kHz stereo
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
            self._pygame = pygame
            self._generate_tones()
            logger.info(f"AudioController: Pygame mixer initialized. Config: {self._pygame.mixer.get_init()}")

            if sdl2_audio:
                try:
                    devices = sdl2_audio.get_audio_device_names(False)
                    logger.info(f"Available Audio Devices: {devices}")
                except Exception as e:
                    logger.error(f"Failed to list audio devices: {e}")
            else:
                logger.info("pygame._sdl2.audio not available, cannot list devices.")
        except Exception as e:
            logger.error("AudioController: Pygame init failed: %s", e)
            self.enabled = False

        # Init TTS Engine (offline)
        if self.enabled:
            pass # Lazy init in worker or just checks here? 
            # safe approach: init engine here, but only run loop in worker if needed.
            pass

        # Start worker
        self._thread = threading.Thread(target=self._worker, args=(tts_rate,), daemon=True)
        self._thread.start()

    def _worker(self, tts_rate):
        """Background thread to handle TTS generation and playback."""
        # Fix for Windows COM threading (pyttsx3/sapi5)
        # Ensure platform is imported if not available in scope (it should be global, but let's be safe)
        import platform # Local import to be safe against shadowing
        if platform.system().lower().startswith("win"):
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except ImportError:
                logger.warning("AudioController: pythoncom not found, TTS might fail on Windows.")
            except Exception as e:
                logger.warning("AudioController: CoInitialize failed: %s", e)

        # Engine will be initialized per-request to avoid SAPI5 state issues
        pass
        
        logger.info("AudioController: Worker started.")
        while not self._stop_event.is_set():
            try:
                # Get request: (type, data, pan)
                # type='speak' -> data=text
                # type='tone' -> data=name
                item = self._queue.get(timeout=0.1)
                
                if item is None: break # Poison pill
                
                cmd, data, pan = item
                
                if cmd == 'tone':
                    self._play_tone_sync(data, pan)
                elif cmd == 'speak':
                    self._speak_sync(tts_rate, data, pan)
                
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("AudioController: Worker error: %s", e)
        
        # Cleanup engine
        # Cleanup

        
        if platform.system().lower().startswith("win"):
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except:
                pass

    def _generate_tones(self):
        """Pre-generate warning tones as stereo sounds."""
        if not self._pygame:
            return
        
        sample_rate = 44100
        for name, cfg in self.TONES.items():
            freq = cfg['freq']
            duration = cfg['duration']
            pulses = cfg.get('pulses', 1)
            sweep = cfg.get('sweep', 1.0)
            
            total_samples = int(sample_rate * duration * pulses)
            t = np.linspace(0, duration * pulses, total_samples)
            
            if sweep != 1.0:
                freqs = np.linspace(freq, freq * sweep, total_samples)
                wave = np.sin(2 * np.pi * freqs * t / pulses)
            else:
                pulse_len = int(sample_rate * duration * 0.7)
                gap_len = int(sample_rate * duration * 0.3)
                single_pulse = np.concatenate([
                    np.sin(2 * np.pi * freq * np.linspace(0, duration * 0.7, pulse_len)),
                    np.zeros(gap_len)
                ])
                wave = np.tile(single_pulse, pulses)[:total_samples]
            
            # Envelope
            fade = int(sample_rate * 0.01)
            envelope = np.ones_like(wave)
            if len(envelope) > 2 * fade:
                envelope[:fade] = np.linspace(0, 1, fade)
                envelope[-fade:] = np.linspace(1, 0, fade)
            wave = wave * envelope
            
            # 16-bit stereo
            wave_int = (wave * 32767 * self.volume).astype(np.int16)
            stereo = np.column_stack((wave_int, wave_int))
            self._sounds[name] = self._pygame.sndarray.make_sound(stereo)

    def play_tone(self, name: str, pan: float = 0.0):
        """Queue a tone playback."""
        if self.enabled:
            # Tones are fast, we could play directly, but queuing preserves order with speech
            # self._play_tone_sync(name, pan) 
            # Actually, playing directly is better for latency of warnings. 
            # But let's verify if mixing is thread safe (pygame mixer usually is).
            # We'll queue it if we want strict ordering, but for warnings, invalidating old speech is good.
            # Let's direct play for latency.
            self._play_tone_sync(name, pan)

    def speak(self, text: str, pan: float = 0.0):
        """Queue a TTS request."""
        if self.enabled and text:
            # Drain queue of old speech if it's backing up?
            # For navigation, latest is greatest.
            if self._queue.qsize() > 2:
                try:
                    while not self._queue.empty():
                        self._queue.get_nowait()
                        self._queue.task_done()
                except queue.Empty:
                    pass
            self._queue.put(('speak', text, pan))

    def _play_tone_sync(self, name, pan):
        if not self._pygame: return
        sound = self._sounds.get(name)
        if sound:
            self._play_sound(sound, pan)

    def _speak_sync(self, tts_rate, text, pan):
        if not self._pygame: return
        
        fname = os.path.join(self._temp_dir, f"{uuid.uuid4().hex}.wav")
        engine = None
        try:
            import pyttsx3
            # Initialize engine for this request
            engine = pyttsx3.init()
            engine.setProperty("rate", int(tts_rate))
            engine.setProperty("volume", self.volume)
            
            # Generate WAV
            engine.save_to_file(text, fname)
            engine.runAndWait()
        except Exception as e:
            logger.error(f"TTS generation error: {e}")
        finally:
            if engine:
                try:
                    engine.stop()
                    del engine
                except: pass

        if os.path.exists(fname):
            fsize = os.path.getsize(fname)
            logger.debug(f"Audio saved to {fname} (Size: {fsize} bytes)")
            
            if fsize < 100:
                logger.warning(f"Audio file is suspiciously small ({fsize} bytes). SAPI might have failed silently.")

            try:
                sound = self._pygame.mixer.Sound(fname)
                duration = sound.get_length()
                logger.debug(f"Loaded sound. Duration: {duration:.2f}s")
                self._play_sound(sound, pan)
            except Exception as e:
                logger.error(f"Pygame sound play error: {e}")
            
            # Cleanup file
            try:
                if os.path.exists(fname):
                    os.unlink(fname)
            except Exception:
                pass


    def _play_sound(self, sound, pan: float):
        try:
            channel = sound.play()
            if channel:
                pan = max(-1.0, min(1.0, pan))
                left = 1.0 - max(0.0, pan)
                right = 1.0 + min(0.0, pan)
                channel.set_volume(left * self.volume, right * self.volume)
                logger.debug(f"Setting volume: L={left * self.volume:.2f} R={right * self.volume:.2f} (Master: {self.volume})")
        except Exception as e:
            logger.error("Sound Play Error: %s", e)

    def shutdown(self):
        self._stop_event.set()
        if self._thread:
            self._queue.put(None) # wake up
            self._thread.join(timeout=2.0)
            
        if self._pygame:
            self._pygame.mixer.quit()
        
        # Robust cleanup
        try:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception:
            pass


# -----------------------------
# Depth ROI stats + helpers
# -----------------------------
@dataclass
class RoiStats:
    valid_ratio: float
    near_mm: int
    median_mm: int
    far_mm: int


def roi_stats(depth_mm: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> RoiStats:
    roi = depth_mm[y0:y1, x0:x1]
    if roi.size == 0:
        return RoiStats(0.0, 0, 0, 0)

    vals = roi.reshape(-1)
    valid = vals[vals > 0]
    valid_ratio = float(valid.size) / float(vals.size)

    if valid.size < 80:
        return RoiStats(valid_ratio, 0, 0, 0)

    near = int(np.percentile(valid, 10))
    med = int(np.median(valid))
    far = int(np.percentile(valid, 90))
    return RoiStats(valid_ratio, near, med, far)


def classify_light(brightness: Optional[float]) -> str:
    if brightness is None:
        return "unknown"
    b = float(brightness)
    if b < 35:
        return "dark"
    if b < 75:
        return "dim"
    return "normal"


def direction_text(free_left: bool, free_center: bool, free_right: bool, dist_str: str = "") -> str:
    opts = []
    if free_left:
        opts.append(i18n('dir_left'))
    if free_center:
        opts.append(i18n('dir_center'))
    if free_right:
        opts.append(i18n('dir_right'))

    if not opts:
        return i18n('stop')
    if len(opts) == 3:
        # Distance usually not relevant if all clear, but this path is reached via 'blocked' check?
        # If blocked was True but all free?? Logic gap. 
        # If 'blocked' is true, at least one SHOULD be blocked.
        # But if free_left/center/right check considers 'clear_mm' (2m) and blocked considers 'obstacle_mm' (1.2m),
        # there is a zone 1.2-2.0m where it is NOT free but NOT blocked?
        # Actually: free logic: near > clear_mm (2000). Blocked logic: near < obstacle_mm (1200).
        # Between 1.2 and 2.0, it is NEITHER free NOR blocked. 
        # Returns 'uncertain' in main loop if not reliable_any.
        return i18n('go_left_center_right')
    
    if len(opts) == 2:
        return i18n('obstacle_ahead_go', dist_str, opts[0], opts[1])
    return i18n('obstacle_go', dist_str, opts[0])



def detect_dropoff(depth_mm: np.ndarray, roi_cache: Dict[str, Any], 
                   base_dropoff_mm: int, dropoff_invalid_ratio: float, 
                   min_valid: float, require_center_for_clear: bool,
                   stB: RoiStats, stL: RoiStats, stC: RoiStats, stR: RoiStats,
                   pitch_deg: float = 0.0, camera_height_m: float = 1.5) -> bool:
    
    # Use pre-calculated stats (stB, stL, stC, stR)
    invalid_frac = 1.0 - stB.valid_ratio

    # Path visibility check
    # If the path ahead is clearly visible, we should be careful about declaring a dropoff 
    # based solely on invalid data (which could be reflections).
    path_visible = (stC.valid_ratio > 0.15) or (stL.valid_ratio > 0.15) or (stR.valid_ratio > 0.15)
    
    # Dynamic Drop-off Threshold Calculation
    # Calculate the expected distance to the ground at the center of the bottom ROI
    # ROI center Y angle (approximate).
    # Bottom ROI is ~0.88 * H. Center is 0.5 * H. 
    # VFOV ~50 deg. 
    # rel_y = (0.88 - 0.5) * 2 = 0.76 (normalized -1..1 scale relative to center)
    # angle_offset = 0.76 * (50/2) = 19 degrees down from optical axis
    
    angle_down_deg = pitch_deg + 19.0
    
    # If angle is too shallow (looking level or up), geometry says ground is FAR.
    # We should relax the threshold to avoid false positives.
    if angle_down_deg < 5.0:
        # Looking horizontal or up. Ground is effectively at infinity or very far.
        # We cannot reliability detect a dropoff by distance here.
        # Use a very high threshold (only invalid data triggers, or deep pits)
        adaptive_threshold_mm = max(base_dropoff_mm, 6000)
    else:
        # Expected ground distance
        expected_mm = (camera_height_m * 1000.0) / math.tan(math.radians(angle_down_deg))
        # Allow some tolerance (e.g. 50% further) before calling it a dropoff
        adaptive_threshold_mm = max(base_dropoff_mm, int(expected_mm * 1.5))

    is_too_deep = (stB.median_mm != 0 and stB.median_mm > adaptive_threshold_mm)
    
    # For invalid ratio: if path is visible (looking ahead), high invalidity at feet 
    # might be reflections (shiny floor). We increase tolerance if path is visible.
    effective_invalid_ratio = dropoff_invalid_ratio + 0.1 if path_visible else dropoff_invalid_ratio
    is_invalid = (invalid_frac >= effective_invalid_ratio)

    potentially_dropoff = is_invalid or is_too_deep
    
    if potentially_dropoff:
         if path_visible:
             return True
         else:
             # Dark room or obscured sensor. Be conservative.
             return False
    return False

def detect_stairs(depth_mm: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> Optional[str]:
    h = y1 - y0
    if h < 100:
        return None

    bins = 7
    ys = np.linspace(y0, y1, bins + 1).astype(int)
    medians: List[int] = []
    valids: List[float] = []
    flatnessS: List[bool] = [] # Is the bin "flat" (like a step tread) vs sloped?

    for i in range(bins):
        st = roi_stats(depth_mm, x0, ys[i], x1, ys[i + 1])
        medians.append(st.median_mm)
        valids.append(st.valid_ratio)
        
        # Ramp Detection: A ramp has high variance/spread in depth within the bin.
        # A stair tread should be relatively flat (low spread).
        # st.far_mm - st.near_mm
        if st.valid_ratio > 0.1 and st.near_mm > 0:
            spread = st.far_mm - st.near_mm
            # If spread is huge, it's likely a slope, not a flat step.
            # Reduced to 200mm to robustly filter ramps
            flatnessS.append(spread < 200) 
        else:
            flatnessS.append(True) # inconclusive, assume OK

    if float(np.mean(valids)) < 0.35:
        return None

    diffs = []
    for a, b in zip(medians[:-1], medians[1:]):
        if a == 0 or b == 0:
            diffs.append(0)
        else:
            diffs.append(b - a)

    step_mm = 160
    
    # Filter for significant jumps
    jumps = [d for d in diffs if abs(d) >= step_mm]
    
    if len(jumps) < 2:
        return None 

    # Check consistency
    pos_jumps = sum(1 for j in jumps if j > 0)
    neg_jumps = sum(1 for j in jumps if j < 0)
    
    # Flatness check: majority of bins should be "flat-ish" if they are steps?
    # Actually, the jumps happen BETWEEN bins. The bins themselves might capture the flat part.
    # If we are looking at a ramp, EVERY bin is sloped.
    flat_bins = sum(1 for f in flatnessS if f)
    if flat_bins < (bins // 2):
        return None # Too sloped throughout (Ramp)

    if pos_jumps > 0 and neg_jumps > 0:
        return None

    if medians[0] == 0 or medians[-1] == 0:
        return "stairs" 

    overall = medians[-1] - medians[0]
    
    if overall <= -250:
        return "stairs_up"
    if overall >= 250:
        return "stairs_down"
    return "stairs"

# -----------------------------
# Jitter reduction helpers
# -----------------------------
class DebouncedBool:
    def __init__(self, on_count: int = 3, off_count: int = 2):
        self.on_count = max(1, int(on_count))
        self.off_count = max(1, int(off_count))
        self._true_run = 0
        self._false_run = 0
        self.state = False

    def update(self, cond: bool) -> bool:
        if cond:
            self._true_run += 1
            self._false_run = 0
        else:
            self._false_run += 1
            self._true_run = 0

        if not self.state and self._true_run >= self.on_count:
            self.state = True
            self._true_run = 0
        elif self.state and self._false_run >= self.off_count:
            self.state = False
            self._false_run = 0
        return self.state


class MajorityLabel:
    def __init__(self, k: int = 5):
        self.k = max(1, int(k))
        self.buf: Deque[str] = deque(maxlen=self.k)

    def update(self, label: str) -> str:
        self.buf.append(label)
        counts: Dict[str, int] = {}
        for x in self.buf:
            counts[x] = counts.get(x, 0) + 1
        latest = self.buf[-1]
        best = max(counts.items(), key=lambda kv: (kv[1], 1 if kv[0] == latest else 0))[0]
        return best


# -----------------------------
# Pothole detection (depth-only)
# -----------------------------
def pothole_score(depth_mm: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    roi = depth_mm[y0:y1, x0:x1].astype(np.float32)
    if roi.size == 0:
        return 0.0
    valid = roi > 0
    if valid.mean() < 0.35:
        return 0.0

    vals = roi[valid]
    if vals.size < 200:
        return 0.0

    med = np.median(vals)
    norm = roi.copy()
    norm[~valid] = med
    norm = norm / max(1.0, med)

    gx = np.abs(norm[:, 1:] - norm[:, :-1])
    gy = np.abs(norm[1:, :] - norm[:-1, :])
    rough = float(np.mean(gx) + np.mean(gy))

    dip_frac = float(np.mean((roi > 0) & (roi > (med * 1.18))))

    score = 0.0
    score += min(1.0, rough * 8.0) * 0.6
    score += min(1.0, dip_frac * 3.0) * 0.4
    return max(0.0, min(1.0, score))


# -----------------------------
# Software Tilt Estimation
# -----------------------------
def estimate_pitch_from_depth(depth_mm: np.ndarray, camera_height_m: float = 1.5, vfov_deg: float = 50.0) -> float:
    """
    Estimate camera pitch (tilt) based on the assumption of a flat ground plane.
    Returns pitch in degrees. Positive = Looking down, Negative = Looking up.
    
    Formula: theta = arctan(H / Z) - alpha_pixel
    """
    h, w = depth_mm.shape
    # Look at the bottom 20% of the image where the ground is most likely to be
    y0 = int(h * 0.80)
    y1 = h
    
    roi = depth_mm[y0:y1, :]
    valid = roi > 0
    if valid.sum() < 100:
        return 0.0 # No data, assume level
        
    # Get median depth of the bottom area
    z_mm = np.median(roi[valid])
    if z_mm < 100: 
        return 0.0
        
    z_m = z_mm / 1000.0
    
    # Angle of the center of this ROI relative to optical axis
    # The ROI center is at y = (y0 + y1) / 2
    # Optical center is h / 2
    y_center_roi = (y0 + y1) / 2
    y_center_img = h / 2
    
    # Pixel offset from center (normalized -1 to 1)
    # y=0(top) -> -1, y=h(bottom) -> 1
    # But wait, standard pinhole:
    # y_angle = (y - cy) / fy ... 
    # Let's use VFOV linear approximation for simplicity
    
    dy_norm = (y_center_roi - y_center_img) / (h / 2) # 0 to 1 for bottom half
    
    half_vfov_rad = math.radians(vfov_deg / 2.0)
    alpha_pixel_rad = dy_norm * half_vfov_rad
    
    # theta = arctan(H / Z) - alpha
    # angle_to_ground = arctan(H / Z)
    angle_to_ground_rad = math.atan2(camera_height_m, z_m)
    
    pitch_rad = angle_to_ground_rad - alpha_pixel_rad
    return float(math.degrees(pitch_rad))



# -----------------------------
# OCR engines (Tesseract / EasyOCR / Auto)
# -----------------------------
def _text_quality_score(text: str) -> float:
    """
    Heuristic quality score for OCR output: 0..1
    Higher is better.
    """
    t = " ".join(text.strip().split())
    if not t:
        return 0.0
    # Remove obvious junk-only
    alnum = sum(ch.isalnum() for ch in t)
    ratio = alnum / max(1, len(t))
    # Reward length a bit, but cap it
    length_score = min(1.0, len(t) / 20.0)
    # Penalize if mostly symbols
    return max(0.0, min(1.0, 0.65 * ratio + 0.35 * length_score))


def _clean_text(text: str) -> str:
    t = " ".join(text.split())
    # Drop repeated weird punctuation runs
    t = re.sub(r"[_=~\-]{4,}", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def try_init_tesseract(enable: bool):
    if not enable:
        return None
    try:
        import pytesseract  # type: ignore
        return pytesseract
    except Exception as e:
        logger.warning("pytesseract not available (%s). Tesseract OCR disabled.", e)
        return None


class EasyOCRWrapper:
    """
    Lazy EasyOCR loader.
    We only create Reader when needed (auto-fallback or forced easyocr).
    """
    def __init__(self, use_gpu: bool = False):
        self.reader = None
        self.available = False
        self._err = None
        self._use_gpu = use_gpu

    def ensure_reader(self, langs: List[str]):
        if self.reader is not None:
            return True
        try:
            import easyocr  # type: ignore
            self.reader = easyocr.Reader(langs, gpu=self._use_gpu)
            self.available = True
            return True
        except Exception as e:
            self._err = e
            self.available = False
            self.reader = None
            return False

    def read(self, bgr: np.ndarray) -> str:
        if self.reader is None:
            return ""
        # easyocr expects RGB array
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = self.reader.readtext(rgb, detail=0, paragraph=True)
        if not results:
            return ""
        # results could be a list of strings
        if isinstance(results, list):
            return " ".join(str(x) for x in results if x)
        return str(results)


def tesseract_ocr(pytesseract_mod, bgr: np.ndarray, tesseract_lang: str) -> str:
    if pytesseract_mod is None:
        return ""
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 60, 60)
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 8)
        txt = pytesseract_mod.image_to_string(thr, lang=tesseract_lang, config="--oem 1 --psm 6")
        return _clean_text(txt)
    except Exception:
        return ""


def map_tesseract_lang_to_easyocr_langs(tesseract_lang: str) -> List[str]:
    """
    Map 'eng' / 'eng+hin' to EasyOCR codes ['en'] / ['en','hi'].
    """
    parts = [p.strip() for p in tesseract_lang.split("+") if p.strip()]
    out = []
    for p in parts:
        if p == "eng":
            out.append("en")
        elif p == "hin":
            out.append("hi")
    # Always ensure english if user gave something odd
    if not out:
        out = ["en"]
    # Deduplicate
    return list(dict.fromkeys(out))


def run_ocr_auto(
    bgr: np.ndarray,
    ocr_engine: str,
    tesseract_lang: str,
    pytesseract_mod,
    easyocr_wrap: EasyOCRWrapper,
    min_len: int,
    auto_fallback_min_quality: float,
) -> str:
    """
    Returns text or "".
    - tesseract: use tesseract only
    - easyocr: use easyocr only (if available)
    - auto: tesseract first, then fallback to easyocr if quality is low
    """
    ocr_engine = ocr_engine.lower().strip()

    if ocr_engine == "tesseract":
        txt = tesseract_ocr(pytesseract_mod, bgr, tesseract_lang)
        return txt if len(txt) >= min_len else ""

    if ocr_engine == "easyocr":
        langs = map_tesseract_lang_to_easyocr_langs(tesseract_lang)
        if not easyocr_wrap.ensure_reader(langs):
            return ""  # forced easyocr but not installed/working
        txt = _clean_text(easyocr_wrap.read(bgr))
        return txt if len(txt) >= min_len else ""

    # auto
    txt_t = tesseract_ocr(pytesseract_mod, bgr, tesseract_lang)
    txt_t = txt_t if len(txt_t) >= min_len else ""
    q = _text_quality_score(txt_t) if txt_t else 0.0
    if txt_t and q >= auto_fallback_min_quality:
        return txt_t

    # fallback to easyocr if available
    langs = map_tesseract_lang_to_easyocr_langs(tesseract_lang)
    if not easyocr_wrap.ensure_reader(langs):
        return txt_t  # return whatever we got from tesseract
    txt_e = _clean_text(easyocr_wrap.read(bgr))
    if len(txt_e) < min_len:
        return txt_t
    # Prefer better quality
    if _text_quality_score(txt_e) >= _text_quality_score(txt_t):
        return txt_e
    return txt_t


# -----------------------------
# DepthAI v3 pipeline builder
# -----------------------------
def build_pipeline_and_queues(
    fps_depth: float,
    confidence: int,
    lr_check: bool,
    extended_disparity: bool,
    subpixel: bool,
    enable_yolo: bool,
    yolo_fps: float,
    enable_ocr: bool,
    ocr_fps: float,
    ocr_size: Tuple[int, int],
    enable_imu: bool,
):
    """
    - Stereo depth from CAM_B/C
    - Brightness preview from CAM_B
    - Optional YOLOv6-nano DetectionNetwork on CAM_A (downloads from model zoo)
    - Optional high-res CAM_A output queue for OCR (only if enable_ocr)
    - Optional IMU rotation vector
    """
    pipeline = dai.Pipeline()

    # CAM_B: stereo + brightness preview
    camL_node = pipeline.create(dai.node.Camera)
    camL = camL_node.build(dai.CameraBoardSocket.CAM_B)
    
    left_stereo = camL.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8, fps=fps_depth)
    left_preview = camL.requestOutput((320, 200), type=dai.ImgFrame.Type.GRAY8, fps=fps_depth)

    # CAM_C: stereo
    camR_node = pipeline.create(dai.node.Camera)
    camR = camR_node.build(dai.CameraBoardSocket.CAM_C)
    
    right_stereo = camR.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8, fps=fps_depth)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
    stereo.initialConfig.setConfidenceThreshold(confidence)
    stereo.setLeftRightCheck(lr_check)
    stereo.setExtendedDisparity(extended_disparity)
    stereo.setSubpixel(subpixel)
    try:
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    except AttributeError as e:
        logger.warning("MedianFilter setting not supported: %s", e)

    left_stereo.link(stereo.left)
    right_stereo.link(stereo.right)

    q_depth = stereo.depth.createOutputQueue(maxSize=2, blocking=False)
    q_preview = left_preview.createOutputQueue(maxSize=1, blocking=False)

    q_det = None
    q_yolo_rgb = None
    label_map = None
    q_ocr_rgb = None
    q_imu = None

    if enable_imu:
        try:
            imu = pipeline.create(dai.node.IMU)
            imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, 10)
            imu.setBatchReportThreshold(1)
            imu.setMaxBatchReports(10)
            q_imu = imu.out.createOutputQueue(maxSize=10, blocking=False)
            logger.info("IMU enabled in pipeline.")
        except Exception as e:
            logger.warning("Failed to enable IMU node: %s", e)
            q_imu = None

    if enable_yolo or enable_ocr:
        camA_node = pipeline.create(dai.node.Camera)
        camA = camA_node.build(dai.CameraBoardSocket.CAM_A)

        # Note: setFps not supported on Camera node. FPS is controlled via requestOutput.
        
        if enable_ocr:
            ocr_out = camA.requestOutput(
                ocr_size,
                type=dai.ImgFrame.Type.BGR888p,
                fps=float(ocr_fps),
                resizeMode=dai.ImgResizeMode.CROP,
            )
            q_ocr_rgb = ocr_out.createOutputQueue(maxSize=2, blocking=False)
        else:
            # If only YOLO is enabled, we might want to constrain FPS if possible, 
            # or let it run at default. To ensure consistency if yolo_fps is critical, 
            # we could create a dummy output, but for now we follow standard behavior.
            pass

        if enable_yolo:
            det = pipeline.create(dai.node.DetectionNetwork).build(camA, dai.NNModelDescription("yolov6-nano"))
            label_map = det.getClasses()
            q_yolo_rgb = det.passthrough.createOutputQueue(maxSize=2, blocking=False)
            q_det = det.out.createOutputQueue(maxSize=4, blocking=False)

    return pipeline, q_depth, q_preview, q_det, q_yolo_rgb, label_map, q_ocr_rgb, q_imu



def main():
    # 1. Load config file first (defaults)
    # Check for default locations
    default_config = "config.yaml"
    if not os.path.exists(default_config):
        default_config = "/etc/walkingpal/config.yaml"
    
    # Pre-parse just for config arg
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=default_config, help="Path to config.yaml")
    known_args, _ = pre_parser.parse_known_args()
    
    file_defaults = {}
    if known_args.config and os.path.exists(known_args.config):
        cfg = load_config(known_args.config)
        # Flatten and map to args
        flat = flatten_config(cfg)
        # Mapping renames if necessary.
        # Most names match (fps, confidence, etc)
        # 'camera_height' -> camera_height
        file_defaults = flat

    ap = argparse.ArgumentParser(description="OAK-D blind nav (DepthAI v3)")
    ap.add_argument("--config", type=str, default=default_config, help="Path to config.yaml")
    
    # Set defaults from file if present
    ap.set_defaults(**file_defaults)

    # Depth
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--confidence", type=int, default=200)
    ap.add_argument("--lr_check", action="store_true")
    ap.add_argument("--extended_disparity", action="store_true")
    ap.add_argument("--subpixel", action="store_true")

    # Navigation thresholds
    ap.add_argument("--obstacle_m", type=float, default=1.2)
    ap.add_argument("--clear_m", type=float, default=2.0)

    ap.add_argument("--dropoff_m", type=float, default=3.5)
    ap.add_argument("--dropoff_invalid_ratio", type=float, default=0.75)

    # Adaptation
    ap.add_argument("--min_valid_normal", type=float, default=0.20)
    ap.add_argument("--min_valid_dim", type=float, default=0.30)
    ap.add_argument("--dropoff_invalid_boost_dim", type=float, default=0.10)

    # Jitter reduction
    ap.add_argument("--confirm_frames", type=int, default=5)
    ap.add_argument("--clear_frames", type=int, default=2)
    ap.add_argument("--dir_window", type=int, default=5)

    # Potholes
    ap.add_argument("--enable_potholes", action="store_true")
    ap.add_argument("--pothole_score", type=float, default=0.55)
    ap.add_argument("--pothole_roi_valid", type=float, default=0.40)
    ap.add_argument("--pothole_cooldown_s", type=float, default=2.0)
    ap.add_argument("--camera_height", type=float, default=1.5,
                    help="Approximate camera height in meters (used for SW tilt fallback).")

    # YOLO hazards
    ap.add_argument("--enable_yolo", action="store_true")
    ap.add_argument("--yolo_fps", type=float, default=4.0)
    ap.add_argument("--yolo_conf", type=float, default=0.50)
    ap.add_argument("--hazards", type=str, default="dog,cat,cow,person",
                    help="Comma list of YOLO classes to announce.")
    ap.add_argument("--hazard_on", type=int, default=2)
    ap.add_argument("--hazard_off", type=int, default=3)
    ap.add_argument("--hazard_cooldown_s", type=float, default=2.0)

    # OCR (high-res stream)
    ap.add_argument("--enable_ocr", action="store_true")
    ap.add_argument("--ocr_engine", type=str, default="auto",
                    choices=["tesseract", "easyocr", "auto"],
                    help="OCR engine. 'auto' tries tesseract then falls back to easyocr if installed.")
    ap.add_argument("--ocr_lang", type=str, default="eng",
                    help="Tesseract language string: 'eng' or 'eng+hin'.")
    ap.add_argument("--ocr_fps", type=float, default=1.25, help="High-res OCR stream FPS (0.2–1 recommended).")
    ap.add_argument("--ocr_width", type=int, default=1280)
    ap.add_argument("--ocr_height", type=int, default=720)
    ap.add_argument("--ocr_every_s", type=float, default=5.0)
    ap.add_argument("--ocr_min_len", type=int, default=7)
    ap.add_argument("--ocr_cooldown_s", type=float, default=7.0)
    ap.add_argument("--ocr_auto_min_quality", type=float, default=0.55,
                    help="In auto mode: if tesseract quality < this, try easyocr fallback.")
    ap.add_argument("--easyocr_gpu", action="store_true",
                    help="Enable GPU acceleration for EasyOCR (requires CUDA).")

    # Speech/debug
    ap.add_argument("--speak_every_s", type=float, default=1.1)
    ap.add_argument("--volume", type=float, default=1.0, help="Master volume (0.0-1.0).")
    ap.add_argument("--tts_off", action="store_true")
    ap.add_argument("--tts_rate", type=int, default=175)
    ap.add_argument("--debug", action="store_true")
    
    # Spatial audio
    ap.add_argument("--spatial_audio", action="store_true",
                    help="Enable positional audio cues (requires headphones).")
    ap.add_argument("--spatial_only", action="store_true",
                    help="Use only spatial audio tones, disable TTS voice.")
    ap.add_argument("--spatial_volume", type=float, default=0.7,
                    help="Volume for spatial audio tones (0.0-1.0).")
    
    # Production features
    ap.add_argument("--log_file", type=str, default=None,
                    help="Path to log file (with rotation, max 5MB, 3 backups).")
    ap.add_argument("--language", type=str, default="en", choices=["en", "hi"],
                    help="Language for TTS messages (en=English, hi=Hindi).")
    ap.add_argument("--max_retries", type=int, default=5,
                    help="Max device reconnection attempts before shutdown.")
    ap.add_argument("--watchdog_s", type=float, default=5.0,
                    help="Watchdog timeout in seconds (0 to disable).")
    ap.add_argument("--self_test", action="store_true",
                    help="Run startup self-test before navigation.")
    ap.add_argument("--monitor", action="store_true",
                    help="Enable text-based console monitor (for SSH/debugging).")
    ap.add_argument("--disable_imu", action="store_true",
                    help="Disable IMU-based tilt detection (pitch filtering).")

    args = ap.parse_args()

    # Sanity
    if args.yolo_fps <= 0 or args.yolo_fps > 30:
        raise SystemExit("yolo_fps must be in (0, 30].")
    if args.enable_ocr and args.ocr_fps <= 0:
        raise SystemExit("ocr_fps must be > 0 when OCR is enabled.")
    if args.ocr_every_s <= 0:
        raise SystemExit("ocr_every_s must be > 0.")
    if not re.fullmatch(r"[a-z]{3}(\+[a-z]{3})*", args.ocr_lang.strip()):
        raise SystemExit("ocr_lang must look like 'eng' or 'eng+hin' (3-letter codes separated by +).")
    if not (0.0 <= args.ocr_auto_min_quality <= 1.0):
        raise SystemExit("ocr_auto_min_quality must be 0..1.")

    # === Production initialization ===
    setup_logging(debug=args.debug, log_file=args.log_file)
    set_language(args.language)
    install_signal_handlers()
    
    logger.info("WalkingPal starting (language=%s, debug=%s)", args.language, args.debug)

    obstacle_mm = int(args.obstacle_m * 1000)
    clear_mm = int(args.clear_m * 1000)
    dropoff_mm = int(args.dropoff_m * 1000)

    # Initialize Audio Controller (Positional TTS + Tones)
    audio_controller = AudioController(tts_rate=args.tts_rate, volume=args.volume)

    # Startup self-test
    if args.self_test or args.monitor:
        logger.info("Running startup self-test/audio check...")
        if args.monitor:
            print(f"--- AUDIO CHECK: Playing test sound... ---", flush=True)
        
        try:
            audio_controller.speak(i18n('self_test'))
            time.sleep(0.5)
            if audio_controller.enabled:
                audio_controller.play_tone('obstacle', pan=0.0)
                time.sleep(0.3)
            audio_controller.speak(i18n('self_test_pass'))
            logger.info("Self-test passed.")
        except Exception as e:
            logger.error("AUDIO CHECK FAILED: %s", e)
            if args.monitor:
                print(f"!!! AUDIO CHECK FAILED: {e} !!!", flush=True)
    
    # Watchdog timer
    watchdog: Optional[Watchdog] = None
    if args.watchdog_s > 0:
        watchdog = Watchdog(timeout_s=args.watchdog_s)
        logger.debug("Watchdog enabled (timeout=%.1fs)", args.watchdog_s)

    # OCR init
    # - tesseract wrapper always attempted if enable_ocr and engine != easyocr-only
    need_tesseract = args.enable_ocr and (args.ocr_engine in ("tesseract", "auto"))
    pytesseract_mod = try_init_tesseract(need_tesseract)
    easyocr_wrap = EasyOCRWrapper(use_gpu=args.easyocr_gpu)

    # Async executor for OCR
    ocr_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="OCR_Worker")
    ocr_future: Optional[Future] = None

    hazards = {c.strip().lower() for c in args.hazards.split(",") if c.strip()}
    if not hazards:
        hazards = {"dog", "cat", "cow", "person"}

    # Initialize Depth Processor
    # Assuming OAK-D standard resolution for depth is 640x400
    nav_processor = DepthProcessor(width=640, height=400)

    # === Device connection/reconnection loop ===
    retry_count = 0
    while retry_count < args.max_retries and not _shutdown_requested.is_set():
        try:
            logger.info("Connecting to OAK-D device (attempt %d/%d)...", retry_count + 1, args.max_retries)
            
            pipeline, q_depth, q_preview, q_det, q_yolo_rgb, label_map, q_ocr_rgb, q_imu = build_pipeline_and_queues(
                fps_depth=args.fps,
                confidence=args.confidence,
                lr_check=args.lr_check,
                extended_disparity=args.extended_disparity,
                subpixel=args.subpixel,
                enable_yolo=args.enable_yolo,
                yolo_fps=args.yolo_fps,
                enable_ocr=args.enable_ocr,
                ocr_fps=args.ocr_fps,
                ocr_size=(args.ocr_width, args.ocr_height),
                enable_imu=not args.disable_imu,
            )

            # Debouncers/smoothers (reset on each connection)
            dropoff_db = DebouncedBool(on_count=args.confirm_frames, off_count=args.clear_frames)
            stairs_db = DebouncedBool(on_count=args.confirm_frames, off_count=args.clear_frames)
            blocked_db = DebouncedBool(on_count=max(1, args.confirm_frames - 1), off_count=args.clear_frames)
            uncertain_db = DebouncedBool(on_count=max(1, args.confirm_frames - 1), off_count=args.clear_frames)
            pothole_db = DebouncedBool(on_count=max(2, args.confirm_frames), off_count=max(2, args.clear_frames))
            low_light_db = DebouncedBool(on_count=10, off_count=10) # ~0.5-1s persistence
            hazard_db = DebouncedBool(on_count=args.hazard_on, off_count=args.hazard_off)
            dir_smoother = MajorityLabel(k=args.dir_window)

            last_spoken = ""
            last_spoken_ts = 0.0
            last_pothole_announce_ts = 0.0
            last_hazard_announce_ts = 0.0
            last_ocr_ts = 0.0
            last_ocr_announce_ts = 0.0
            last_ocr_announce_ts = 0.0
            last_low_light_announce_ts = 0.0
            last_monitor_ts = 0.0
            
            # Reset OCR future on reconnection
            ocr_future = None

            brightness: Optional[float] = None
            last_stairs_label: Optional[str] = None
            last_hazard_label: Optional[str] = None
            prev_stairs: bool = False
            roi_cache: Optional[Dict[str, Any]] = None
            pitch_deg: float = 0.0

            pipeline.start()
            logger.info("Pipeline started successfully.")
            
            # Reset retry count on successful connection
            retry_count = 0

            audio_controller.speak(i18n('nav_started'))
    
            # Start watchdog
            if watchdog:
                watchdog.start()

            last_depth_ts = time.time()
            if True: # with pipeline:
                while pipeline.isRunning() and not _shutdown_requested.is_set():
                    # Reset watchdog at start of each loop iteration
                    if watchdog:
                        watchdog.reset()
                    
                    # logger.info("Trace: Loop Start")
                    # Brightness (CAM_B preview)
                    pmsg = q_preview.tryGet()
                    if pmsg is not None:
                        brightness = float(np.mean(pmsg.getFrame()))
                    light = classify_light(brightness)
                    # logger.info("Trace: Light Classified")

                    # Low light warning
                    is_dark = (light == "dark")
                    low_light_active = low_light_db.update(is_dark)
                
                    if low_light_active and (now - last_low_light_announce_ts) >= 15.0:
                        # Announce every 15s if persistent
                        if (now - last_spoken_ts) >= args.speak_every_s:
                            msg = i18n('low_light')
                            audio_controller.speak(msg)
                            last_spoken = msg
                            last_spoken_ts = now
                            last_low_light_announce_ts = now

                    if light in ("dark", "dim"):
                        min_valid = float(args.min_valid_dim)
                        dropoff_invalid = min(0.95, float(args.dropoff_invalid_ratio) + float(args.dropoff_invalid_boost_dim))
                        require_center_for_clear = True
                    else:
                        min_valid = float(args.min_valid_normal)
                        dropoff_invalid = float(args.dropoff_invalid_ratio)
                        require_center_for_clear = False

                    now = time.time()


                    # YOLO hazards (optional)
                    hazard_detected = False
                    if args.enable_yolo and q_det is not None and q_yolo_rgb is not None:
                        if q_det.has():
                            det_msg: dai.ImgDetections = q_det.tryGet()
                            best = None
                            best_conf = 0.0
                            if det_msg is not None:
                                for d in det_msg.detections:
                                    if d.confidence < args.yolo_conf:
                                        continue
                                    if label_map and 0 <= d.label < len(label_map):
                                        label = str(label_map[d.label]).lower()
                                    else:
                                        label = str(d.label)
                                    if label in hazards and d.confidence > best_conf:
                                        best = label
                                        best_conf = d.confidence
                            hazard_detected = best is not None
                            if hazard_detected:
                                last_hazard_label = best
                    
                        # Drain unused RGB queue to prevent memory buildup
                        while q_yolo_rgb.has():
                            q_yolo_rgb.tryGet()
                    
                        # Update debouncer every frame for consistent state
                        hazard_present = hazard_db.update(hazard_detected)
                        if not hazard_present:
                            last_hazard_label = None

                        if hazard_present and last_hazard_label and (now - last_hazard_announce_ts) >= args.hazard_cooldown_s:
                            msg = i18n('hazard', last_hazard_label)
                            if (now - last_spoken_ts) >= args.speak_every_s:
                                # Spatial audio: only speech announcement
                                audio_controller.speak(msg)
                                last_spoken = msg
                                last_spoken_ts = now
                                last_hazard_announce_ts = now

                    # HIGH-RES OCR (optional) with graceful degradation
                    if args.enable_ocr and q_ocr_rgb is not None:
                        # 1. Check if ANY recurring text result is ready
                        if ocr_future and ocr_future.done():
                            try:
                                txt = ocr_future.result()
                                ocr_future = None # clear slot
                                if txt and (now - last_ocr_announce_ts) >= args.ocr_cooldown_s:
                                    omsg = i18n('sign_reads', sanitize_tts_text(txt))
                                    if (now - last_spoken_ts) >= args.speak_every_s:
                                        audio_controller.speak(omsg)
                                        last_spoken = omsg
                                        last_spoken_ts = now
                                        last_ocr_announce_ts = now
                            except Exception as e:
                                logger.error("OCR thread crashed: %s", e)
                                ocr_future = None

                        # 2. Submit new frame if idle
                        if ocr_future is None and q_ocr_rgb.has():
                            # Drain to get the newest frame
                            latest_ocr_pkt = None
                            while q_ocr_rgb.has():
                                latest_ocr_pkt = q_ocr_rgb.tryGet()
                        
                            if latest_ocr_pkt and (now - last_ocr_ts) >= args.ocr_every_s:
                                last_ocr_ts = now
                                frame_bgr = latest_ocr_pkt.getCvFrame().copy() # copy important for thread safety
                                ocr_future = ocr_executor.submit(
                                    run_ocr_auto,
                                    bgr=frame_bgr,
                                    ocr_engine=args.ocr_engine,
                                    tesseract_lang=args.ocr_lang.strip(),
                                    pytesseract_mod=pytesseract_mod,
                                    easyocr_wrap=easyocr_wrap,
                                    min_len=args.ocr_min_len,
                                    auto_fallback_min_quality=args.ocr_auto_min_quality,
                                )

                    # logger.info("Trace: Depth Check")
                    # Depth frame
                    depth_pkt = q_depth.tryGet()
                
                    if depth_pkt is None:
                        if time.time() - last_depth_ts > 5.0:
                            logger.warning("No depth frames received for 5 seconds - check device connection/bandwidth.")
                            last_depth_ts = time.time()
                        time.sleep(0.004)
                        continue
                    last_depth_ts = time.time()
                
                    depth = depth_pkt.getFrame()
                    h, w = depth.shape[:2]

                    # Cache ROI boundaries (computed once, resolution is constant)
                    if roi_cache is None:
                        third = w // 3
                        roi_cache = {
                            'band_y0': int(h * 0.35),
                            'band_y1': int(h * 0.72),
                            'third': third,
                            'roiL': (0, int(h * 0.35), third, int(h * 0.72)),
                            'roiC': (third, int(h * 0.35), 2 * third, int(h * 0.72)),
                            'roiR': (2 * third, int(h * 0.35), w, int(h * 0.72)),
                            'bottom_y0': int(h * 0.78),
                            'bottom_y1': int(h * 0.98),
                            'bottom_x0': int(w * 0.38),
                            'bottom_x1': int(w * 0.62),
                            'stairs_x0': int(w * 0.40),
                            'stairs_x1': int(w * 0.60),
                            'stairs_y0': int(h * 0.30),
                            'stairs_y1': int(h * 0.95),
                        }

                    # Update Pitch (Tilt) estimate
                    # Consolidated logic: Prefer IMU, fallback to SW.
                    # Sign convention: Negative = Looking Up, Positive = Looking Down (verified for SW).
                    p_curr = 0.0
                    has_imu_data = False
                    
                    if q_imu:
                        while q_imu.has():
                            imu_pkt = q_imu.tryGet()
                            if imu_pkt and imu_pkt.packets:
                                rv = imu_pkt.packets[-1].rotationVector
                                _, p_rad, _ = euler_from_quaternion(rv.i, rv.j, rv.k, rv.real)
                                p_curr = math.degrees(p_rad)
                                has_imu_data = True
                    
                    if not has_imu_data:
                        # Fallback (SW estimation)
                        # Note: estimate_pitch_from_depth returns Negative for Up, Positive for Down.
                        p_curr = estimate_pitch_from_depth(depth, camera_height_m=args.camera_height)
                    
                    # Apply simple IIR filter
                    pitch_deg = pitch_deg * 0.7 + p_curr * 0.3

                    # === NEW ROBUST NAVIGATION LOGIC ===
                    
                    nav_result = nav_processor.process_frame(depth)
                    
                    # Unwrap results
                    nav_hints = nav_result['nav']     # {'L': 'free'|'blocked', 'C': ..., 'R': ...}
                    nav_dists = nav_result['dists']   # {'L': dist_mm, ...}
                    dropoff_detected_now = nav_result['dropoff']
                    plane_eq = nav_result['plane']
                    debug_img = nav_result['debug_img']
                    
                    # Update Debug Visualization
                    if args.debug or args.monitor:
                         try:
                             # Overlay text
                             status_text = f"Plane: {'OK' if plane_eq is not None else 'LOST'}"
                             cv2.putText(debug_img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                             if dropoff_detected_now:
                                 cv2.putText(debug_img, "DROPOFF", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                             
                             if args.debug:
                                 cv2.imshow("WalkingPal Debug", debug_img)
                                 key = cv2.waitKey(1)
                                 if key == ord('q'):
                                      _shutdown_requested.set()
                         except Exception as e:
                             pass

                    # Logic Mapping to Tones/Speech
                    
                    # Dropoff
                    dropoff = dropoff_db.update(dropoff_detected_now)
                    
                    # Blocked?
                    is_blocked_L = (nav_hints['L'] == 'blocked')
                    is_blocked_C = (nav_hints['C'] == 'blocked')
                    is_blocked_R = (nav_hints['R'] == 'blocked')
                    
                    blocked_now = is_blocked_L or is_blocked_C or is_blocked_R
                    blocked = blocked_db.update(blocked_now)
                    
                    # Uncertain if no plane found?
                    uncertain_now = (plane_eq is None)
                    uncertain = uncertain_db.update(uncertain_now)

                    # Calculating 'stB' as it is used by potholes below
                    stB = roi_stats(depth, roi_cache['bottom_x0'], roi_cache['bottom_y0'], 
                                    roi_cache['bottom_x1'], roi_cache['bottom_y1'])

                    # Stairs
                    stairs_label_raw = detect_stairs(depth, roi_cache['stairs_x0'], roi_cache['stairs_x1'],
                                                      roi_cache['stairs_y0'], roi_cache['stairs_y1'])
                    stairs_raw = stairs_label_raw is not None
                    stairs = stairs_db.update(stairs_raw)
                    # Capture stable label on rising edge (transition from False to True)
                    if stairs and not prev_stairs:
                        last_stairs_label = stairs_label_raw or last_stairs_label
                    elif not stairs:
                        last_stairs_label = None
                    prev_stairs = stairs

                    # Potholes
                    pothole = False
                    if args.enable_potholes and not dropoff:
                        if stB.valid_ratio >= args.pothole_roi_valid:
                            score = pothole_score(depth, roi_cache['bottom_x0'], roi_cache['bottom_y0'],
                                                  roi_cache['bottom_x1'], roi_cache['bottom_y1'])
                            pothole = pothole_db.update(score >= args.pothole_score)
                        else:
                            pothole_db.update(False)

                    # Direction message
                    dist_msg = ""
                    if blocked:
                        # Find nearest obstacle distance from nav_dists
                        # Filter out invalid/far distances (9999.0)
                        dists = [val for val in nav_dists.values() if val < 9000]
                        if dists:
                            min_d_mm = min(dists)
                            min_d_m = min_d_mm / 1000.0
                            
                            if min_d_m < 3.0:
                                steps = int(round(min_d_m / 0.75))
                                if steps < 1: steps = 1
                                step_key = 'step' if steps == 1 else 'steps'
                                step_word = i18n(step_key)
                                dist_msg = f"{steps} {step_word}"
                            else:
                                dist_msg = f"{min_d_m:.1f}m"
                        
                        # Decide direction
                        free_left = not is_blocked_L
                        free_center = not is_blocked_C
                        free_right = not is_blocked_R
                        
                        nav_msg_raw = direction_text(free_left, free_center, free_right, dist_msg)
                    else:
                        nav_msg_raw = i18n('uncertain') if uncertain else i18n('clear')
                    
                    # Logic Tuning: If drop-off detected, suppress "Clear ahead" or "Uncertain"
                    # We only allow directional escapes ("Go left") or "Stop".
                    if dropoff:
                        # If the message suggests it's 'Clear', override it because of the dropoff.
                        # We don't want "Warning. Drop off ahead. Clear ahead."
                        if not blocked and not uncertain:
                             # Was "Clear ahead", now empty so we just hear "Drop off ahead"
                             nav_msg_raw = ""
                        elif uncertain:
                             if not blocked:
                                 # "Drop off ahead. Uncertain path." -> Okay-ish, but maybe redundant.
                                 pass 

                    nav_msg = dir_smoother.update(nav_msg_raw)

                    # Compose
                    parts = []
                    spatial_played = False  # Track if we played a spatial cue this frame
                
                    if dropoff:
                        # Dropoff warning message
                        parts.append(i18n('dropoff'))
                        
                    if stairs and last_stairs_label:
                        parts.append(i18n(last_stairs_label))
                        
                    if pothole:
                        parts.append(i18n('pothole'))
                
                    # Obstacle direction with positional audio
                    obstacle_pan = 0.0
                    if blocked:
                        # Determine obstacle pan
                        if stL.near_mm != 0 and stL.near_mm < obstacle_mm:
                            obstacle_pan = -0.8
                        elif stR.near_mm != 0 and stR.near_mm < obstacle_mm:
                            obstacle_pan = 0.8
                        
                        # Note: No tones, only speech panning below
                
                    parts.append(nav_msg)
                    final = " ".join(parts)

                    # Note: pothole warning is included in 'final' message, no separate announcement needed
                    # Just update cooldown timestamp when pothole is in final message
                    if pothole and (now - last_pothole_announce_ts) >= args.pothole_cooldown_s:
                        last_pothole_announce_ts = now

                    if final != last_spoken and (now - last_spoken_ts) >= args.speak_every_s:
                        # Speak with panning logic
                        # If blocked, use the calculated obstacle_pan. Otherwise center.
                        speak_pan = obstacle_pan if blocked else 0.0
                        audio_controller.speak(final, pan=speak_pan)
                        
                        last_spoken = final
                        last_spoken_ts = now

                    # Log every frame processed to debug depth values clearly
                    logger.info(
                        f"FRAME: Light={light} "
                        f"L={stL.near_mm}mm({stL.valid_ratio:.2f}) "
                        f"C={stC.near_mm}mm({stC.valid_ratio:.2f}) "
                        f"R={stR.near_mm}mm({stR.valid_ratio:.2f}) | "
                        f"Blocked={blocked} Uncertain={uncertain} Dropoff={dropoff} Pothole={pothole} Stairs={stairs} | "
                        f"MSG='{final}' | "
                        f"AudioEn={audio_controller.enabled}"
                    )
                
                    # Console Monitor (kept for compatibility if user uses --monitor)
                    if args.monitor:
                         pass

                    if args.debug:
                        d = np.clip(depth, 0, 5000)
                        d8 = (d / 5000.0 * 255.0).astype(np.uint8)
                        vis = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
                        header = (
                            f"{final} | light={light} "
                            f"| PITCH={pitch_deg:.1f} "
                            f"| yolo={'on' if args.enable_yolo else 'off'} "
                            f"| ocr={'on' if args.enable_ocr else 'off'} engine={args.ocr_engine} lang={args.ocr_lang}"
                        )
                        cv2.putText(vis, header, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3)
                        cv2.putText(vis, header, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
                        try:
                            cv2.imshow("OAK-D Blind Nav (YOLO+HiOCR) - press q", vis)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break
                        except Exception:
                            # Headless or X11 forwarding failed
                            pass

                # Normal exit from navigation loop (break or pipeline stopped)
            logger.info("Navigation loop exited normally.")
            break  # Exit retry loop on clean exit
        
        except KeyboardInterrupt:
            # Clean shutdown requested via signal
            logger.info("Keyboard interrupt received.")
            break
            
        except (RuntimeError, Exception) as dev_err:
            # Device disconnection or pipeline error
            if _shutdown_requested.is_set():
                break
            
            retry_count += 1
            logger.warning("Device error (%s). Retry %d/%d...", dev_err, retry_count, args.max_retries)
            audio_controller.speak(i18n('device_disconnect'))
            
            # Stop watchdog during reconnection
            if watchdog:
                watchdog.stop()
            
            if retry_count >= args.max_retries:
                logger.error("Max retries reached. Shutting down.")
                audio_controller.speak(i18n('device_reconnect_fail'))
                break
            
            # Wait before retry
            time.sleep(2.0)
            continue
    
    # === Cleanup ===
    if watchdog:
        watchdog.stop()
    
    # Shutdown OCR pool
    ocr_executor.shutdown(wait=False)
    
    try:
        audio_controller.speak(i18n('nav_stopped'))
        # Give it a moment to speak
        time.sleep(1.0)
    except:
        pass

    logger.info("Navigation stopped.")
    
    audio_controller.shutdown()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
