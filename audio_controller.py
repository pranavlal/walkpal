import os
import threading
import queue
import time
import tempfile
import uuid
import shutil
import numpy as np
import logging
import platform
import re
from typing import Dict, Any, Optional

logger = logging.getLogger("walkingpal.audio")

# -----------------------------
# Text sanitization for TTS
# -----------------------------
def sanitize_tts_text(text: str) -> str:
    """Remove control chars and limit length for safe TTS."""
    if not text: return ""
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    return text[:500]  # Prevent excessively long speech

# -----------------------------
# Unified Audio Controller (Positional TTS + Tones)
# -----------------------------
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
        self._sounds: Dict[str, Any] = {}
        self._temp_dir = tempfile.mkdtemp(prefix="walkpal_tts_")
        self._current_channel = None # Track active channel for interruption
        self._channel_lock = threading.Lock() # Protect channel access
        
        # Async setup
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = None
        
        # Init Pygame Mixer
        try:
            import pygame
            # Check for pygame._sdl2.audio for device listing (optional info)
            try:
                import pygame._sdl2.audio as sdl2_audio
                devices = sdl2_audio.get_audio_device_names(False)
                logger.info(f"Available Audio Devices: {devices}")
            except ImportError:
                pass
            except Exception:
                pass

            # standard 44.1kHz stereo
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
            self._pygame = pygame
            self._generate_tones()
            logger.info(f"AudioController: Pygame mixer initialized. Config: {self._pygame.mixer.get_init()}")
        except Exception as e:
            logger.error("AudioController: Pygame init failed: %s. Switching to FALLBACK TTS (Non-spatial).", e)
            self._pygame = None
            # Do NOT disable. We will use direct engine.say() as fallback.
            self.enabled = True

        # Start worker
        self._thread = threading.Thread(target=self._worker, args=(tts_rate,), daemon=True)
        self._thread.start()

    def _worker(self, tts_rate):
        """Background thread to handle TTS generation and playback."""
        # Fix for Windows COM threading (pyttsx3/sapi5)
        if platform.system().lower().startswith("win"):
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except ImportError:
                logger.warning("AudioController: pythoncom not found, TTS might fail on Windows.")
            except Exception as e:
                logger.warning("AudioController: CoInitialize failed: %s", e)

        # Engine will be initialized once
        engine = None
        if platform.system().lower().startswith("win"):
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", int(tts_rate))
                engine.setProperty("volume", self.volume)
            except Exception as e:
                logger.error(f"Failed to init pyttsx3 in worker: {e}")

        logger.info("AudioController: Worker started.")
        while not self._stop_event.is_set():
            try:
                # Get request: (type, data, pan)
                item = self._queue.get(timeout=0.1)
                
                if item is None: break # Poison pill
                
                cmd, data, pan = item
                
                if cmd == 'tone':
                    self._play_tone_sync(data, pan)
                elif cmd == 'speak':
                    # Pass the persistent engine
                    self._speak_sync(engine, tts_rate, data, pan)
                
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("AudioController: Worker error: %s", e)
        
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
            # Direct play for immediate feedback if possible, but going through queue ensures
            # we respect the single-threaded nature if that was a concern. 
            # However, for alarms, we might want to skip the queue?
            # WalkingPal original code queued it with `_play_tone_sync`.
            self._play_tone_sync(name, pan)

    def speak(self, text: str, pan: float = 0.0):
        """Queue a TTS request."""
        if self.enabled and text:
            # Drain queue of old speech if it's backing up to prioritize new info
            if self._queue.qsize() > 0:
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
            # Force stop previous speech for immediate hazard tone
            with self._channel_lock:
                if self._current_channel and self._current_channel.get_busy():
                    self._current_channel.stop()
            self._play_sound(sound, pan)

    def _speak_sync(self, engine, tts_rate, text, pan):
        # Fallback if pygame is dead or engine failed
        if not self._pygame or not engine:
             # Try one-off init as last resort
             try:
                 import pyttsx3
                 temp_engine = pyttsx3.init()
                 temp_engine.setProperty("rate", int(tts_rate))
                 temp_engine.setProperty("volume", self.volume)
                 temp_engine.say(text)
                 temp_engine.runAndWait()
                 temp_engine.stop()
             except Exception as e:
                 logger.error(f"Fallback TTS error: {e}")
             return

        fname = os.path.join(self._temp_dir, f"{uuid.uuid4().hex}.wav")
        try:
            # Generate WAV using persistent engine
            engine.save_to_file(text, fname)
            engine.runAndWait()
        except Exception as e:
            logger.error(f"TTS generation error: {e}")
            pass

        if os.path.exists(fname):
            fsize = os.path.getsize(fname)
            
            if fsize < 100:
                logger.warning(f"Audio file is suspiciously small ({fsize} bytes). SAPI might have failed silently.")

            try:
                sound = self._pygame.mixer.Sound(fname)
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
            # STOP PREVIOUS AUDIO if playing
            with self._channel_lock:
                if self._current_channel and self._current_channel.get_busy():
                    self._current_channel.stop()
                
                channel = sound.play()
                if channel:
                    self._current_channel = channel # Track it
                    pan = max(-1.0, min(1.0, pan))
                    left = 1.0 - max(0.0, pan)
                    right = 1.0 + min(0.0, pan)
                    channel.set_volume(left * self.volume, right * self.volume)
        except Exception as e:
            logger.error("Sound Play Error: %s", e)

    def shutdown(self):
        self._stop_event.set()
        if self._thread:
            self._queue.put(None) # wake up
            self._thread.join(timeout=2.0)
            
        if self._pygame:
            self._pygame.mixer.quit()
        
        try:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception:
            pass
