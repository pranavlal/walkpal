
import pyttsx3
import pygame
import time
import sys
import os

def test_tts():
    print("Testing TTS (pyttsx3)...")
    try:
        engine = pyttsx3.init()
        engine.say("Testing text to speech.")
        engine.runAndWait()
        print("TTS command sent. Did you hear it?")
    except Exception as e:
        print(f"TTS Failed: {e}")

def test_spatial():
    print("\nTesting Spatial Audio (pygame)...")
    try:
        # Dummy audio generation (sine wave)
        import numpy as np
        
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        duration = 1.0
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        wave = np.sin(2 * np.pi * 440 * t) # 440Hz A4
        
        # Stereo
        wave_int = (wave * 32767 * 0.5).astype(np.int16)
        stereo = np.column_stack((wave_int, wave_int))
        
        sound = pygame.sndarray.make_sound(stereo)
        channel = sound.play()
        
        # Pan left then right
        print("Playing tone: Center...")
        time.sleep(0.5)
        print("Pan Left...")
        channel.set_volume(1.0, 0.0)
        time.sleep(0.5)
        print("Pan Right...")
        channel.set_volume(0.0, 1.0)
        time.sleep(0.5)
        
        pygame.mixer.stop()
        pygame.mixer.quit()
        print("Spatial audio test complete.")
    except Exception as e:
        print(f"Spatial Audio Failed: {e}")

if __name__ == "__main__":
    test_tts()
    test_spatial()
