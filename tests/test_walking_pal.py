
import unittest
import sys
import os
import numpy as np

# Add parent directory to path to import walkingPal
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import walkingPal

class TestWalkingPalLogic(unittest.TestCase):

    def test_classify_light(self):
        """Test light level classification thresholds."""
        self.assertEqual(walkingPal.classify_light(None), "unknown")
        self.assertEqual(walkingPal.classify_light(10), "dark")
        self.assertEqual(walkingPal.classify_light(34), "dark")
        self.assertEqual(walkingPal.classify_light(35), "dim")
        self.assertEqual(walkingPal.classify_light(50), "dim")
        self.assertEqual(walkingPal.classify_light(74), "dim")
        self.assertEqual(walkingPal.classify_light(75), "normal")
        self.assertEqual(walkingPal.classify_light(200), "normal")

    def test_sanitize_tts_text(self):
        """Test text sanitization for TTS safety."""
        # Standard text
        self.assertEqual(walkingPal.sanitize_tts_text("Hello World"), "Hello World")
        # Control characters
        self.assertEqual(walkingPal.sanitize_tts_text("Hello\x00World"), "HelloWorld")
        # Length limit (mocking a long string)
        long_str = "a" * 600
        self.assertEqual(len(walkingPal.sanitize_tts_text(long_str)), 500)

    def test_direction_text(self):
        """Test navigation instruction generation."""
        # Set language to english for predictable strings
        walkingPal.set_language('en')
        
        # Stop
        self.assertEqual(walkingPal.direction_text(False, False, False), "Stop. No safe path.")
        
        # Single options
        self.assertEqual(walkingPal.direction_text(True, False, False), "Obstacle ahead. Go left.")
        self.assertEqual(walkingPal.direction_text(False, False, True), "Obstacle ahead. Go right.")
        
        # Note: Center-only isn't typical "go center" logic in code but let's check current behavior
        # It currently says "Obstacle ahead. Go center." which implies side blocks.
        self.assertEqual(walkingPal.direction_text(False, True, False), "Obstacle ahead. Go center.")

        # Dual options
        self.assertEqual(walkingPal.direction_text(True, False, True), "Obstacle ahead. You can go left or right.")
        
        # All clear
        self.assertEqual(walkingPal.direction_text(True, True, True), "Clear. You can go left, center, or right.")

    def test_debounced_bool(self):
        """Test DebouncedBool hysteresis."""
        db = walkingPal.DebouncedBool(on_count=3, off_count=2)
        
        # Initial state
        self.assertFalse(db.state)
        
        # Trigger true (needs 3)
        self.assertFalse(db.update(True)) # 1
        self.assertFalse(db.update(True)) # 2
        self.assertTrue(db.update(True))  # 3 - Switch!
        self.assertTrue(db.state)
        
        # Trigger false (needs 2)
        self.assertTrue(db.update(False)) # 1
        self.assertFalse(db.update(False))# 2 - Switch!
        self.assertFalse(db.state)

    def test_majority_label(self):
        """Test MajorityLabel voting."""
        ml = walkingPal.MajorityLabel(k=3)
        
        # Tie-breaking prefers most recent if counts equal? 
        # Code: max(counts.items(), key=lambda kv: (kv[1], 1 if kv[0] == latest else 0))[0]
        # Yes, standard voting with recency bias
        
        self.assertEqual(ml.update("A"), "A") # [A] -> A
        self.assertEqual(ml.update("B"), "B") # [A, B] -> B (tie, B recent)
        self.assertEqual(ml.update("A"), "A") # [A, B, A] -> A (2 vs 1)
        self.assertEqual(ml.update("C"), "C") # [B, A, C] -> C (tie A/B/C, favors latest C)
        # Buf: [B, A, C]. Counts: A:1, B:1, C:1. Latest: C.
        # Max key sort:
        # A: (1, 0)
        # B: (1, 0)
        # C: (1, 1) -> Selected
        self.assertEqual(ml.update("C"), "C") # [A, C, C] -> C

    def test_roi_stats_empty(self):
        """Test ROI stats with empty/invalid data."""
        # 0x0 image
        empty = np.zeros((0, 0), dtype=np.uint16)
        stats = walkingPal.roi_stats(empty, 0, 0, 10, 10)
        self.assertEqual(stats.valid_ratio, 0.0)

        # All zeros (invalid depth)
        zeros = np.zeros((10, 10), dtype=np.uint16)
        stats = walkingPal.roi_stats(zeros, 0, 0, 10, 10)
        self.assertEqual(stats.valid_ratio, 0.0)

    def test_roi_stats_valid(self):
        """Test ROI stats with valid data."""
        # 10x10 ROI
        data = np.zeros((10, 10), dtype=np.uint16)
        # Fill with 1000mm
        data[:, :] = 1000
        stats = walkingPal.roi_stats(data, 0, 0, 10, 10)
        
        self.assertEqual(stats.valid_ratio, 1.0)
        self.assertEqual(stats.median_mm, 1000)
        self.assertEqual(stats.near_mm, 1000)
        self.assertEqual(stats.far_mm, 1000)

        # Mixed data
        data[0:5, :] = 500  # Half 500
        data[5:10, :] = 1500 # Half 1500
        stats = walkingPal.roi_stats(data, 0, 0, 10, 10)
        
        self.assertEqual(stats.valid_ratio, 1.0)
        # Median of [500...500, 1500...1500] is 1000
        self.assertEqual(stats.median_mm, 1000)

if __name__ == '__main__':
    unittest.main()
