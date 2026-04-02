import unittest
import numpy as np
import sys
import os

# Add parent directory to path to import walkingPal
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from walkingPal import detect_dropoff, RoiStats

class TestDropoffLogic(unittest.TestCase):
    def setUp(self):
        # Common params
        self.roi_cache = {}
        self.base_dropoff_mm = 3500
        self.dropoff_invalid_ratio = 0.6
        self.min_valid = 0.2
        self.require_center_for_clear = False
        self.camera_height_m = 1.5
        
        # Create dummy depth map (unused by function as it uses stats, but required for signature)
        self.depth_mm = np.zeros((100, 100), dtype=np.uint16)

    def test_safe_walking_good_light(self):
        """Scenario: Walking on floor, good visibility, no dropoff."""
        # Stats: High validity, reasonable distance (2m)
        stB = RoiStats(valid_ratio=0.9, near_mm=1000, median_mm=2000, far_mm=3000)
        # Visible oath ahead
        stC = RoiStats(valid_ratio=0.9, near_mm=2000, median_mm=3000, far_mm=4000)
        
        result = detect_dropoff(
            self.depth_mm, self.roi_cache, self.base_dropoff_mm, self.dropoff_invalid_ratio,
            self.min_valid, self.require_center_for_clear,
            stB, stB, stC, stB, # Use same for L/R for simplicity
            pitch_deg=0.0, camera_height_m=self.camera_height_m
        )
        self.assertFalse(result, "Should be SAFE (False)")

    def test_dropoff_invalid_pixels_good_light(self):
        """Scenario: Cliff edge, floor disappears (invalid pixels), good visibility elsewhere."""
        # Stats: Low validity at bottom (cliff)
        stB = RoiStats(valid_ratio=0.1, near_mm=0, median_mm=0, far_mm=0)
        # Path ahead is visible (e.g. looking across the gap)
        stC = RoiStats(valid_ratio=0.8, near_mm=5000, median_mm=6000, far_mm=7000)
        
        result = detect_dropoff(
            self.depth_mm, self.roi_cache, self.base_dropoff_mm, self.dropoff_invalid_ratio,
            self.min_valid, self.require_center_for_clear,
            stB, stB, stC, stB,
            pitch_deg=0.0, camera_height_m=self.camera_height_m
        )
        self.assertTrue(result, "Should detect DROPOFF (True) due to invalid pixels with visible path")

    def test_dark_room_safety(self):
        """Scenario: Dark room, sensor returns invalid pixels everywhere."""
        # Stats: Low validity everywhere
        stB = RoiStats(valid_ratio=0.05, near_mm=0, median_mm=0, far_mm=0)
        stC = RoiStats(valid_ratio=0.05, near_mm=0, median_mm=0, far_mm=0)
        stL = RoiStats(valid_ratio=0.05, near_mm=0, median_mm=0, far_mm=0)
        stR = RoiStats(valid_ratio=0.05, near_mm=0, median_mm=0, far_mm=0)
        
        result = detect_dropoff(
            self.depth_mm, self.roi_cache, self.base_dropoff_mm, self.dropoff_invalid_ratio,
            self.min_valid, self.require_center_for_clear,
            stB, stL, stC, stR,
            pitch_deg=0.0, camera_height_m=self.camera_height_m
        )
        self.assertFalse(result, "Should return FALSE (Safe) in dark check to avoid false dropoff warning (Main loop handles 'Uncertain')")

    def test_looking_up_sky(self):
        """Scenario: Looking up at sky (-20 deg pitch). Geometry says ground is infinite."""
        # Stats: Valid but far (sky/ceiling or just noise) or invalid
        stB = RoiStats(valid_ratio=0.2, near_mm=5000, median_mm=9000, far_mm=10000)
        stC = RoiStats(valid_ratio=0.2, near_mm=10000, median_mm=10000, far_mm=10000)
        
        result = detect_dropoff(
            self.depth_mm, self.roi_cache, self.base_dropoff_mm, self.dropoff_invalid_ratio,
            self.min_valid, self.require_center_for_clear,
            stB, stB, stC, stB,
            pitch_deg=-20.0, camera_height_m=self.camera_height_m
        )
        self.assertFalse(result, "Should ignore dropoff potential when looking up")

if __name__ == '__main__':
    unittest.main()
