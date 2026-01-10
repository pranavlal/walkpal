
import unittest
import numpy as np
import sys
import os

# Ensure import of local module
sys.path.append(os.getcwd())
from walkingPal import estimate_pitch_from_depth, roi_stats, flatten_config

class TestWalkingPal(unittest.TestCase):
    
    def test_config_flatten(self):
        cfg = {"camera": {"fps": 30, "height": 1.5}, "debug": True}
        flat = flatten_config(cfg)
        self.assertEqual(flat['fps'], 30)
        self.assertEqual(flat['height'], 1.5)
        self.assertEqual(flat['debug'], True)

    def test_roi_stats_empty(self):
        # Empty depth
        d = np.zeros((100, 100), dtype=np.uint16)
        st = roi_stats(d, 0, 0, 10, 10)
        self.assertEqual(st.valid_ratio, 0.0)
        self.assertEqual(st.near_mm, 0)
    
    def test_roi_stats_full(self):
        # 2 meters everywhere
        d = np.ones((100, 100), dtype=np.uint16) * 2000
        st = roi_stats(d, 0, 0, 100, 100)
        self.assertEqual(st.valid_ratio, 1.0)
        self.assertAlmostEqual(st.median_mm, 2000, delta=10)
        
    def test_pitch_sign(self):
        # Verify Negative = Looking Up
        h, w = 400, 640
        # Z far (Look Up) -> Should be negative
        d_up = np.ones((h, w), dtype=np.uint16) * 8000 # 8m
        p_up = estimate_pitch_from_depth(d_up, camera_height_m=1.5)
        self.assertLess(p_up, -5.0) # Should be around -10 deg
        
        # Z near (Look Down) -> Should be positive
        d_down = np.ones((h, w), dtype=np.uint16) * 1500 # 1.5m
        p_down = estimate_pitch_from_depth(d_down, camera_height_m=1.5)
        self.assertGreater(p_down, 10.0) # Should be around +25 deg

if __name__ == '__main__':
    unittest.main()
