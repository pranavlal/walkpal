
import unittest
import numpy as np
import sys
import os

# Mock dependencies if needed, or ensure path is set
sys.path.append("/home/pranav/walkpal")

# We need to mock depthai and pygame before importing walkingPal
from unittest.mock import MagicMock
sys.modules['depthai'] = MagicMock()
sys.modules['pygame'] = MagicMock()
sys.modules['pygame._sdl2'] = MagicMock()
sys.modules['pygame._sdl2.audio'] = MagicMock()
sys.modules['cv2'] = MagicMock()

import walkingPal

class TestWalkingPalRobustness(unittest.TestCase):

    def test_detect_stairs_flat_ramp(self):
        """Flat ramp/floor should NOT be detected as stairs."""
        # Create a gradient depth map (flat floor tilting away)
        h, w = 400, 640
        depth_col = np.linspace(5000, 1000, h).astype(np.uint16)
        depth = np.tile(depth_col[:, None], (1, w))
        
        # stairs ROI
        roi_cache = {
             'stairs_x0': int(w * 0.40),
             'stairs_x1': int(w * 0.60),
             'stairs_y0': int(h * 0.30),
             'stairs_y1': int(h * 0.95),
        }
        
        # Add noise to make it realistic
        noise = np.random.randint(-50, 50, depth.shape).astype(np.int16)
        depth = np.clip(depth + noise, 0, 10000).astype(np.uint16)
        
        result = walkingPal.detect_stairs(depth, 
                                          roi_cache['stairs_x0'], roi_cache['stairs_x1'],
                                          roi_cache['stairs_y0'], roi_cache['stairs_y1'])
                                          
        print(f"Flat Ramp Result: {result}")
        self.assertIsNone(result, "Flast ramp matching stairs logic incorrectly")

    def test_dropoff_logic(self):
        """Verify dropoff logic using refactored function."""
        # ROI constants 
        h, w = 400, 640
        roi_cache = {
            'bottom_y0': int(h * 0.78),
            'bottom_y1': int(h * 0.98),
            'bottom_x0': int(w * 0.38),
            'bottom_x1': int(w * 0.62),
            'third': w // 3,
            'roiL': (0, int(h * 0.35), w // 3, int(h * 0.72)),
            'roiC': (w // 3, int(h * 0.35), 2 * (w // 3), int(h * 0.72)),
            'roiR': (2 * (w // 3), int(h * 0.35), w, int(h * 0.72)),
        }
        
        dropoff_m = 3500
        dropoff_invalid_ratio = 0.75
        min_valid = 0.2
        require_center_for_clear = False
        
        # 1. Real Dropoff: Center Valid (Path visible), Bottom Invalid (Cliff)
        depth = np.zeros((h, w), dtype=np.uint16)
        # Center is VALID and Close (2m)
        roiC = roi_cache['roiC']
        depth[roiC[1]:roiC[3], roiC[0]:roiC[2]] = 2000 
        
        # Bottom is INVALID (0)
        
        is_dropoff = walkingPal.detect_dropoff(depth, roi_cache, dropoff_m, dropoff_invalid_ratio, min_valid, require_center_for_clear)
        print(f"Real Dropoff Detected: {is_dropoff}")
        self.assertTrue(is_dropoff, "Should detect real dropoff")
        
        # 2. Dark Room (False Negative protection): Everything Invalid
        depth.fill(0)
        is_dropoff_dark = walkingPal.detect_dropoff(depth, roi_cache, dropoff_m, dropoff_invalid_ratio, min_valid, require_center_for_clear)
        print(f"Dark Room Detected: {is_dropoff_dark}")
        self.assertFalse(is_dropoff_dark, "Should NOT detect dropoff in full darkness/obscured view")

if __name__ == '__main__':
    unittest.main()
