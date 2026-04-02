import numpy as np
from walkingPal import WalkPalApp
import argparse

def test_roi_scaling():
    args = argparse.Namespace(max_retries=5, tts_rate=175, volume=1.0, openai_model="gpt-4o-mini",
                              enable_local_vlm=False, confirm_frames=5, clear_frames=2, dir_window=5,
                              watchdog_s=5.0, record=False)
    app = WalkPalApp(args, {})
    
    # Simulating 1280x720 frame
    depth_720 = np.zeros((720, 1280), dtype=np.uint16)
    app.ensure_roi(depth_720)
    
    roi = app.state['roi']
    print(f"720p ROI: {roi}")
    assert roi['bx0'] == int(1280 * 0.38)
    assert roi['by1'] == int(720 * 0.98)
    
    # Reset and test 640x400
    app.state['roi'] = None
    depth_400 = np.zeros((400, 640), dtype=np.uint16)
    app.ensure_roi(depth_400)
    roi2 = app.state['roi']
    print(f"400p ROI: {roi2}")
    assert roi2['bx0'] == int(640 * 0.38)
    
    print("ROI Scaling Test PASSED")

if __name__ == "__main__":
    test_roi_scaling()
