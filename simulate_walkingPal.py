
import sys
import time
import unittest.mock as mock
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)

# Mock depthai before importing anything
sys.modules['depthai'] = mock.Mock()
import depthai as dai

# Setup mock constants
dai.CameraBoardSocket.CAM_B = "CAM_B"
dai.CameraBoardSocket.CAM_C = "CAM_C"
dai.CameraBoardSocket.CAM_A = "CAM_A"
dai.ImgFrame.Type.GRAY8 = "GRAY8"
dai.ImgFrame.Type.BGR888p = "BGR888p"
dai.ImgResizeMode.CROP = "CROP"
dai.MedianFilter.KERNEL_7x7 = "7x7"
dai.node.StereoDepth.PresetMode.FAST_DENSITY = "FAST_DENSITY"

# Mock classes for walkingPal
@dataclass
class MockFrames:
    depth: Optional[np.ndarray]
    preview: Optional[np.ndarray]
    scene: Optional[np.ndarray]
    video: Optional[np.ndarray]

class MockOakDCamera:
    def __init__(self, **kwargs):
        self.label_map = ["person", "chair", "bottle"]
        self.running = False
        
    def start(self):
        self.running = True
        return True
        
    def stop(self):
        self.running = False
        
    def is_running(self):
        # Run for a fixed number of frames then stop
        # access global counter if needed, or just run endlessly until exception
        # Let's say we check a global or static
        return self.running

    def get_frames(self):
        # 1.2m = 1200mm. 
        depth = np.ones((400, 640), dtype=np.uint16) * 1200
        # Make a clear path in center? No, let's test obstacle.
        # depth[100:300, 200:440] = 500 # 0.5m obstacle
        
        preview = np.zeros((200, 320), dtype=np.uint8) + 100 # brightness
        scene = np.zeros((300, 300, 3), dtype=np.uint8)
        
        return MockFrames(depth, preview, scene, None)
        
    def get_detections(self):
        return []
        
    def get_imu(self):
        return None

# Import walkingPal
import walkingPal

def simulate():
    print(">>> Starting WalkingPal SIMULATION (Mocked Hardware) <<<")
    
    # Patch OakDCamera
    with mock.patch('walkingPal.OakDCamera', side_effect=MockOakDCamera):
        
        # Also patch WebcamCamera just in case logic falls through
        with mock.patch('walkingPal.WebcamCamera', side_effect=MockOakDCamera):
            
            # Patch AudioController to avoid actual audio or thread issues?
            # Actually we want to test Threading improvements, so let's keep it real 
            # but maybe suppress sound output if needed. 
            # The code uses pyttsx3 or pygame. simpler to let it run.
            
            # Mock CLI args
            sys.argv = ["walkingPal.py", "--headless"]
            
            # Run main with a timeout or interrupt
            # We can use a thread to stop it after 5 seconds
            import threading
            def stopper():
                time.sleep(5)
                print(">>> Stopping Simulation <<<")
                walkingPal.request_shutdown()
                
            t = threading.Thread(target=stopper, daemon=True)
            t.start()
            
            try:
                walkingPal.main()
            except SystemExit:
                pass
            except Exception as e:
                print(f"Simulation crashed: {e}")
                raise

if __name__ == "__main__":
    simulate()
