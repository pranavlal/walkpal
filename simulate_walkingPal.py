
import sys
import time
import unittest.mock as mock
import numpy as np
import logging

# Configure logging to see output
logging.basicConfig(level=logging.INFO)

# Mock depthai before importing walkingPal if possible, 
# but walkingPal imports depthai at top level.
# We need to mock it in sys.modules.

sys.modules['depthai'] = mock.Mock()
import depthai as dai # This is now the mock

# Setup mock constants that walkingPal uses
dai.CameraBoardSocket.CAM_B = "CAM_B"
dai.CameraBoardSocket.CAM_C = "CAM_C"
dai.CameraBoardSocket.CAM_A = "CAM_A"
dai.ImgFrame.Type.GRAY8 = "GRAY8"
dai.ImgFrame.Type.BGR888p = "BGR888p"
dai.ImgResizeMode.CROP = "CROP"
dai.MedianFilter.KERNEL_7x7 = "7x7"
dai.node.StereoDepth.PresetMode.FAST_DENSITY = "FAST_DENSITY"

# Now import walkingPal
import walkingPal

# Mock pipeline and queues
def simulate():
    print(">>> Starting WalkingPal SIMULATION (No Hardware) <<<")
    
    # Mock the pipeline build to return mock queues
    q_depth = mock.Mock()
    q_preview = mock.Mock()
    
    # Mock depth frame: standard 640x400
    # Let's make a frame that is close to the camera (blocked)
    # 1.2m = 1200mm. Let's put 500mm (0.5m) in the center.
    blocked_frame = np.ones((400, 640), dtype=np.uint16) * 500
    
    # Mock data packet
    mock_depth_pkt = mock.Mock()
    mock_depth_pkt.getFrame.return_value = blocked_frame
    
    # Mock preview (brightness)
    mock_preview_pkt = mock.Mock()
    mock_preview_pkt.getFrame.return_value = np.zeros((200, 320), dtype=np.uint8) + 100 # Normal brightness
    
    # Queue behavior: return data once, then empty? 
    # Or just always return data.
    q_depth.has.return_value = True
    q_depth.get.return_value = mock_depth_pkt
    
    q_preview.has.return_value = True
    q_preview.get.return_value = mock_preview_pkt
    
    # Patch build_pipeline_and_queues to return our mocks
    # return pipeline, q_depth, q_preview, q_det, q_yolo_rgb, label_map, q_ocr_rgb
    with mock.patch('walkingPal.build_pipeline_and_queues') as mock_build:
        mock_pipeline = mock.Mock()
        mock_pipeline.__enter__ = mock.Mock(return_value=mock_pipeline)
        mock_pipeline.__exit__ = mock.Mock(return_value=None)
        mock_pipeline.isRunning.side_effect = [True, True, True, False] # Run 3 loops then stop
        
        mock_build.return_value = (mock_pipeline, q_depth, q_preview, None, None, None, None)
        
        # Patch pipeline.start() to do nothing
        
        # Run main!
        # We need to pass args. We can mock argparse or pass via sys.argv
        sys.argv = ["walkingPal.py", "--monitor", "--speak_every_s", "0.5"]
        
        try:
            walkingPal.main()
        except SystemExit:
            pass
        except Exception as e:
            print(f"Simulation crashed: {e}")

if __name__ == "__main__":
    simulate()
