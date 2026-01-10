
import depthai as dai
import time
import sys

def test_stream():
    print("Testing OAK-D Camera Stream...")
    pipeline = dai.Pipeline()
    
    # Create RGB camera
    cam_node = pipeline.create(dai.node.Camera)
    cam = cam_node.build(dai.CameraBoardSocket.CAM_A)
    
    # Try to set FPS via requestOutput as learned
    xout = cam.requestOutput((300, 300), type=dai.ImgFrame.Type.BGR888p, fps=10.0)
    q = xout.createOutputQueue(maxSize=1, blocking=True)
    
    print("Pipeline defined. Connecting...")
    try:
        pipeline.start()
        print("Pipeline started. Running loop...")
        
        with pipeline:
            # Try to get 3 frames
            for i in range(3):
                print(f"Waiting for frame {i+1}...")
                img = q.get() # Blocking call
                if img:
                    print(f"Got frame {i+1}: {img.getWidth()}x{img.getHeight()} size={len(img.getData())}")
            
        print("Stream test PASSED.")
            
    except Exception as e:
        print(f"Stream test FAILED: {e}")

if __name__ == "__main__":
    test_stream()
