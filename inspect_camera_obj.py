
import depthai as dai
import sys

def inspect():
    pipeline = dai.Pipeline()
    try:
        camL_node = pipeline.create(dai.node.Camera)
        camL = camL_node.build(dai.CameraBoardSocket.CAM_B)
        
        print("Type of camL:", type(camL))
        print("Dir of camL:", dir(camL))
        
        # Check if setFps exists
        if hasattr(camL, 'setFps'):
            print("camL has setFps")
        else:
            print("camL DOES NOT have setFps")
            
        # Check if requestOutput takes fps
        # We can't easily check signature of bound C++ method, but we can try calling it
        try:
            camL.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8, fps=15.0)
            print("requestOutput accepts fps argument")
        except Exception as e:
            print("requestOutput failed with fps arg:", e)

    except Exception as e:
        print("Inspection failed:", e)

if __name__ == "__main__":
    inspect()
