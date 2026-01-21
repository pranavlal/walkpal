import blobconverter
import shutil
import os

def download_yolo():
    print("Attempting to download YOLOv8n 640x352...")
    try:
        # Try fetching specific model name often used in OAK examples
        # If this fails, we might need to specify parameters differently
        blob_path = blobconverter.from_zoo(
            name="yolov8n_coco_640x352",
            zoo_type="depthai",
            shaves=6
        )
        print(f"Downloaded: {blob_path}")
        
        target = "yolov8n.blob"
        shutil.move(blob_path, target)
        print(f"Moved to {target}")
        return True
    except Exception as e:
        print(f"Failed standard download: {e}")
        return False


def download_monodepth():
    print("Attempting to download MiDaS v2.1 Small (ONNX)...")
    url = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"
    target = "model-small.onnx"
    
    if os.path.exists(target):
        print(f"{target} already exists.")
        return True
        
    try:
        import urllib.request
        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, target)
        print(f"Downloaded: {target}")
        return True
    except Exception as e:
        print(f"Failed to download MiDaS: {e}")
        return False


def download_mobilenet():
    print("Checking MobileNet-SSD blob...")
    target = "mobilenet-ssd.blob"
    if os.path.exists(target):
        print(f"{target} already exists.")
        return True

    try:
        print("Downloading MobileNet-SSD...")
        blob_path = blobconverter.from_zoo(name="mobilenet-ssd", zoo_type="intel", shaves=6)
        shutil.move(blob_path, target)
        print(f"Downloaded: {target}")
        return True
    except Exception as e:
        print(f"Failed to download MobileNet: {e}")
        return False

if __name__ == "__main__":
    download_yolo()
    download_monodepth()
    download_mobilenet()
