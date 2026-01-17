import time
import logging
import depthai as dai
import numpy as np
from typing import Optional, Any
from .base import CameraInterface, CameraFrames

logger = logging.getLogger("walkingpal.oakd")

class OakDCamera(CameraInterface):
    def __init__(self, enable_yolo=False, enable_potholes=False, enable_ocr=False):
        self.enable_yolo = enable_yolo
        self.enable_potholes = enable_potholes
        self.enable_ocr = enable_ocr
        
        self.device = None
        self.pipeline = None
        
        self.q_depth = None
        self.q_det = None 
        self.q_yolo_rgb = None
        self.q_ocr_rgb = None
        self.q_scene_rgb = None
        self.q_imu = None
        self.q_preview = None
        
        self.label_map = None

    def start(self) -> bool:
        """Initializes the OAK-D pipeline."""
        try:
            self.pipeline = self._build_pipeline()
            
            # Connect to device
            self.device = dai.Device(self.pipeline)
            self.device.startPipeline()
            
            # Create queues based on stream names defined in _build_pipeline
            self.q_depth = self.device.getOutputQueue(name="depth", maxSize=2, blocking=False)
            self.q_preview = self.device.getOutputQueue(name="preview", maxSize=1, blocking=False)
            
            if self.enable_imu:
                try:
                    self.q_imu = self.device.getOutputQueue(name="imu", maxSize=10, blocking=False)
                except Exception:
                    logger.warning("IMU queue not found (maybe failed to create node).")

            if self.enable_yolo:
                try:
                    self.q_det = self.device.getOutputQueue(name="nn", maxSize=4, blocking=False)
                    self.q_yolo_rgb = self.device.getOutputQueue(name="yolo_rgb", maxSize=2, blocking=False)
                except Exception:
                    pass

            if self.enable_ocr:
                try:
                    self.q_ocr_rgb = self.device.getOutputQueue(name="ocr_rgb", maxSize=2, blocking=False)
                except Exception:
                    pass
            
            # Scene RGB
            if not self.enable_ocr:
                try:
                    self.q_scene_rgb = self.device.getOutputQueue(name="scene_rgb", maxSize=2, blocking=False)
                except Exception:
                    pass

            logger.info("OAK-D Pipeline started successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start OAK-D: {e}")
            return False

    def stop(self):
        if self.device:
            self.device.close()
            self.device = None

    def is_running(self) -> bool:
        if self.device and not self.device.isClosed():
             # In some DepthAI versions isClosed() or isPipelineRunning(). 
             # Safe check:
             return True 
        return False

            
    def get_frames(self) -> CameraFrames:
        frames = CameraFrames()
        frames.timestamp = time.time()
        
        if not self.device:
            return frames

        # 1. Depth
        if self.q_depth:
            d_pkt = self.q_depth.tryGet()
            if d_pkt:
                frames.depth = d_pkt.getFrame()

        # 2. Video / OCR Frame
        if self.q_ocr_rgb:
            ocr_pkt = self.q_ocr_rgb.tryGet()
            if ocr_pkt:
                frames.video = ocr_pkt.getCvFrame() 
        elif self.q_yolo_rgb:
             # If YOLO is used but OCR is off, we still get RGB frames from passthrough
             yolo_pkt = self.q_yolo_rgb.tryGet()
             if yolo_pkt:
                 frames.video = yolo_pkt.getCvFrame()
             
        # 3. Scene Frame
        if self.q_scene_rgb:
            # Dedicated scene queue
            s_pkt = self.q_scene_rgb.tryGet()
            if s_pkt:
                frames.scene = s_pkt.getCvFrame()
        
        # 4. Preview (Brightness)
        if self.q_preview:
            p_pkt = self.q_preview.tryGet()
            if p_pkt:
                frames.preview = p_pkt.getFrame() # Grayscale usually
        
        # Fallback for scene frame if using shared stream
        if frames.scene is None and frames.video is not None:
             frames.scene = frames.video

        return frames
        
    def get_detections(self):
        """OAK-D specific method to get NN results."""
        if self.q_det:
             det = self.q_det.tryGet()
             if det:
                 return det.detections
        return []
        
    def get_imu(self):
        pkt = None
        if self.q_imu:
             # Drain queue to get latest
             while self.q_imu.has():
                 pkt = self.q_imu.tryGet()
        return pkt

    def get_intrinsics(self):
        if self.device:
             try:
                 calib = self.device.readCalibration()
                 return calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C) # Right camera commonly used for depth alignment reference in some pairs, 
                 # But we used align=CAM_A. 
                 # Let's return CAM_A intrinsics?
                 # walkingPal didn't seem to use intrinsics explicitly in the parts I read, but nav_processor might.
                 # Actually, CameraInterface just needs "something".
                 # I'll return CAM_A intrinsics.
                 return calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A)
             except Exception:
                 pass
        return None

    def _build_pipeline(self):
        """Standard DepthAI pipeline construction."""
        pipeline = dai.Pipeline()

        fps_depth = 30.0
        confidence = 200
        lr_check = True
        extended_disparity = False
        subpixel = False
        
        # CAM_B (Left)
        camL = pipeline.create(dai.node.MonoCamera)
        camL.setBoardSocket(dai.CameraBoardSocket.LEFT)
        camL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        camL.setFps(fps_depth)
        
        # CAM_C (Right)
        camR = pipeline.create(dai.node.MonoCamera)
        camR.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        camR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        camR.setFps(fps_depth)

        # Stereo Node
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.initialConfig.setConfidenceThreshold(confidence)
        stereo.setLeftRightCheck(lr_check)
        stereo.setExtendedDisparity(extended_disparity)
        stereo.setSubpixel(subpixel)
        
        try:
            stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        except AttributeError:
            pass

        camL.out.link(stereo.left)
        camR.out.link(stereo.right)

        # XLinkOut: Depth
        xoutDepth = pipeline.create(dai.node.XLinkOut)
        xoutDepth.setStreamName("depth")
        stereo.depth.link(xoutDepth.input)
        
        # IMU
        # Note: We enable IMU if requested, though original code had arg disable_imu
        # We'll just enable it if possible.
        try:
            imu = pipeline.create(dai.node.IMU)
            imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, 10)
            imu.setBatchReportThreshold(1)
            imu.setMaxBatchReports(10)
            
            xoutImu = pipeline.create(dai.node.XLinkOut)
            xoutImu.setStreamName("imu")
            imu.out.link(xoutImu.input)
            self.enable_imu = True
        except Exception:
            self.enable_imu = False
            
        # CAM_A (Color)
        camA = pipeline.create(dai.node.ColorCamera)
        camA.setBoardSocket(dai.CameraBoardSocket.RGB)
        camA.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        
        # Define output for generic usage or NN
        # We need a manip or specific preview setup if model expects 300x300 (mobilenet-ssd)
        # MobileNet-SSD expects 300x300.
        
        MODEL_W, MODEL_H = 300, 300
        camA.setPreviewSize(MODEL_W, MODEL_H)
        camA.setInterleaved(False)
        camA.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        camA.setFps(15) 
        
        # XLinkOut: Preview (Brightness/Low-res)
        xoutPreview = pipeline.create(dai.node.XLinkOut)
        xoutPreview.setStreamName("preview")
        camA.preview.link(xoutPreview.input)

        if self.enable_ocr:
            # OCR needs high res. We can use video output or still.
            # let's use video output for OCR queue
            camA.setVideoSize(1920, 1080)
            
            xoutOcr = pipeline.create(dai.node.XLinkOut)
            xoutOcr.setStreamName("ocr_rgb")
            camA.video.link(xoutOcr.input)
        else:
             # Scene RGB (generic) - reuse video output but smaller?
             camA.setVideoSize(640, 480)
             xoutScene = pipeline.create(dai.node.XLinkOut)
             xoutScene.setStreamName("scene_rgb")
             camA.video.link(xoutScene.input)

        if self.enable_yolo:
            try:
                # Using MobileNet Detection Network
                nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
                nn.setConfidenceThreshold(0.5)
                nn.setBlobPath("mobilenet-ssd.blob")
                nn.setNumInferenceThreads(2)
                
                # Manual Label Map for MobileNet-SSD (VOC)
                self.label_map = [
                    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
                    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
                ]
                
                camA.preview.link(nn.input)
                
                xoutYoloRgb = pipeline.create(dai.node.XLinkOut)
                xoutYoloRgb.setStreamName("yolo_rgb")
                nn.passthrough.link(xoutYoloRgb.input)
                
                xoutDet = pipeline.create(dai.node.XLinkOut)
                xoutDet.setStreamName("nn")
                nn.out.link(xoutDet.input)
                
            except Exception as e:
                logger.warning(f"Failed to create NN node: {e}")
                self.enable_yolo = False

        return pipeline
