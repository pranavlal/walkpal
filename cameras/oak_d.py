import cv2
import time
import logging
import depthai as dai
import numpy as np
from typing import Optional, Any, Dict, List
from .base import CameraInterface, CameraFrames

logger = logging.getLogger("walkingpal.oakd")

class OakDCamera(CameraInterface):
    def __init__(self, detection_config: Dict[str, Any] = None, enable_potholes=False, enable_ocr=False):
        self.detection_config = detection_config or {}
        self.enable_potholes = enable_potholes
        self.enable_ocr = enable_ocr
        
        # Parse detection config
        self.det_model = self.detection_config.get('model', 'none').lower()
        if self.det_model == 'false': self.det_model = 'none' # Handle legacy boolean mishap if any
        
        self.enable_nn = (self.det_model not in ['none', ''])
        
        self.device = None
        self.pipeline = None
        
        self.q_depth = None
        self.q_det = None 
        self.q_nn_input = None # Generalized input queue (was yolo_rgb)
        self.q_ocr_rgb = None
        self.q_scene_rgb = None
        self.q_imu = None
        self.q_preview = None
        
        self.label_map = self.detection_config.get('labels', [])

    def start(self) -> bool:
        """Initializes the OAK-D pipeline."""
        try:
            self.pipeline = self._build_pipeline()
            
            # Connect to device
            self.device = dai.Device(self.pipeline)
            self.device.startPipeline()
            
            # Create queues
            self.q_depth = self.device.getOutputQueue(name="depth", maxSize=2, blocking=False)
            self.q_preview = self.device.getOutputQueue(name="preview", maxSize=1, blocking=False)
            
            if self.enable_imu:
                try:
                    self.q_imu = self.device.getOutputQueue(name="imu", maxSize=10, blocking=False)
                except Exception:
                    logger.warning("IMU queue not found.")

            if self.enable_nn:
                try:
                    self.q_det = self.device.getOutputQueue(name="nn", maxSize=4, blocking=False)
                    self.q_nn_input = self.device.getOutputQueue(name="nn_input", maxSize=2, blocking=False)
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

            logger.info(f"OAK-D Pipeline started. Detection Model: {self.det_model}")
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
        elif self.q_nn_input:
             # If NN is used but OCR is off, we still get RGB frames from passthrough
             nn_pkt = self.q_nn_input.tryGet()
             if nn_pkt:
                 frames.video = nn_pkt.getCvFrame()
             
        # 3. Scene Frame
        if self.q_scene_rgb:
            s_pkt = self.q_scene_rgb.tryGet()
            if s_pkt:
                frames.scene = s_pkt.getCvFrame()
        
        # 4. Preview
        if self.q_preview:
            p_pkt = self.q_preview.tryGet()
            if p_pkt:
                frames.preview = p_pkt.getFrame()
        
        # Fallback
        if frames.scene is None and frames.video is not None:
             frames.scene = frames.video

        return frames
        
    def get_detections(self):
        """Get NN results. Handles different output formats."""
        if self.q_det:
             det_data = self.q_det.tryGet()
             if det_data:
                 # Standard ImgDetections (MobileNet, YOLO)
                 if hasattr(det_data, 'detections'):
                     return det_data.detections
                 
                 # RT-DETR / RF-DETR / Raw NN Output
                 if self.det_model in ['rtdetr', 'rf-detr']:
                     return self._decode_detr(det_data)
                 elif self.det_model == 'yolov8':
                     return self._decode_yolov8(det_data)
        return []

    def _decode_detr(self, nn_data):
        """
        Host-side decoding for DETR-like models (RT-DETR, RF-DETR).
        Expected output layers: 
        - class scores (logits)
        - bounding boxes (cx, cy, w, h)
        
        Note: Layer names vary by export. We look for typical shapes.
        """
        try:
            # depthai.NNData
            # nn_data.getLayerNames()
            # nn_data.getLayerFp16(name)
            
            layers = nn_data.getAllLayerNames()
            conf_layer = None
            box_layer = None
            
            # Heuristic to find layers
            # Scores: [1, 300, 80] or similar
            # Boxes: [1, 300, 4]
            
            for name in layers:
                tensor = np.array(nn_data.getLayerFp16(name))
                # RT-DETR usually [300, 80] flattened or similar. 
                # OAK-D NeuralNetwork output is often flattened 1D list in getLayerFp16.
                # Use getLayer(name).to_numpy() ? No, standard is getLayerFp16 returns list.
                # We need to know shape. 
                # If we don't know shape, we can't reshape. 
                # HOWEVER: Roboflow RF-DETR specific export often names layers 'output1', 'output2'.
                
                # Assume standard RT-DETR shapes for now: 300 queries.
                # Box layer length = 300 * 4 = 1200
                if len(tensor) == 1200: 
                    box_layer = tensor.reshape((300, 4))
                # Score layer length = 300 * num_classes. 
                # If we have 80 classes -> 24000. 
                # If we don't match exactly, we might fail.
                elif len(tensor) > 1200:
                    conf_layer = tensor
            
            if box_layer is None or conf_layer is None:
                return []
                
            num_queries = 300
            num_classes = len(conf_layer) // num_queries
            conf_layer = conf_layer.reshape((num_queries, num_classes))
            
            detections = []
            confidence_thresh = self.detection_config.get('confidence', 0.5)
            
            for i in range(num_queries):
                # Softmax or Sigmoid? RT-DETR usually Sigmoid for focal loss.
                # But let's check max score.
                scores = conf_layer[i]
                # scores often raw logits. 
                # simple validation:
                # if raw, range is -inf to inf. 
                # sigmoid: 0 to 1.
                
                # Fast exp/softmax
                # For efficiency, find max first
                class_id = np.argmax(scores)
                score = scores[class_id]
                
                # Simple sigmoid approximation or just use raw if it's already prob (some exports do)
                # If score > 1.0, it's a logit.
                if score > 1.0 or score < 0.0:
                    # Apply sigmoid
                    score = 1 / (1 + np.exp(-score))
                
                if score > confidence_thresh:
                    # Decode box
                    # cx, cy, w, h are normalized 0..1 usually
                    cx, cy, w, h = box_layer[i]
                    
                    xmin = cx - w/2
                    ymin = cy - h/2
                    xmax = cx + w/2
                    ymax = cy + h/2
                    
                    # Create ImgDetection-like object
                    det = dai.ImgDetection()
                    det.label = int(class_id)
                    det.confidence = float(score)
                    det.xmin = max(0.0, float(xmin))
                    det.ymin = max(0.0, float(ymin))
                    det.xmax = min(1.0, float(xmax))
                    det.ymax = min(1.0, float(ymax))
                    
                    detections.append(det)
                    
            return detections
            
            return detections
            
        except Exception as e:
            # Log only occasionally to avoid spam
            if time.time() % 5.0 < 0.1:
                logger.warning(f"RF-DETR decoding failed: {e}")
            return []

    def _decode_yolov8(self, nn_data):
        """
        Host-side decoding for YOLOv8 raw output (NeuralNetwork).
        Expected output: (1, 84, 8400) or similar.
        84 channels: 4 box coords (cx, cy, w, h) + 80 classes.
        """
        try:
            layers = nn_data.getAllLayerNames()
            output_tensor = None
            
            # Find the main output layer
            for name in layers:
                layer_data = np.array(nn_data.getLayerFp16(name))
                # Heuristic: 84 * 8400 = 705600 floats (or 640x352 specific count)
                # YOLOv8n 640x640 -> 8400 anchors.
                # YOLOv8n 640x352 -> 8400? No, stride 8,16,32. 
                # 640/8=80, 352/8=44 -> 3520
                # 640/16=40, 352/16=22 -> 880
                # 640/32=20, 352/32=11 -> 220
                # Total = 3520+880+220 = 4620 anchors.
                # So length should be 4620 * 84 = 388080.
                # We can just reshape based on channels=84.
                
                if len(layer_data) % 84 == 0:
                     num_anchors = len(layer_data) // 84
                     output_tensor = layer_data.reshape((1, 84, num_anchors))
                     break
            
            if output_tensor is None:
                return []
                
            # Transpose to (Num_Anchors, 84) -> (4620, 84)
            # output_tensor shape: (1, 84, 4620)
            data = output_tensor[0].transpose() # (4620, 84)
            
            # Extract basic info
            # 0,1,2,3 -> cx, cy, w, h
            # 4..83 -> class scores
            
            boxes = data[:, 0:4]
            scores = data[:, 4:]
            
            # Find max score and class ID for each anchor
            class_ids = np.argmax(scores, axis=1)
            confidences = np.max(scores, axis=1)
            
            # Thresholding
            thresh = self.detection_config.get('confidence', 0.5)
            mask = confidences > thresh
            
            if not np.any(mask):
                return []
                
            boxes = boxes[mask]
            confidences = confidences[mask]
            class_ids = class_ids[mask]
            
            # Prepare for NMS
            # NMSBoxes expects [x, y, w, h] (top-left) in pixels (or normalized).
            # Output from YOLOv8 is cx, cy, w, h.
            # We need valid boxes list for cv2
            
            nms_boxes = []
            nms_confidences = []
            
            input_w, input_h = self.detection_config.get('input_size', (640, 352))
            
            for i in range(len(boxes)):
                cx, cy, w, h = boxes[i]
                
                # Check if normalized (0..1) or pixels
                # If any value > 1.0, likely pixels.
                # But boxes can be small. Safest is to normalize if max(boxes) > 1.0 globally?
                # Usually pixels.
                
                # Convert to top-left
                x = cx - w/2
                y = cy - h/2
                
                nms_boxes.append([x, y, w, h])
                nms_confidences.append(float(confidences[i]))

            # NMS
            indices = cv2.dnn.NMSBoxes(nms_boxes, nms_confidences, thresh, 0.5)
            
            results = []
            
            # NMSBoxes returns a tuple of lists or list of indices
            if len(indices) > 0:
                 # Flatten if needed
                 indices = indices.flatten()
                 
                 for idx in indices:
                     box = nms_boxes[idx]
                     score = nms_confidences[idx]
                     cls_id = class_ids[idx]
                     
                     x, y, w, h = box
                     
                     # Normalize for ImgDetection (0..1)
                     # Assuming pixel coords if input_w input_h provided
                     # Note: if model output is already normalized, we check:
                     # If max value < 1.1, assume normalized.
                     
                     norm_x = x / input_w if input_w > 0 else x
                     norm_y = y / input_h if input_h > 0 else y
                     norm_w = w / input_w if input_w > 0 else w
                     norm_h = h / input_h if input_h > 0 else h
                     
                     # Simple check: if we normalized 0.5 (which was already norm) by 640 -> tiny.
                     # Heuristic: if box[2] (w) > 1.0, it was pixels.
                     if box[2] < 1.0 and box[3] < 1.0:
                         # Was likely already normalized
                         norm_x, norm_y, norm_w, norm_h = x, y, w, h
                     
                     det = dai.ImgDetection()
                     det.label = int(cls_id)
                     det.confidence = float(score)
                     det.xmin = max(0.0, float(norm_x))
                     det.ymin = max(0.0, float(norm_y))
                     det.xmax = min(1.0, float(norm_x + norm_w))
                     det.ymax = min(1.0, float(norm_y + norm_h))
                     
                     results.append(det)
                     
            return results

        except Exception as e:
             if time.time() % 5.0 < 0.1:
                logger.warning(f"YOLOv8 decoding failed: {e}")
             return []
        
    def get_imu(self):
        pkt = None
        if self.q_imu:
             while self.q_imu.has():
                 pkt = self.q_imu.tryGet()
        return pkt

    def get_intrinsics(self):
        if self.device:
             try:
                 calib = self.device.readCalibration()
                 return calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A)
             except Exception:
                 pass
        return None

    def _build_pipeline(self):
        pipeline = dai.Pipeline()

        fps_depth = 30.0
        confidence = 200
        
        # Stereo Setup
        camL = pipeline.create(dai.node.MonoCamera)
        camL.setBoardSocket(dai.CameraBoardSocket.LEFT)
        camL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        camL.setFps(fps_depth)
        
        camR = pipeline.create(dai.node.MonoCamera)
        camR.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        camR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        camR.setFps(fps_depth)

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.initialConfig.setConfidenceThreshold(confidence)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)
        stereo.setSubpixel(False)
        try:
            stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        except AttributeError: pass

        camL.out.link(stereo.left)
        camR.out.link(stereo.right)

        xoutDepth = pipeline.create(dai.node.XLinkOut)
        xoutDepth.setStreamName("depth")
        stereo.depth.link(xoutDepth.input)
        
        # IMU
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
            
        # Color Camera
        camA = pipeline.create(dai.node.ColorCamera)
        camA.setBoardSocket(dai.CameraBoardSocket.RGB)
        camA.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camA.setInterleaved(False)
        camA.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        camA.setFps(15)

        # NN Setup
        if self.enable_nn:
            self._setup_nn_node(pipeline, camA)
        else:
             # Basic preview if no NN
             camA.setPreviewSize(300, 300)
             
        # Preview Output
        xoutPreview = pipeline.create(dai.node.XLinkOut)
        xoutPreview.setStreamName("preview")
        camA.preview.link(xoutPreview.input)

        if self.enable_ocr:
            camA.setVideoSize(1920, 1080)
            xoutOcr = pipeline.create(dai.node.XLinkOut)
            xoutOcr.setStreamName("ocr_rgb")
            camA.video.link(xoutOcr.input)
        else:
             camA.setVideoSize(640, 480)
             xoutScene = pipeline.create(dai.node.XLinkOut)
             xoutScene.setStreamName("scene_rgb")
             camA.video.link(xoutScene.input)

        return pipeline

    def _setup_nn_node(self, pipeline, camA):
        """Creates the appropriate NN node based on configuration."""
        blob_path = self.detection_config.get('blobs', {}).get(self.det_model, None)
        if not blob_path: 
            logger.warning(f"No blob define for {self.det_model}, trying default naming.")
            blob_path = f"{self.det_model}.blob"

        # Check file existence? DepthAI throws meaningful error if missing.
        
        try:
            nn = None
            # Default input size depends on model but can be overridden
            default_input_sizes = {
                'mobilenet': (300, 300),
                'yolo': (416, 416),
                'yolov8': (640, 352),
                'rtdetr': (640, 640),
                'rf-detr': (640, 640)
            }
            # Use specific config size if provided, else default
            cfg_size = self.detection_config.get('input_size')
            if cfg_size:
                input_size = tuple(cfg_size)
            else:
                input_size = default_input_sizes.get(self.det_model, (300, 300))

            if self.det_model == 'mobilenet':
                nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
                nn.setConfidenceThreshold(self.detection_config.get('confidence', 0.5))
                nn.setBlobPath(blob_path)
                nn.setNumInferenceThreads(2)
                
                # Default labels if not provided
                if not self.label_map:
                    self.label_map = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
                        "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
                        
            elif self.det_model == 'yolo':
                nn = pipeline.create(dai.node.YoloDetectionNetwork)
                nn.setConfidenceThreshold(self.detection_config.get('confidence', 0.5))
                nn.setBlobPath(blob_path)
                nn.setNumInferenceThreads(2)
                
                # Standard OAK-D YOLO defaults (usually v4-tiny or v3-tiny if just "yolo")
                # If using v5/v6/v8 via "yolo" tag, might need tuning.
                logger.info(f"Configuring standard YOLO node with input {input_size}")
                
            elif self.det_model == 'yolov8':
                # Use Generic NeuralNetwork for YOLOv8 (Output: 84x8400 raw)
                # YoloDetectionNetwork requires specific metadata often missing in raw blobs
                nn = pipeline.create(dai.node.NeuralNetwork)
                nn.setBlobPath(blob_path)
                nn.setNumInferenceThreads(2)
                
                # Input size is handled by preview link size
                logger.info(f"Configuring YOLOv8 node (NeuralNetwork) with input {input_size}. Host-side decoding active.")
                
            elif self.det_model in ['rtdetr', 'rf-detr']:
                # RT-DETR / RF-DETR
                # If checking for "RF-DETR", users usually mean RT-DETR.
                # Use Generic NeuralNetwork or YoloDetectionNetwork if exported as YOLO.
                # We'll use MobileNet as a fallback container or NeuralNetwork ?
                # NeuralNetwork is safest for raw blobs.
                nn = pipeline.create(dai.node.NeuralNetwork)
                nn.setBlobPath(blob_path)
                input_size = (640, 640) # RT-DETR/RF-DETR common size
                logger.info("Configuring RF/RT-DETR (NeuralNetwork node). Output parsing active.")
            
            else:
                 logger.error(f"Unknown detection model: {self.det_model}")
                 return

            if nn:
                camA.setPreviewSize(input_size)
                camA.preview.link(nn.input)
                
                # Outputs
                xoutYoloRgb = pipeline.create(dai.node.XLinkOut)
                xoutYoloRgb.setStreamName("nn_input")
                nn.passthrough.link(xoutYoloRgb.input)
                
                xoutDet = pipeline.create(dai.node.XLinkOut)
                xoutDet.setStreamName("nn")
                nn.out.link(xoutDet.input)
                
        except Exception as e:
            logger.error(f"Failed to create NN node ({self.det_model}): {e}")
            self.enable_nn = False
