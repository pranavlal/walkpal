import os
import json
import time
import cv2
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional
import shutil

class SessionLogger:
    def __init__(self, base_dir: str = "logs", record_depth: bool = False):
        self.base_dir = base_dir
        self.record_depth = record_depth
        
        # Create session directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(base_dir, f"session_{timestamp}")
        self.img_dir = os.path.join(self.session_dir, "images")
        self.depth_dir = os.path.join(self.session_dir, "depth")
        
        os.makedirs(self.img_dir, exist_ok=True)
        if self.record_depth:
            os.makedirs(self.depth_dir, exist_ok=True)
            
        self.meta_file = os.path.join(self.session_dir, "metadata.jsonl")
        self.file_handle = open(self.meta_file, "a", encoding="utf-8")
        
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Logger")
        self.frame_count = 0
        self.lock = threading.Lock()
        
    def log(self, 
            frame: np.ndarray, 
            depth: Optional[np.ndarray], 
            metadata: Dict[str, Any]):
        """
        Log a single frame and its metadata.
        This is non-blocking (offloaded to worker thread).
        """
        # Copy data to avoid modification during write
        # Frame and depth can be large, but we need copies if they might change
        frame_copy = frame.copy() if frame is not None else None
        depth_copy = depth.copy() if (depth is not None and self.record_depth) else None
        
        # Snapshot frame ID
        with self.lock:
            self.frame_count += 1
            fid = self.frame_count
            
        # Submit to worker
        self.executor.submit(self._write_task, fid, frame_copy, depth_copy, metadata)

    def _write_task(self, fid: int, frame, depth, metadata):
        try:
            timestamp = time.time()
            img_filename = f"frame_{fid:06d}.jpg"
            depth_filename = f"depth_{fid:06d}.png"
            
            # Save Image
            if frame is not None:
                path = os.path.join(self.img_dir, img_filename)
                cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                
            # Save Depth
            if depth is not None:
                path = os.path.join(self.depth_dir, depth_filename)
                # Save as 16-bit PNG (mm)
                cv2.imwrite(path, depth.astype(np.uint16))
            
            # Update metadata with file paths
            log_entry = {
                "frame_id": fid,
                "timestamp": timestamp,
                "image_file": img_filename if frame is not None else None,
                "depth_file": depth_filename if depth is not None else None,
                **metadata
            }
            
            # Write JSONL
            # Use specific separator for atomic append safety in some contexts, but file handle is usually fine in one process
            json_str = json.dumps(log_entry)
            self.file_handle.write(json_str + "\n")
            self.file_handle.flush()
            
        except Exception as e:
            print(f"Logging error: {e}")

    def close(self):
        self.executor.shutdown(wait=True)
        self.file_handle.close()
