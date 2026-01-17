from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Any
import numpy as np

@dataclass
class CameraFrames:
    """Standard container for camera frames."""
    depth: Optional[np.ndarray] = None          # Depth map (metric or relative)
    video: Optional[np.ndarray] = None          # RGB for display/OCR
    scene: Optional[np.ndarray] = None          # RGB for SceneDescriber (optimized?)
    preview: Optional[np.ndarray] = None        # Low-res frame for brightness/preview
    
    # Metadata for depth interpretation
    timestamp: float = 0.0

class CameraInterface(ABC):
    """Abstract base class for all camera implementations."""

    @abstractmethod
    def start(self) -> bool:
        """Initialize and start the camera stream. Returns True if successful."""
        pass

    @abstractmethod
    def stop(self):
        """Release resources and stop the stream."""
        pass

    @abstractmethod
    def get_frames(self) -> CameraFrames:
        """
        Retrieve the latest frames in a non-blocking way.
        Returns a CameraFrames object. Fields may be None if no new data.
        """
        pass

    @abstractmethod
    def get_intrinsics(self) -> Any:
        # TODO: Define standard intrinsic structure if needed generically.
        # For now, may return specific camera object or intrinsics matrix.
        pass
