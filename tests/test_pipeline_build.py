
import unittest
import sys
import os
import depthai as dai

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import walkingPal

class TestPipelineConstruction(unittest.TestCase):
    def test_build_pipeline_basic(self):
        """Test building the pipeline with default settings."""
        try:
            print("Attempting to build pipeline...")
            pipeline, _, _, _, _, _, _ = walkingPal.build_pipeline_and_queues(
                fps_depth=15.0,
                confidence=200,
                lr_check=True,
                extended_disparity=False,
                subpixel=False,
                enable_yolo=False,
                yolo_fps=4.0,
                enable_ocr=False,
                ocr_fps=1.0,
                ocr_size=(1280, 720)
            )
            print("Pipeline built successfully.")
            # Helper to inspect internal objects if we could access them.
            # checks passed.
            self.assertIsInstance(pipeline, dai.Pipeline)
        except AttributeError as e:
            self.fail(f"Pipeline build failed with AttributeError (deprecated/missing API?): {e}")
        except Exception as e:
            self.fail(f"Pipeline build failed with unexpected error: {e}")

    def test_build_pipeline_full(self):
        """Test building pipeline with YOLO and OCR enabled."""
        try:
            print("Attempting to build full pipeline (YOLO+OCR)...")
            # Note: This might try to download blobs if not mocked, but build_pipeline usually just sets up nodes.
            # However, walkingPal code: 
            # det = pipeline.create(dai.node.DetectionNetwork).build(camA, dai.NNModelDescription("yolov6-nano"))
            # This .build(...) might behave differently.
            
            pipeline, _, _, _, _, _, _ = walkingPal.build_pipeline_and_queues(
                fps_depth=15.0,
                confidence=200,
                lr_check=True,
                extended_disparity=False,
                subpixel=False,
                enable_yolo=True,
                yolo_fps=4.0,
                enable_ocr=True,
                ocr_fps=1.0,
                ocr_size=(1280, 720)
            )
            print("Full pipeline built successfully.")
            self.assertIsInstance(pipeline, dai.Pipeline)
        except AttributeError as e:
            self.fail(f"Full pipeline build failed with AttributeError: {e}")
        except Exception as e:
            self.fail(f"Full pipeline build failed with error: {e}")

if __name__ == '__main__':
    unittest.main()
