# WalkingPal - Smart Navigation Assistant for the Visually Impaired

WalkingPal is a Python-based assistive application that uses an **OAK-D (Luxonis)** depth camera to provide real-time navigation feedback, scene description, and object recognition for blind and visually impaired users.

## 🚀 Key Features

*   **Virtual Cane**: Detects obstacles in Left, Center, and Right zones using stereo depth.
*   **Hazard Detection**: Identifies critical hazards like **Drop-offs** (stairs down/cliffs), **Stairs** (up), and **Potholes**.
*   **Hybrid Intelligence (3-Tier)**:
    1.  **Online Tier**: Uses **Gemini 2.0 Flash** / **Qwen** for detailed analysis (requires internet).
    2.  **Local Smart Tier**: Falls back to **Moondream2 (Local VLM)** on-device for high-quality object naming without internet (requires only CPU/GPU).
    3.  **Offline Tier**: Uses on-camera **MobileNet-SSD** for ultra-fast basic object detection (Person, Chair, Bottle).
*   **Smart Audio (Sighted Guide Mode)**: Reduces audio clutter. Bundles object names with navigation instructions (e.g., "Chair ahead, Go Left") and supports **Pre-emptive Interruption** to ensure urgent warnings cut off earlier messages.
*   **Resilient Fallback**: Automatically switches between Online, Local, and Offline modes based on connectivity and API health.
*   **OCR (Text Reading)**: reads signs and text in the environment using a hybrid Tesseract/EasyOCR engine.
*   **Validation Logging**: Records synchronized video and metadata for offline validation and debugging.

---

## 🛠️ Technical Architecture

### 1. Hardware Abstraction (`cameras/oak_d.py`)
The application interacts with the OAK-D camera via the `depthai` library (pinned to version `<3.0` for stability).
*   **Pipeline**:
    *   **Stereo Depth**: 640x400 resolution, HIGH_DENSITY preset for maximum detail.
    *   **Color Camera**: 1080p for user viewing/OCR, 300x300 preview for AI models.
    *   **On-Camera AI**: Runs `MobileNet-SSD` (YOLO) directly on the OAK-D VPU for low-latency object detection (Person, Dog, Chair, etc.).
    *   **IMU**: Fuses accelerometer and gyroscope data to detect camera tilt, ensuring accurate floor plane projection.

### 2. Core Logic (`walkingPal.py`)
This is the main event loop. It fetches frames and routes them to various processors:

#### A. Obstacle Detection (Virtual Cane)
The depth map is divided into three vertical ROI (Region of Interest) bands representing Left, Center, and Right.
*   **Safety Zone**: Objects within `1.2 meters` trigger a "Blocked" state.
*   **Clear Zone**: Objects beyond `2.0 meters` are considered clear.
*   **Feedback**: Spatial audio tones and directional speech instructions ("Go Left", "Stop").

#### B. Hazard Detection Algorithms
*   **Drop-offs**: Analyzes the ratio of "invalid" (black) pixels in the floor area. Requires **>5% density of invalid pixels** within the critical zone to trigger, filtering out sensor noise.
*   **Stairs**: Uses vertical edge detection on the depth map. A sequence of "steps" (regular jumps in depth) triggers "Stairs Up" or "Stairs Down".
*   **Potholes**: Surface roughness analysis. It looks for "dips" (pixels significantly deeper than the median neighborhood) in the immediate ground plane.

#### C. Smart Audio Filter & Controller
*   **Interruption**: The `AudioController` actively manages the mixer channels. If a new message arrives, it stops the current one immediately to ensure the user hears the most relevant update.
*   **Bundling**: Messages are constructed as single utterances ("Table ahead. Stop.") to prevent self-interruption.

### 3. Scene Understanding (`scene_describer.py` & `local_describer.py`)
WalkingPal uses a sophisticated **Arbitration Logic**:
1.  **Check Online**: If internet is up, tries OpenRouter APIs (Gemini/Qwen).
2.  **Fallback to Local VLM**: If APIs fail (429/404/Offline), it instantiates `LocalDescriber` which runs the **Moondream2** model locally via PyTorch/Transformers.
3.  **Fallback to Offline**: If Moondream is disabled or busy, it uses the OAK-D's built-in MobileNet detections.

### 4. Text Recognition (OCR)
A hybrid engine provides the best trade-off between speed and accuracy:
1.  **Tesseract**: Fast, good for printed block text.
2.  **EasyOCR**: AI-based, slower but better for handwritten or distinctive text.
3.  **Auto Mode**: Tries Tesseract first; if confidence is low, transparently switches to EasyOCR.

---

## 📦 Installation

Requirements:
*   Python 3.10+
*   Dependencies: `depthai<3.0`, `opencv-python`, `numpy`, `pygame`, `requests`, `transformers`, `torch`, `einops`

```bash
pip install -r requirements.txt
python install.py  # Checks env and downloads small models
```

**First Run**:
When you first run with `--enable_local_vlm`, the application will download the Moondream2 model (~3GB) from Hugging Face.

## 🎮 Usage

**Standard Navigation (Auto-Configured)**:
```bash
python3 launch.py
```
*Note: `launch.py` automatically enables Recording, Logging, and Local VLM.*

**Manual Launch**:
```bash
python3 walkingPal.py --enable_local_vlm --record
```

## ⚠️ Safety Disclaimer
This is an **assistive prototype**, not a safety-certified medical device. It relies on camera data which can be unreliable on transparent surfaces (glass), highly reflective floors, or in absolute darkness. **Always use a white cane or guide dog.**
