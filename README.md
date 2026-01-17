# WalkingPal - Smart Navigation Assistant for the Visually Impaired

WalkingPal is a Python-based assistive application that uses an **OAK-D (Luxonis)** depth camera to provide real-time navigation feedback, scene description, and object recognition for blind and visually impaired users.

## 🚀 Key Features

*   **Virtual Cane**: Detects obstacles in Left, Center, and Right zones using stereo depth.
*   **Hazard Detection**: Identifies critical hazards like **Drop-offs** (stairs down/cliffs), **Stairs** (up), and **Potholes**.
*   **Smart Audio (Sighted Guide Mode)**: Reduces audio clutter by only speaking when the scene changes or a hazard appears. Names specific obstacles (e.g., "Chair ahead") using AI.
*   **Scene Description**: Uses **Multimodal LLMs** (Gemini 2.0 Flash, Llama 3.2 Vision) via OpenRouter to describe the scene on demand (or upon significant scene changes).
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
*   **Drop-offs**: Analyzes the ratio of "invalid" (black) pixels in the floor area. If the floor suddenly disappears (high invalid ratio) or becomes impossibly distant, a drop-off is flagged.
    *   *Dynamic Threshold*: Adjusts sensitivity based on camera pitch (IMU).
*   **Stairs**: Uses vertical edge detection on the depth map. A sequence of "steps" (regular jumps in depth) triggers "Stairs Up" or "Stairs Down".
*   **Potholes**: Surface roughness analysis. It looks for "dips" (pixels significantly deeper than the median neighborhood) in the immediate ground plane.

#### C. Smart Audio Filter (Sighted Guide)
To prevent "audio fatigue", the system waits for state changes before speaking:
1.  **Hazards First**: Danger signals (Drop-off, Stairs) always interrupt.
2.  **State Change**: "Clear" -> "Blocked" triggers a message. "Blocked" -> "Blocked" (same condition) is silenced.
3.  **Naming**: If an object is blocking the path, the system correlates it with YOLO detections to say "Chair ahead" instead of "Obstacle ahead".

### 3. Scene Understanding (`scene_describer.py`)
*   **Trigger**: Detects significant visual changes (histogram shift) to proactively describe new environments.
*   **Model Fallback**: Uses a resilience chain. If the primary model (Gemini 2.0 Flash) fails with a 429 Rate Limit, it instantly falls back to Llama 3.2 Vision, ensuring high reliability.
*   **Privacy**: Images are base64 encoded and sent over encrypted HTTPS. No local storage unless `--record` is used.

### 4. Text Recognition (OCR)
A hybrid engine provides the best trade-off between speed and accuracy:
1.  **Tesseract**: Fast, good for printed block text.
2.  **EasyOCR**: AI-based, slower but better for handwritten or distinctive text.
3.  **Auto Mode**: Tries Tesseract first; if confidence is low, transparently switches to EasyOCR.

---

## 📦 Installation

Requirements:
*   Python 3.10+
*   Dependencies: `depthai<3.0`, `opencv-python`, `numpy`, `pygame`, `requests`

```bash
pip install -r requirements.txt
python download_model.py  # (Optional) Updates local YOLO blob
```

## 🎮 Usage

**Standard Navigation**:
```bash
python3 walkingPal.py
```

**Developer/Debug Mode** (Visualizes depth map and AI overlays):
```bash
python3 walkingPal.py --monitor --debug --enable_yolo --enable_ocr
```

**Validation Recording**:
```bash
python3 walkingPal.py --record --record_fps 2.0
```

## ⚠️ Safety Disclaimer
This is an **assistive prototype**, not a safety-certified medical device. It relies on camera data which can be unreliable on transparent surfaces (glass), highly reflective floors, or in absolute darkness. **Always use a white cane or guide dog.**
