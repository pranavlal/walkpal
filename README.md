# WalkingPal - Advanced Blind Navigation Assistant

WalkingPal is an AI-powered perception system for the OAK-D camera, designed to assist blind and visually impaired users navigating their environment. It combines depth-based obstacle avoidance with advanced scene understanding using Online (Gemini/GPT-4) and Local (MiniCPM/Moondream) Vision Language Models.

## Features
- **Depth-First Navigation**: Real-time detection of obstacles, drop-offs, stairs, and potholes (30fps+).
- **Hybrid AI Intelligence**:
  - **Online**: Uses Gemini 2.0 Flash / GPT-4o for detailed scene analysis when online.
  - **Local High-Accuracy**: Auto-loads **MiniCPM-V 2.0** if a GPU (>6GB VRAM) is detected.
  - **Local Efficient**: Falls back to **Moondream2** on low-power devices (CPU/Edge).
  - **Fast Hazard**: Parallel YOLOv6-nano for instant detection of common hazards (people, cars, dogs).
  - **Smart Power Saving**: Uses **Scene Change Detection** to prevent wasteful AI processing when standing still.
- **Spatial Audio**: 3D binaural audio feedback to intuitively locate obstacles.
- **Accessible Debugging GUI**: Optional, cross-platform GUI with **WCAG 2.2 compliant** accessibility features, high-contrast modes, and live screen-reader descriptions.
- **Mission-Critical Reliability**: Refactored to adhere to **NASA JPL "Power of Ten"** safety rules for deterministic execution and fault tolerance.
- **OCR (Text Reading)**: Advanced sign and text recognition using a hybrid Tesseract/EasyOCR engine.
- **Internationalization**: Full support for English and Hindi.

## System Requirements
- **Camera**: Luxonis OAK-D (OAK-D, OAK-D Lite, or OAK-D Pro).
- **OS**: Linux (Ubuntu 20.04/22.04), Windows 10/11, or macOS (Intel/Apple Silicon).
- **Hardware**:
  - Minimum: Raspberry Pi 4 / Standard Laptop (Use Moondream2).
  - Recommended for MiniCPM: Laptop with NVIDIA GPU (6GB+ VRAM) or Apple Silicon (M1/M2/M3).

*   **Smart Audio (Sighted Guide Mode)**: Reduces audio clutter. Bundles object names with navigation instructions (e.g., "Chair ahead, Go Left") and supports **Pre-emptive Interruption** to ensure urgent warnings cut off earlier messages.
*   **Resilient Fallback**: Automatically switches between Online, Local, and Offline modes based on connectivity and API health.
*   **OCR (Text Reading)**: Reads signs and text in the environment using a hybrid Tesseract/EasyOCR engine. Supports `--enable_ocr`.
*   **Validation Logging**: Records synchronized video and metadata for offline validation and debugging.
*   **One-Click Installers**: Native setup scripts for Windows and Linux to automate complex environment configuration.

---

### Windows (One-Click)
Right-click `WalkingPal_Setup.ps1` and select **Run with PowerShell**. This automates Python detection, virtual environment creation, and Desktop Shortcut generation.

### Linux (One-Click)
```bash
chmod +x walkingpal_setup.sh
./walkingpal_setup.sh
```
Installs system dependencies (`libgl1`, `tesseract`), sets up **UDEV hardware rules** for OAK-D, and creates a Desktop Entry. Supported on **Ubuntu, Fedora, and Arch Linux**.

### Legacy/Manual (macOS/Terminal)
```bash
python install.py  # Universal installer
./install_mac.sh   # macOS specific
```

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

**Accessible Debugging GUI**:
```bash
python3 walkingPal.py --gui
```
*Launches the optional WCAG 2.2 compliant visual console for sighted assistants or developers.*

## Running over SSH (Headless)
If you are running remotely (e.g., sticking the Pi in a backpack and SSHing in), use the `--headless` flag to prevent the app from trying to open a window:

```bash
# 1. Install (First run only)
python3 install.py

# 2. Launch
python3 launch.py --headless
```

The app will still generate audio (ensure your speakers are connected to the Pi/Laptop) but will not crash due to missing display.

**Manual Launch**:
```bash
python3 walkingPal.py --enable_local_vlm --record
```

## ⚠️ Safety Disclaimer
This is an **assistive prototype**, not a safety-certified medical device. It relies on camera data which can be unreliable on transparent surfaces (glass), highly reflective floors, or in absolute darkness. **Always use a white cane or guide dog.**
