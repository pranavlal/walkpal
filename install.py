#!/usr/bin/env python3
"""
Universal installer for WalkingPal (Windows + Linux).

Creates a local venv, installs deps, runs runtime checks:
- Python version
- pip works
- depthai import + version
- OpenCV import
- numpy import
- optional OCR engines import (pytesseract / easyocr)
- optional: probe for OAK device via depthai

Usage:
  python install.py
  python install.py --with-easyocr
  python install.py --probe-device
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

MIN_PY = (3, 9)

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"

REQ_CORE = ROOT / "requirements-core.txt"
REQ_FULL = ROOT / "requirements.txt"
REQ_OCR_EXTRA = ROOT / "requirements-ocr-extra.txt"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def venv_python_path() -> Path:
    if platform.system().lower().startswith("win"):
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def venv_pip_cmd() -> list[str]:
    return [str(venv_python_path()), "-m", "pip"]


def ensure_python_version() -> None:
    if sys.version_info < MIN_PY:
        raise SystemExit(f"Python {MIN_PY[0]}.{MIN_PY[1]}+ required. You have {sys.version.split()[0]}.")


def create_venv(force: bool = False) -> None:
    py_bin = venv_python_path()
    
    # Check if venv exists but is broken (dir exists, binary missing)
    if VENV_DIR.exists() and not py_bin.exists():
        print(f"Venv directory exists but python binary missing. Recreating...")
        force = True

    if force and VENV_DIR.exists():
        print(f"Removing existing venv: {VENV_DIR}")
        shutil.rmtree(VENV_DIR)

    if not VENV_DIR.exists():
        print(f"Creating venv at: {VENV_DIR}")
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    else:
        print(f"Venv already exists: {VENV_DIR}")


def install_requirements(with_easyocr: bool) -> None:
    py = venv_python_path()
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    # Prefer split files if present, otherwise requirements.txt
    if REQ_CORE.exists():
        run(venv_pip_cmd() + ["install", "-r", str(REQ_CORE)])
        if with_easyocr and REQ_OCR_EXTRA.exists():
            run(venv_pip_cmd() + ["install", "-r", str(REQ_OCR_EXTRA)])
        elif with_easyocr and REQ_FULL.exists():
            run(venv_pip_cmd() + ["install", "-r", str(REQ_FULL)])
    elif REQ_FULL.exists():
        run(venv_pip_cmd() + ["install", "-r", str(REQ_FULL)])
    else:
        raise SystemExit("No requirements file found. Create requirements-core.txt or requirements.txt.")


def runtime_checks(probe_device: bool) -> None:
    py = venv_python_path()

    checks = r"""
import sys, platform
print("Python:", sys.version)
print("Platform:", platform.platform())

import numpy as np
print("numpy:", np.__version__)

import cv2
print("opencv:", cv2.__version__)

import depthai as dai
print("depthai:", getattr(dai, "__version__", "unknown"))

import torch
print("torch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

import transformers
print("transformers:", transformers.__version__)

# OCR wrappers
try:
    import pytesseract
    print("pytesseract:", pytesseract.get_tesseract_version() if hasattr(pytesseract, "get_tesseract_version") else "ok")
except Exception as e:
    print("pytesseract: NOT OK:", e)

try:
    import easyocr
    print("easyocr: ok")
except Exception as e:
    print("easyocr: not installed/ok:", e)

if {probe}:
    try:
        devs = dai.Device.getAllAvailableDevices()
        print("Devices:", devs)
        if not devs:
            print("WARNING: No OAK devices detected. Check USB + permissions.")
    except Exception as e:
        print("Device probe failed:", e)
""".format(probe="True" if probe_device else "False")

    run([str(py), "-c", checks])


def print_next_steps(with_easyocr: bool) -> None:
    py = venv_python_path()
    print("\n=== Installed. Next steps ===")
    print("Activate venv (optional):")
    if platform.system().lower().startswith("win"):
        print(rf"  {VENV_DIR}\Scripts\activate")
    else:
        print(f"  source {VENV_DIR}/bin/activate")

    print("\nRun WalkingPal with all features enabled:")
    print(f"  {str(py)} launch.py")
    
    print("\nOr manually:")
    cmd = [
        str(py),
        "walkingPal.py",
        "--enable_yolo",
        "--enable_potholes",
        "--enable_ocr",
        "--ocr_engine",
        "auto" if with_easyocr else "tesseract",
        "--ocr_lang",
        "eng+hin",
        "--ocr_fps",
        "1.25",
    ]
    print("  " + " ".join(cmd))
    print("\nNOTE: Tesseract engine itself must be installed system-wide for OCR to work.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-easyocr", action="store_true", help="Install EasyOCR + torch (heavy).")
    ap.add_argument("--probe-device", action="store_true", help="Probe for OAK device after install.")
    ap.add_argument("--force", action="store_true", help="Delete and recreate venv.")
    args = ap.parse_args()

    ensure_python_version()
    create_venv(force=args.force)
    install_requirements(with_easyocr=args.with_easyocr)
    ensure_model_blob()
    system_checks()
    runtime_checks(probe_device=args.probe_device)
    print_next_steps(with_easyocr=args.with_easyocr)

def ensure_model_blob() -> None:
    print("\n>>> Downloading Models (YOLOv8, MiDaS, MobileNet)...")
    py = venv_python_path()
    
    # Run the dedicated download script
    try:
        run([str(py), "download_models.py"], cwd=ROOT)
    except Exception:
        print("[WARN] Failed to run download_models.py. Models might be missing.")



def system_checks() -> None:
    print("\n=== System Checks ===")
    
    # Check for Tesseract
    tess = shutil.which("tesseract")
    if tess:
        print(f"[OK] Tesseract found: {tess}")
    else:
        print("[WARN] 'tesseract' binary not found in PATH. OCR will not work unless installed system-wide.")
        if platform.system().lower().startswith("linux"):
            print("       Debian/Ubuntu: sudo apt install tesseract-ocr")
            print("       Arch Linux:    sudo pacman -S tesseract")
            print("       Fedora/RHEL:   sudo dnf install tesseract")
        elif platform.system().lower().startswith("win"):
            print("       Install from: https://github.com/UB-Mannheim/tesseract/wiki")

    # Check for eSpeak (TTS requirement for Linux)
    if platform.system().lower().startswith("linux"):
        espeak = shutil.which("espeak") or shutil.which("espeak-ng")
        if espeak:
             print(f"[OK] eSpeak found: {espeak}")
        else:
             print("[WARN] 'espeak' or 'espeak-ng' not found. TTS may not work.")
             print("       Debian/Ubuntu: sudo apt install espeak-ng")
             print("       Arch Linux:    sudo pacman -S espeak-ng")
             print("       Fedora:        sudo dnf install espeak-ng")

    # Check for Pygame Mixer (SDL Dependencies)
    py = venv_python_path()
    try:
        run([str(py), "-c", "import pygame.mixer; pygame.mixer.init()"], cwd=ROOT)
        print(f"[OK] Pygame mixer init success.")
    except Exception:
        print("[WARN] Pygame mixer failed to init. Missing system dependencies?")
        if platform.system().lower().startswith("linux"):
             print("       Debian/Ubuntu/Raspberry Pi: sudo apt install libsdl2-mixer-2.0-0")
             print("       Arch:                       sudo pacman -S sdl2_mixer")
             print("       Fedora:                     sudo dnf install SDL2_mixer")

    # Check for Linux UDEV rules (OAK-D requirement)
    if platform.system().lower().startswith("linux"):
        rules = Path("/etc/udev/rules.d/80-movidius.rules")
        if rules.exists():
            print(f"[OK] OAK-D UDEV rules found: {rules}")
        else:
            print("[WARN] OAK-D UDEV rules missing. Camera might not be detected.")
            print("       Run commands below to fix user permissions:")
            print('       echo \'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"\' | sudo tee /etc/udev/rules.d/80-movidius.rules')
            print('       sudo udevadm control --reload-rules && sudo udevadm trigger')


if __name__ == "__main__":
    main()
