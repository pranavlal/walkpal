#!/usr/bin/env python3
"""
Universal installer for WalkingPal (Windows + Linux).
Creates a local venv, installs deps, and runs runtime checks.

NASA 'Power of Ten' Safety Edition.
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
IS_WIN = platform.system().lower().startswith("win")
REQ_FULL = ROOT / "requirements.txt"

def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)

def venv_python_path() -> Path:
    if IS_WIN: return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"

def create_venv(force: bool = False) -> None:
    if force and VENV_DIR.exists():
        print(f"Removing existing venv: {VENV_DIR}")
        shutil.rmtree(VENV_DIR)
    if not VENV_DIR.exists():
        print(f"Creating venv at: {VENV_DIR}")
        run([sys.executable, "-m", "venv", str(VENV_DIR)])

def install_requirements() -> None:
    py = venv_python_path()
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    if REQ_FULL.exists():
        run([str(py), "-m", "pip", "install", "-r", str(REQ_FULL)])
    else:
        print("[WARN] requirements.txt not found. Please install manualy.")

def runtime_checks(probe_device: bool) -> None:
    py = venv_python_path()
    checks = r"""
import sys, platform, torch, cv2, numpy as np, transformers, openai, google.genai, anthropic, depthai as dai
print(f"Python: {sys.version}")
print(f"OpenCV: {cv2.__version__}")
print(f"Numpy: {np.__version__}")
print(f"PyTorch: {torch.__version__} (CUDA={torch.cuda.is_available()})")
print(f"DepthAI: {getattr(dai, '__version__', 'ok')}")
print(f"OpenAI: {getattr(openai, '__version__', 'ok')}")
print(f"Google GenAI: ok")
print(f"Anthropic: ok")

if {probe}:
    try:
        devs = dai.Device.getAllAvailableDevices()
        print("Devices:", devs)
        if not devs: print("WARNING: No OAK devices detected.")
    except Exception as e: print("Device probe failed:", e)
""".replace("{probe}", "True" if probe_device else "False")
    run([str(py), "-c", checks])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-device", action="store_true", help="Probe for OAK device.")
    ap.add_argument("--force", action="store_true", help="Delete and recreate venv.")
    args = ap.parse_args()

    if sys.version_info < MIN_PY:
        raise SystemExit(f"Python {MIN_PY[0]}.{MIN_PY[1]}+ required.")
        
    create_venv(force=args.force)
    install_requirements()
    
    # Download models if script exists
    if (ROOT / "download_models.py").exists():
        try: run([str(venv_python_path()), "download_models.py"])
        except Exception: print("[WARN] Model download failed.")
        
    runtime_checks(probe_device=args.probe_device)
    print("\n=== WalkingPal Refactored (NASA Standard) Installed ===")
    print(f"Run: {str(venv_python_path())} launch.py")

if __name__ == "__main__":
    main()
