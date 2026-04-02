#!/usr/bin/env python3
"""
Universal launch script for WalkingPal.
Detects OS, finds venv, checks setup, and launches walkingPal.py with features enabled.
"""

import sys
import os
import platform
import subprocess
from pathlib import Path

# Configuration for "All Features Enabled"
LAUNCH_ARGS = [
    "--enable_yolo",
    "--enable_potholes",
    "--enable_ocr",
    "--ocr_engine", "tesseract",  # Default to Tesseract (CPU) to avoid GPU OOM/Context clash
    "--ocr_lang", "eng+hin",
    "--speak_every_s", "2.5",     # Less chatter (was 1.1)
    "--hazard_cooldown_s", "5.0", # Less hazard repetition
    "--record",      # Enable SessionLogger (JSONL + Images)
    "--record_fps", "2.0",
    "--enable_local_vlm", # Enable Moondream2/MiniCPM-V fallback
]

def main():
    # 1. Determine paths
    root_dir = Path(__file__).resolve().parent
    venv_dir = root_dir / ".venv"
    if not venv_dir.exists():
        venv_dir = root_dir / "venv" # Fallback to common name
        
    script_path = root_dir / "walkingPal.py"
    env_path = root_dir / ".env"

    # 2. Find Python interpreter in venv
    if platform.system().lower().startswith("win"):
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:
        # Linux / MacOS
        venv_python = venv_dir / "bin" / "python"

    # 3. Validation
    if not venv_python.exists():
        print(f"Error: Virtual environment python not found at: {venv_python}")
        print("Please run 'python install.py' first.")
        sys.exit(1)

    if not script_path.exists():
        print(f"Error: walkingPal.py not found at: {script_path}")
        sys.exit(1)

    # Check for .env (Important for OpenAI/OpenRouter)
    if not env_path.exists():
        print("-" * 50)
        print("WARNING: .env file not found!")
        print("AI Scene analysis will NOT work without an API key.")
        print("Please create .env and add:")
        print("  OPENAI_API_KEY=sk-...")
        print("  OR OPEN_ROUTER_API_KEY=sk-or-...")
        print("-" * 50)
    else:
        try:
            content = env_path.read_text()
            openai_found = "OPENAI_API_KEY" in content
            or_found = "OPEN_ROUTER_API_KEY" in content or "open_router_api_key" in content
            if openai_found: print("[OK] OpenAI API Key detected.")
            elif or_found: print("[OK] OpenRouter API Key detected.")
            else: print("WARNING: .env found, but API keys seem missing.")
        except Exception: pass

    # 4. Construct command
    cmd = [str(venv_python), str(script_path)] + LAUNCH_ARGS
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    print(f"Launching WalkingPal in '{platform.system()}' mode...")
    print(f"Environment: {venv_python}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    # 5. Execute
    try:
        log_filename = "debug_output.txt"
        print(f"Logging stdout/stderr to '{log_filename}'...")
        with open(log_filename, "w", encoding="utf-8") as f:
            subprocess.check_call(cmd, stdout=f, stderr=subprocess.STDOUT)
    except KeyboardInterrupt:
        print("\nLauncher: Interrupted by user.")
    except subprocess.CalledProcessError as e:
        print(f"\nLauncher: Application exited with error code {e.returncode}.")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\nLauncher: Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
