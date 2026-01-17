#!/usr/bin/env python3
"""
Universal launch script for WalkingPal.
Detects OS, finds invalid venv, checks setup, and launches walkingPal.py with features enabled.
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
    "--ocr_engine", "auto",
    "--ocr_lang", "eng+hin",
    "--ocr_fps", "1.25",
    # Scene Description is always enabled via .env key presence, 
    # but we can ensure standard thresholds are passed if needed.
]

def main():
    # 1. Determine paths
    root_dir = Path(__file__).resolve().parent
    venv_dir = root_dir / ".venv"
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

    # Check for .env (Important for OpenRouter)
    if not env_path.exists():
        print("-" * 50)
        print("WARNING: .env file not found!")
        print("OpenRouter Scene Description will NOT work without an API key.")
        print("Please create .env and add: open_router_api_key=sk-or-...")
        print("-" * 50)
        # We don't exit, just warn.
    else:
        # Simple check if key is inside
        try:
            content = env_path.read_text()
            if "open_router_api_key" not in content and "OPEN_ROUTER_API_KEY" not in content:
                 print("WARNING: .env found, but 'open_router_api_key' seems missing.")
        except Exception:
            pass

    # 4. Construct command
    cmd = [str(venv_python), str(script_path)] + LAUNCH_ARGS

    # Pass through any extra arguments provided to this script
    # e.g. python launch.py --debug
    save_log = False
    args = sys.argv[1:]
    if "--save_log" in args:
        save_log = True
        args.remove("--save_log")
    
    if args:
        cmd.extend(args)

    print(f"Launching WalkingPal in '{platform.system()}' mode...")
    print(f"Environment: {venv_python}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    if platform.system().lower().startswith("win"):
        print("TIP: If the application hangs or you hear only the start message:")
        print("1. Ensure you have installed OAK-D drivers (WinUSB) via Zadig.")
        print("2. Check that your USB cable is 3.0 capable.")
        print("-" * 50)

    # 5. Execute
    try:
        if save_log:
            log_filename = "debug_output.txt"
            print(f"Logging stdout/stderr to '{log_filename}'...")
            with open(log_filename, "w", encoding="utf-8") as f:
                subprocess.check_call(cmd, stdout=f, stderr=subprocess.STDOUT)
                print(f"Check '{log_filename}' for output.")
        else:
            subprocess.check_call(cmd)
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
