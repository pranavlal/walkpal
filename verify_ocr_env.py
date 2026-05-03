import sys
import subprocess
try:
    import pytesseract
    print(f"pytesseract: {pytesseract.__version__}")
except ImportError:
    print("pytesseract: FAILED")

try:
    import easyocr
    print(f"easyocr: {easyocr.__version__} (Available)")
except ImportError:
    print("easyocr: FAILED")

# Check tesseract command
try:
    ver = subprocess.check_output(["tesseract", "--version"]).decode("utf-8").splitlines()[0]
    print(f"tesseract binary: {ver}")
except Exception as e:
    print(f"tesseract binary: FAILED ({e})")
