#!/bin/bash
# WalkingPal Linux One-Click Setup
# Handles Apt dependencies, USB permissions (UDEV), and Desktop Entry creation.

echo ">>> Starting WalkingPal Linux Setup <<<"

# 1. System Dependencies (Requires sudo)
if [ -x "$(command -v apt-get)" ]; then
    echo "Common dependencies (Ubuntu/Debian detected)..."
    sudo apt-get update || exit 1
    sudo apt-get install -y python3-venv python3-pip libgl1-mesa-glx tesseract-ocr usbutils || exit 1
elif [ -x "$(command -v dnf)" ]; then
    echo "Common dependencies (Fedora detected)..."
    sudo dnf install -y python3-devel mesa-libGL tesseract usbutils || exit 1
elif [ -x "$(command -v pacman)" ]; then
    echo "Common dependencies (Arch Linux detected)..."
    sudo pacman -S --needed --noconfirm python python-pip mesa tesseract usbutils || exit 1
fi

# 2. USB Permissions (Rule 03: NASA Mission-Critical Hardware)
# OAK-D / DepthAI UDEV Rules
echo "Setting up OAK-D Hardware Permissions (UDEV)..."
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules > /dev/null || exit 1
sudo udevadm control --reload-rules && sudo udevadm trigger || exit 1
sudo usermod -aG plugdev $USER || echo "Warning: Could not add user to plugdev group"

# 3. Virtual Environment
echo "Initializing Python Virtual Environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Desktop Entry
echo "Creating Desktop Entry..."
DESKTOP_ENTRY_PATH="$HOME/.local/share/applications/WalkingPal.desktop"
cat <<EOF > "$DESKTOP_ENTRY_PATH"
[Desktop Entry]
Version=1.0
Type=Application
Name=WalkingPal
Comment=Navigation Assistant for the Blind
Exec=$(pwd)/.venv/bin/python3 $(pwd)/walkingPal.py
Icon=utilities-terminal
Terminal=false
Categories=Accessibility;Utility;
EOF
[ -f "$DESKTOP_ENTRY_PATH" ] || exit 1
chmod +x "$DESKTOP_ENTRY_PATH"

echo ">>> Setup Complete! You can now find WalkingPal in your application menu. <<<"
echo "NOTE: You may need to logout and log back in for USB 'plugdev' permissions to take effect."
