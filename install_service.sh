#!/bin/bash
# install_service.sh
# Installs walkingPal as a systemd service.

set -e

SERVICE_FILE="walkingpal.service"
INSTALL_PATH="/etc/systemd/system/${SERVICE_FILE}"
USER_NAME=${SUDO_USER:-$USER}
USER_HOME=$(eval echo ~$USER_NAME)
APP_DIR="$(pwd)"
PYTHON_BIN="${APP_DIR}/.venv/bin/python"

echo "Installing walkingPal service..."
echo "User: ${USER_NAME}"
echo "App Dir: ${APP_DIR}"

# Generate service file
cat <<EOF > ${SERVICE_FILE}
[Unit]
Description=WalkingPal Blind Navigation Assistant
After=network.target sound.target

[Service]
Type=simple
User=${USER_NAME}
WorkingDirectory=${APP_DIR}
ExecStart=${PYTHON_BIN} ${APP_DIR}/walkingPal.py --config ${APP_DIR}/config.yaml
Restart=always
RestartSec=3
Environment=PYTHONUNBUFFERED=1
# Add udev rules handled via depthai dependencies, normally not needed here if user in plugdev

[Install]
WantedBy=multi-user.target
EOF

echo "Generated ${SERVICE_FILE}."

# Check if sudo is needed
if [ "$EUID" -ne 0 ]; then
    echo "Please run this script with sudo to install the service."
    echo "  sudo ./install_service.sh"
    exit 1
fi

cp ${SERVICE_FILE} ${INSTALL_PATH}
echo "Copied to ${INSTALL_PATH}"

systemctl daemon-reload
systemctl enable walkingpal.service
echo "Service enabled. Start with: sudo systemctl start walkingpal"
echo "Check logs: journalctl -f -u walkingpal"

exit 0
