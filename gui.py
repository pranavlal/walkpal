import sys
import logging
import cv2
import numpy as np
import time
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QScrollArea, QFrame)
from PySide6.QtGui import (QImage, QPixmap, QShortcut, QKeySequence, QPalette, QColor, QFont)
from PySide6.QtCore import (Qt, QTimer, Slot, Signal, QObject, QThread, QSize)
from PySide6.QtGui import QAccessible

logger = logging.getLogger("walkingpal.gui")

# -----------------------------
# Accessible Styles (WCAG 2.2 Standard: High Contrast)
# -----------------------------
STYLING = """
QMainWindow { background-color: #1a1a1a; }
QLabel { color: #ffffff; font-size: 16px; font-weight: 500; }
QPushButton { 
    background-color: #005a9c; color: white; border-radius: 6px; 
    padding: 12px 24px; font-weight: bold; border: 2px solid transparent; 
}
QPushButton:focus { border: 3px solid #ffcc00; background-color: #0073cc; }
QPushButton:hover { background-color: #0073cc; }
QFrame#Divider { background-color: #333333; min-height: 2px; max-height: 2px; }
QScrollArea { border: none; background: transparent; }
"""

class NavData:
    def __init__(self):
        self.frame = None
        self.nav_msg = "Starting..."
        self.scene_text = "Analyzing scene..."
        self.hazards = []
        self.fps = 0.0

class WalkPalWindow(QMainWindow):
    def __init__(self, data_queue=None):
        super().__init__()
        assert data_queue is not None or "no_queue" in sys.argv, "GUI requires data queue"
        self.data_queue = data_queue
        
        self.setWindowTitle("WalkingPal Accessibility Console")
        self.setMinimumSize(1024, 768)
        self.setStyleSheet(STYLING)
        assert self.minimumWidth() >= 1024
        
        # Main Layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # 1. Video Viewer (The "Eyes")
        self.video_container = QVBoxLayout()
        self.video_label = QLabel("Waiting for camera...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAccessibleName("Live Navigation Feed")
        self.video_label.setAccessibleDescription("A live view of the camera with navigation markers.")
        
        # WCAG 2.2: Large focus indicator for the video feed if it were interactive (it isn't, but we label it)
        self.video_container.addWidget(self.video_label)
        
        # 2. Information Sidebar (The "Brain")
        self.sidebar = QVBoxLayout()
        self.sidebar.setContentsMargins(20, 0, 20, 0)
        
        self.status_header = QLabel("SYSTEM STATUS")
        self.status_header.setFont(QFont("Inter", 18, QFont.Bold))
        
        self.nav_label = QLabel("Clear")
        self.nav_label.setStyleSheet("color: #00ff00; font-size: 32px; font-weight: 800;")
        self.nav_label.setAccessibleName("Navigation Message")
        
        self.divider = QFrame()
        self.divider.setObjectName("Divider")
        
        self.scene_label = QLabel("Analyzing...")
        self.scene_label.setWordWrap(True)
        self.scene_label.setAccessibleName("AI Scene Description")
        
        self.sidebar.addWidget(self.status_header)
        self.sidebar.addWidget(self.nav_label)
        self.sidebar.addWidget(self.divider)
        self.sidebar.addWidget(self.scene_label)
        self.sidebar.addStretch()
        
        # 3. Controls (The "Feet")
        self.controls = QHBoxLayout()
        self.btn_exit = QPushButton("&Exit (ESC)")
        self.btn_exit.clicked.connect(self.close)
        self.btn_exit.setAccessibleDescription("Close the WalkingPal application.")
        
        self.controls.addWidget(self.btn_exit)
        self.sidebar.addLayout(self.controls)
        
        main_layout.addLayout(self.video_container, stretch=3)
        main_layout.addLayout(self.sidebar, stretch=1)
        
        # Update Timer (Rule 02: Fixed frequency loop)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(50) # 20 FPS GUI refresh
        assert self.timer.isActive()
        
    def update_ui(self):
        if not self.data_queue or self.data_queue.empty():
            return
            
        try:
            data = self.data_queue.get_nowait()
            assert isinstance(data, dict), "GUI received invalid data format"
            assert 'nav' in data, "GUI data missing 'nav' key"
            
            # Update Frame
            if 'frame' in data and data['frame'] is not None:
                img = data['frame']
                assert isinstance(img, np.ndarray), "Frame is not numpy array"
                assert img.ndim == 3, "Frame must be RGB/BGR"
                
                h, w, ch = img.shape
                bytes_per_line = ch * w
                q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(
                    self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                
                # Update Accessibility Alt-Text (WCAG 2.2)
                self.video_label.setAccessibleDescription(f"Live feed description: {data.get('scene', 'No description yet.')}")
            
            # Update Status
            msg = data.get('nav', 'Starting...')
            self.nav_label.setText(msg.upper())
            self.nav_label.setStyleSheet(f"color: {'#ff3333' if 'Warning' in msg or 'Stop' in msg else '#00ff00'}; font-size: 32px; font-weight: 800;")
            
            self.scene_label.setText(data.get('scene', 'Analyzing...'))
            
            # Announce critical changes (A11y Event)
            if 'nav' in data and data['nav'] != self.nav_label.text():
                 QAccessible.updateAccessibility(self.nav_label, 0, QAccessible.TextAttributeChanged)
                 
        except Exception as e:
            logger.debug(f"GUI Update skip: {e}")

def run_gui(queue):
    assert queue is not None or "no_queue" in sys.argv
    app = QApplication(sys.argv)
    assert app is not None
    window = WalkPalWindow(queue)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_gui(None)
