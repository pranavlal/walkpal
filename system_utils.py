import threading
import time
import socket
import logging
import signal
import sys
from typing import Optional, Callable

logger = logging.getLogger("walkingpal.system")

# -----------------------------
# Signal Handling
# -----------------------------
_shutdown_requested = threading.Event()

def signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    logger.info("Received %s, initiating shutdown...", sig_name)
    _shutdown_requested.set()
    # We raise KeyboardInterrupt to break out of blocking calls if any
    raise KeyboardInterrupt

def install_signal_handlers():
    """Install handlers for graceful shutdown signals."""
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    logger.debug("Signal handlers installed.")

def is_shutdown_requested() -> bool:
    return _shutdown_requested.is_set()

def request_shutdown():
    _shutdown_requested.set()

# -----------------------------
# Connectivity Monitor
# -----------------------------
class ConnectivityMonitor:
    """
    Checks internet connectivity in a background thread.
    Notifies when status changes (Online <-> Offline).
    """
    def __init__(self, check_interval_s: float = 2.0, host: str = "8.8.8.8"):
        self.check_interval_s = check_interval_s
        self.host = host
        self.online = False
        self._stop_event = threading.Event()
        self._thread = None
        self._status_changed = False
        self._lock = threading.Lock()

    def start(self):
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _worker(self):
        logger.info("ConnectivityMonitor started.")
        while not self._stop_event.is_set():
            is_connected = False
            try:
                # Simple ping via connect (low overhead)
                with socket.create_connection((self.host, 53), timeout=1.5):
                    is_connected = True
            except OSError:
                pass

            with self._lock:
                if is_connected != self.online:
                    self.online = is_connected
                    self._status_changed = True
                    logger.info(f"Connectivity Changed: Online={self.online}")
            
            time.sleep(self.check_interval_s)

    def poll_status_change(self) -> Optional[bool]:
        """Returns new status if changed, else None."""
        with self._lock:
            if self._status_changed:
                self._status_changed = False
                return self.online
            return None
    
    def is_online(self) -> bool:
        with self._lock:
            return self.online

# -----------------------------
# Watchdog Timer
# -----------------------------
class Watchdog:
    """Detects if main loop stalls beyond timeout."""
    
    def __init__(self, timeout_s: float = 5.0, callback: Optional[Callable] = None):
        self.timeout = timeout_s
        self.callback = callback or self._default_callback
        self._timer: Optional[threading.Timer] = None
        self._active = False
    
    def _default_callback(self):
        logger.error("WATCHDOG: Main loop stalled for %.1fs!", self.timeout)
    
    def start(self):
        """Start the watchdog."""
        self._active = True
        self.reset()
    
    def reset(self):
        """Reset the watchdog timer (call this in main loop)."""
        if not self._active:
            return
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(self.timeout, self.callback)
        self._timer.daemon = True
        self._timer.start()
    
    def stop(self):
        """Stop the watchdog."""
        self._active = False
        if self._timer:
            self._timer.cancel()
            self._timer = None
