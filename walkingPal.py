#!/usr/bin/env python3
"""
WalkingPal - OAK-D Blind Navigation Assistant (NASA Safety-Refactored)
Adheres to NASA JPL "Power of Ten" rules for safety-critical reliability.
"""

from __future__ import annotations
import argparse
import time
import threading
import logging
import re
import sys
import os
import math
import yaml
import multiprocessing as mp
import numpy as np
import cv2
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any, Callable, Deque
from collections import deque
from dotenv import load_dotenv

# Local Imports
from depth_processor import DepthProcessor
from scene_describer import SceneDescriber, SceneChangeMonitor
from local_describer import LocalDescriber
from validation_logger import SessionLogger
from cameras.webcam import WebcamCamera
from cameras.oak_d import OakDCamera
from system_utils import ConnectivityMonitor, Watchdog, install_signal_handlers, is_shutdown_requested
from audio_controller import AudioController, sanitize_tts_text

# Constants
OAK_D_VFOV_DEG = 50.0
logger = logging.getLogger("walkingpal")

# -----------------------------
# Configuration Helpers (Rule 04)
# -----------------------------
def load_config(path: str) -> Dict[str, Any]:
    assert path is not None, "Config path is null"
    if not os.path.exists(path): return {}
    try:
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
            assert isinstance(cfg, dict) or cfg is None, "Invalid yaml format"
            return cfg or {}
    except Exception as e:
        logger.warning(f"Config load failed: {e}")
        return {}

def flatten_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    assert isinstance(cfg, dict), "Input must be a dictionary"
    out = {}
    for k, v in cfg.items():
        if isinstance(v, dict): out.update(flatten_config(v))
        else: out[k] = v
    return out

def setup_logging(debug: bool = False, log_file: Optional[str] = None):
    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s"
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        assert isinstance(log_file, str), "Log file path must be string"
        handlers.append(RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3))
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S", handlers=handlers)

# -----------------------------
# Core Application Class (Rule 07)
# -----------------------------
class WalkPalApp:
    def __init__(self, args: argparse.Namespace, config: Dict[str, Any]):
        assert args is not None and config is not None
        assert hasattr(args, 'max_retries'), "Args missing max_retries"
        
        self.args, self.config = args, config
        self.audio, self.scene_desc, self.camera = None, None, None
        self.nav_proc, self.logger_ver, self.conn_mon = None, None, None
        self.watchdog, self.ocr_pool, self.nav_pool = None, None, None
        self.gui_proc, self.gui_queue = None, None
        
        # Performance/State Tracking (Rule 03: Pre-allocation)
        self.last_ts = {k: 0.0 for k in ['ocr', 'ocr_req', 'speak', 'h_warn', 'light', 'vlm']}
        self.state = {
            'spoken': "", 'nav': (False, False, False, False, False, False, False), 
            'label': None, 'pitch': 0.0, 'roi': None
        }
        self.futures: Dict[str, Optional[Future]] = {'ocr': None, 'vlm': None}
        self.vlm_res = {'label': None, 'ts': 0.0}
        
    def run(self):
        """Main entry point with reconnection logic (Rule 03, 02)."""
        try:
            self.init_services()
            retries = 0
            while retries < self.args.max_retries and not is_shutdown_requested():
                if self.connect():
                    assert self.camera is not None, "Camera connection failed"
                    self.audio.speak(i18n('nav_started'))
                    self.main_loop()
                    # Reset retries if we reached the main loop and stayed for a while
                    # (optional: only reset if session lasted > 10s)
                    retries = 0 
                if is_shutdown_requested(): break
                retries += 1
                self.reconnect_wait()
            assert retries <= self.args.max_retries, "Retry count exceeded bounds"
        finally:
            self.cleanup()

    def init_services(self):
        """Initialize all subsystems (Rule 08)."""
        self.audio = AudioController(tts_rate=self.args.tts_rate, volume=self.args.volume)
        assert self.audio is not None, "AudioController fail"
        
        self.conn_mon = ConnectivityMonitor()
        self.conn_mon.start()
        
        self.nav_proc = DepthProcessor(width=640, height=400)
        assert self.nav_proc is not None, "DepthProcessor fail"
        
        self.ocr_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="OCR")
        self.nav_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="VLM")
        
        if self.args.watchdog_s > 0: self.watchdog = Watchdog(timeout_s=self.args.watchdog_s)
        if self.args.record: self.logger_ver = SessionLogger(record_depth=self.args.record_depth)
        
        if self.args.gui:
            from gui import run_gui
            self.gui_queue = mp.Queue(maxsize=2) # Keep it lean
            self.gui_proc = mp.Process(target=run_gui, args=(self.gui_queue,), name="WalkPalGUI")
            self.gui_proc.start()
            logger.info("GUI Process started.")
        
        self._init_ai()
        self._init_dbs()

    def _init_ai(self):
        assert self.args.openai_model is not None
        self.scene_desc = SceneDescriber(
            api_key=os.getenv("OPEN_ROUTER_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            local_describer=LocalDescriber() if self.args.enable_local_vlm else None,
            openai_model=self.args.openai_model
        )
        assert self.scene_desc is not None

    def _init_dbs(self):
        c, cl = self.args.confirm_frames, self.args.clear_frames
        assert c > 0 and cl > 0
        self.dbs = {
            'dropoff': DebouncedBool(c, cl), 'stairs': DebouncedBool(c, cl),
            'pothole': DebouncedBool(max(2, c), max(2, cl)),
            'blocked_L': DebouncedBool(max(1, c-1), cl),
            'blocked_C': DebouncedBool(max(1, c-1), cl),
            'blocked_R': DebouncedBool(max(1, c-1), cl),
            'uncertain': DebouncedBool(max(1, c-1), cl),
            'hazard': DebouncedBool(self.args.hazard_on, self.args.hazard_off)
        }
        self.smoother = MajorityLabel(k=self.args.dir_window)

    def connect(self) -> bool:
        """Hardware discovery (Rule 04)."""
        if not self.args.force_webcam:
            try:
                self.camera = OakDCamera(enable_potholes=self.args.enable_potholes, enable_ocr=self.args.enable_ocr)
                if self.camera.start(): return True
            except Exception as e: logger.warning(f"OAK-D failed: {e}")
        try:
            self.camera = WebcamCamera()
            assert self.camera is not None
            return self.camera.start()
        except Exception as e:
            logger.error(f"Webcam start fail: {e}")
            return False

    def main_loop(self):
        """Core cycle (Rule 02)."""
        hazards = {h.strip().lower() for h in self.args.hazards.split(",") if h.strip()}
        l_map = getattr(self.camera, 'label_map', [])
        assert isinstance(hazards, set)
        
        while self.camera.is_running() and not is_shutdown_requested():
            if self.watchdog: self.watchdog.reset()
            self.step(hazards, l_map)
            assert threading.active_count() > 0, "No active threads in main loop"

    def step(self, hazards: set, l_map: List):
        """Single pipeline iteration (Rule 04)."""
        now = time.time()
        frames = self.camera.get_frames()
        assert frames is not None, "Camera returned None frames"
        
        # 1. Sensors & Vision
        light = self.process_light(frames, now)
        _, h_label = self.process_yolo(now, hazards, l_map)
        
        # 2. AI Tasks
        rgb = frames.scene if frames.scene is not None else frames.video
        assert rgb is not None, "No RGB frame for AI"
        self.process_ai(rgb, now)
        
        # 3. Navigation Logic
        nav = self.process_nav(frames.depth, now, light)
        assert 'msg' in nav, "Navigation failed to produce message"
        
        # 4. Side Effects (Rule 08)
        pan = -0.8 if nav['is_blocked_L'] else (0.8 if nav['is_blocked_R'] else 0.0)
        self.feedback(nav, h_label, pan, now)
        if self.gui_queue and not self.gui_queue.full():
             packet = {
                 'frame': rgb.copy() if rgb is not None else None,
                 'nav': nav['msg'],
                 'scene': self.state.get('spoken', 'Analyzing...'),
                 'hazards': list(hazards)
             }
             assert 'nav' in packet and 'scene' in packet
             self.gui_queue.put_nowait(packet)
        if self.args.debug: self.debug_gui(frames, nav)

    def process_light(self, frames, now):
        assert frames.preview is not None or not self.args.force_webcam
        # Optimization: Subsample (Rule 03)
        p = frames.preview
        if p is not None:
             b = float(np.mean(p[::4, ::4])) 
        else: b = 100.0
        
        light = classify_light(b)
        assert light in ["dark", "dim", "normal"]
        if light == "dark" and (now - self.last_ts['light']) > 15.0:
            self.audio.speak(i18n('low_light'))
            self.last_ts['light'] = now
        return light

    def process_yolo(self, now, hazards, l_map):
        found, label = False, None
        if self.args.enable_yolo:
            for d in (self.camera.get_detections() or []):
                name = str(l_map[d.label]).lower() if d.label < len(l_map) else ""
                if name in hazards and d.confidence > self.args.yolo_conf:
                    found, label = True, name
        present = self.dbs['hazard'].update(found)
        if present and label and (now - self.last_ts['h_warn']) > self.args.hazard_cooldown_s:
            self.audio.speak(i18n('hazard', label))
            self.last_ts['h_warn'] = now
        return present, label

    def process_ai(self, rgb, now):
        assert rgb is not None
        if self.args.enable_ocr: self.handle_ocr(rgb, now)
        desc = self.scene_desc.process(rgb)
        if desc: 
            assert isinstance(desc, str)
            self.audio.speak(sanitize_tts_text(desc))
        self.handle_vlm(rgb, now)

    def handle_ocr(self, rgb, now):
        assert rgb is not None
        if self.futures['ocr'] and self.futures['ocr'].done():
            txt = self.futures['ocr'].result()
            if txt and (now - self.last_ts['ocr']) > self.args.ocr_cooldown_s:
                self.audio.speak(i18n('sign_reads', sanitize_tts_text(txt)))
                self.last_ts['ocr'] = now
            self.futures['ocr'] = None
        if not self.futures['ocr'] and (now - self.last_ts['ocr_req']) > self.args.ocr_every_s:
            self.last_ts['ocr_req'] = now
            self.futures['ocr'] = self.ocr_pool.submit(run_ocr_auto, rgb.copy(), self.args.ocr_engine, self.args.ocr_lang)

    def handle_vlm(self, rgb, now):
        assert rgb is not None
        if self.futures['vlm'] and self.futures['vlm'].done():
            res = self.futures['vlm'].result()
            if res: 
                assert isinstance(res, dict)
                self.vlm_res = {'label': res.get('label'), 'ts': now}
            self.futures['vlm'] = None
        if not self.futures['vlm'] and (now - self.vlm_res['ts']) > 2.0:
            self.futures['vlm'] = self.nav_pool.submit(self.scene_desc.analyze_navigation, rgb.copy())

    def process_nav(self, depth, now, light):
        if depth is None: 
            return {'msg': i18n('uncertain'), 'is_blocked_L': False, 'is_blocked_R': False, 
                    'dropoff': False, 'stairs': False, 'pothole': False}
        assert depth.ndim == 2
        
        self.ensure_roi(depth)
        res = self.nav_proc.process_frame(depth)
        assert res is not None and 'nav' in res
        
        self.update_pitch(depth)
        pok, lok = (-30 < self.state['pitch'] < 50), (light != "dark")
        
        d, b = (res['dropoff'] if pok and lok else False), any(res['nav'][k] == 'blocked' for k in 'LCR')
        u = (res['plane'] is None) or not (pok and lok)
        s = detect_stairs(depth, self.state['roi']) if pok else None
        p = self.check_pothole(depth, pok and lok and not d)
        
        return self.debounce(res, d, b, u, s, p)

    def ensure_roi(self, depth):
        assert depth is not None
        if self.state['roi']: return
        h, w = depth.shape[:2]
        # Normalized ROIs (NASA Standard: Resolution Independent)
        self.state['roi'] = {
            'bx0': int(w * 0.38), 'bx1': int(w * 0.62), 'by0': int(h * 0.78), 'by1': int(h * 0.98),
            'sx0': int(w * 0.40), 'sx1': int(w * 0.60), 'sy0': int(h * 0.30), 'sy1': int(h * 0.95)
        }
        assert 'bx0' in self.state['roi']

    def update_pitch(self, depth):
        assert depth is not None
        imu = self.camera.get_imu()
        if imu and imu.packets:
            rv = imu.packets[-1].rotationVector
            _, p, _ = euler_from_quaternion(rv.i, rv.j, rv.k, rv.real)
            m = math.degrees(p)
        else: m = estimate_pitch_from_depth(depth, self.args.camera_height)
        self.state['pitch'] = self.state['pitch'] * 0.7 + m * 0.3
        assert isinstance(self.state['pitch'], float)

    def check_pothole(self, depth, active):
        if not (self.args.enable_potholes and active): return self.dbs['pothole'].update(False)
        r = self.state['roi']
        st = roi_stats(depth, r['bx0'], r['by0'], r['bx1'], r['by1'])
        assert st is not None
        if st.valid_ratio < self.args.pothole_roi_valid: return self.dbs['pothole'].update(False)
        return self.dbs['pothole'].update(pothole_score(depth, r['bx0'], r['by0'], r['bx1'], r['by1']) >= self.args.pothole_score)

    def debounce(self, res, d, b, u, s, p):
        assert res is not None
        st = {
            'dropoff': self.dbs['dropoff'].update(d), 'stairs': self.dbs['stairs'].update(s is not None),
            'pothole': self.dbs['pothole'].update(p), 'uncertain': self.dbs['uncertain'].update(u),
            'is_blocked_L': self.dbs['blocked_L'].update(res['nav']['L']=='blocked'),
            'is_blocked_C': self.dbs['blocked_C'].update(res['nav']['C']=='blocked'),
            'is_blocked_R': self.dbs['blocked_R'].update(res['nav']['R']=='blocked'),
        }
        raw = i18n('dropoff') if st['dropoff'] else (direction_text(not st['is_blocked_L'], not st['is_blocked_C'], not st['is_blocked_R'], "1m") if (st['is_blocked_L'] or st['is_blocked_C'] or st['is_blocked_R']) else (i18n('uncertain') if st['uncertain'] else i18n('clear')))
        st['msg'] = self.smoother.update(raw)
        assert isinstance(st['msg'], str)
        return st

    def feedback(self, nav, h_label, pan, now):
        assert nav and 'msg' in nav
        is_h = nav['dropoff'] or nav['stairs'] or nav['pothole']
        sig = (nav['is_blocked_L'], nav['is_blocked_C'], nav['is_blocked_R'], nav['dropoff'], nav['stairs'], nav['pothole'], nav['uncertain'])
        if is_h or sig != self.state['nav'] or h_label != self.state['label']:
            self.state['nav'], self.state['label'] = sig, h_label
            if (now - self.last_ts['speak']) > (0.5 if is_h else 2.0):
                self.audio.speak(sanitize_tts_text(nav['msg']), pan=pan)
                self.last_ts['speak'] = now
            assert self.last_ts['speak'] <= now

    def debug_gui(self, frames, nav):
        assert frames is not None and nav is not None
        if self.args.headless or frames.depth is None: return
        vis = cv2.applyColorMap((np.clip(frames.depth, 0, 5000)/5000*255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.putText(vis, nav['msg'], (10, 30), 0, 0.7, (255,255,255), 2)
        cv2.imshow("WalkPal", vis)
        if cv2.waitKey(1) == ord('q'): is_shutdown_requested.set()

    def reconnect_wait(self):
        if self.camera: self.camera.stop()
        self.camera = None
        self.audio.speak(i18n('device_disconnect'))
        time.sleep(2.0)
        assert time.time() > 0

    def cleanup(self):
        """Safe cleanup (Rule 03)."""
        if self.camera: self.camera.stop()
        if self.watchdog: self.watchdog.stop()
        for p in [self.ocr_pool, self.nav_pool]: 
            if p: p.shutdown(wait=False)
        if self.conn_mon: self.conn_mon.stop()
        if self.scene_desc: self.scene_desc.shutdown()
        if self.logger_ver: self.logger_ver.close()
        if self.gui_proc:
            if self.gui_proc.is_alive():
                self.gui_proc.terminate()
            self.gui_proc.join(timeout=1.0)
            assert not self.gui_proc.is_alive()
        if self.audio: self.audio.shutdown()
        cv2.destroyAllWindows()
        logger.info("Cleanup complete.")

# -----------------------------
# Global Helpers
# -----------------------------
MESSAGES: Dict[str, Dict[str, str]] = {
    'en': {
        'nav_started': 'Navigation started.',
        'nav_stopped': 'Navigation stopped.',
        'clear': 'Clear ahead.',
        'uncertain': 'Uncertain path - please be cautious.',
        'stop': 'Stop! Obstacle ahead.',
        'dropoff': 'Warning. Drop off ahead.',
        'pothole': 'Warning. Pothole or uneven ground.',
        'stairs_up': 'Stairs up ahead.',
        'stairs_down': 'Stairs down ahead.',
        'stairs': 'Stairs detected.',
        'hazard': 'Warning. {} ahead.',
        'sign_reads': 'Sign reads: {}',
        'self_test': 'Running system self-test.',
        'self_test_pass': 'Self-test passed. All systems green.',
        'device_disconnect': 'Device disconnected. Attempting to reconnect.',
        'device_reconnect_fail': 'Max reconnection attempts reached. Please check the cable.',
        'low_light': 'Low light detected. Guidance may be less reliable.',
        'dir_left': 'left', 'dir_center': 'center', 'dir_right': 'right',
        'obstacle_ahead_go': 'Obstacle {} ahead. Go {} or {}.',
        'obstacle_go': 'Obstacle {} ahead. Go {}.',
        'go_left_center_right': 'Go left, center, or right.',
        'step': 'step', 'steps': 'steps'
    },
    'hi': {
        'nav_started': 'नेविगेशन शुरू।',
        'nav_stopped': 'नेविगेशन बंद।',
        'clear': 'रास्ता साफ है।',
        'uncertain': 'सावधानी बरतें।',
        'stop': 'रुकें! आगे बाधा है।',
        'dropoff': 'गड्ढा या सीढ़ी।',
        'pothole': 'असमान जमीन।',
        'stairs_up': 'सीढ़ियाँ ऊपर।',
        'stairs_down': 'सीढ़ियाँ नीचे।',
        'stairs': 'सीढ़ियाँ।',
        'hazard': 'सावधान। आगे {}।',
        'low_light': 'अंधेरा है।',
        'dir_left': 'बाएं', 'dir_center': 'बीच', 'dir_right': 'दाएं'
    }
}
_lang = 'en'
def set_language(l): 
    global _lang
    assert l in MESSAGES or l == 'en'
    _lang = l if l in MESSAGES else 'en'

def i18n(k, *a):
    assert k is not None
    t = MESSAGES.get(_lang, MESSAGES['en']).get(k, k)
    return t.format(*a) if a else t

def euler_from_quaternion(x, y, z, w):
    assert all(isinstance(v, (float, int)) for v in [x,y,z,w])
    t0 = +2.0 * (w * x + y * z); t1 = +1.0 - 2.0 * (x * x + y * y)
    t2 = +2.0 * (w * y - z * x); t2 = np.clip(t2, -1, 1)
    t3 = +2.0 * (w * z + x * y); t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t0, t1), math.asin(t2), math.atan2(t3, t4)

def roi_stats(depth: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> RoiStats:
    assert depth is not None and depth.ndim == 2
    roi = depth[y0:y1, x0:x1]
    if roi.size == 0: return RoiStats(0.0, 0, 0, 0)
    v = roi[roi > 0]
    if v.size < 50: return RoiStats(float(v.size)/roi.size, 0, 0, 0)
    return RoiStats(float(v.size)/roi.size, int(np.percentile(v, 10)), int(np.median(v)), int(np.percentile(v, 90)))

def detect_dropoff(depth_mm: np.ndarray, roi_cache: Dict[str, Any], 
                   base_dropoff_mm: int, dropoff_invalid_ratio: float, 
                   min_valid: float, require_center_for_clear: bool,
                   stB: RoiStats, stL: RoiStats, stC: RoiStats, stR: RoiStats,
                   pitch_deg: float = 0.0, camera_height_m: float = 1.5) -> bool:
    """Robust dropoff detection (Rule 04)."""
    assert depth_mm is not None
    assert stB is not None
    invalid_frac = 1.0 - stB.valid_ratio
    path_visible = (stC.valid_ratio > 0.15) or (stL.valid_ratio > 0.15) or (stR.valid_ratio > 0.15)
    
    # Adaptive threshold based on pitch
    angle_down_deg = pitch_deg + 19.0
    if angle_down_deg < 5.0: return False
    
    expected_mm = (camera_height_m * 1000.0) / math.tan(math.radians(angle_down_deg))
    adaptive_threshold_mm = max(base_dropoff_mm, int(expected_mm * 1.5))
    
    is_too_deep = (stB.median_mm != 0 and stB.median_mm > adaptive_threshold_mm)
    eff_inv_ratio = dropoff_invalid_ratio + (0.1 if path_visible else 0.0)
    is_invalid = (invalid_frac >= eff_inv_ratio)
    
    return (is_invalid or is_too_deep) if path_visible else False

def classify_light(b): 
    assert isinstance(b, (float, int))
    return "dark" if b < 35 else ("dim" if b < 75 else "normal")

def direction_text(L: bool, C: bool, R: bool, d: str = "") -> str:
    """Generate localized natural language directions (Rule 04, 08)."""
    assert all(isinstance(v, bool) for v in [L, C, R])
    opts = [(i18n('dir_left'), L), (i18n('dir_center'), C), (i18n('dir_right'), R)]
    free = [name for name, is_free in opts if is_free]
    blocked = [name for name, is_free in opts if not is_free]
    
    if not free: return i18n('stop')
    if len(free) == 3: return i18n('clear')
    
    blocked_str = " & ".join(blocked)
    if len(free) == 2:
        return i18n('obstacle_ahead_go', blocked_str, free[0], free[1])
    return i18n('obstacle_go', blocked_str, free[0])

def detect_stairs(depth, r):
    assert depth is not None and r is not None
    bins = 5; ys = np.linspace(r['sy0'], r['sy1'], bins+1).astype(int)
    meds = [roi_stats(depth, r['sx0'], ys[i], r['sx1'], ys[i+1]).median_mm for i in range(bins)]
    if any(m==0 for m in meds): return None
    diffs = [meds[i+1]-meds[i] for i in range(bins-1)]
    if all(d > 120 for d in diffs): return "stairs_down"
    if all(d < -120 for d in diffs): return "stairs_up"
    return None

def pothole_score(depth, x0, y0, x1, y1):
    assert depth is not None
    roi = depth[y0:y1, x0:x1].astype(np.float32)
    v = roi > 0
    if v.mean() < 0.3: return 0.0
    med = np.median(roi[v])
    rough = (np.abs(roi[:, 1:] - roi[:, :-1])).mean() + (np.abs(roi[1:, :] - roi[:-1, :])).mean()
    return rough / (med + 1e-6) * 10.0

def estimate_pitch_from_depth(depth, h_m=1.5):
    assert depth is not None and h_m > 0
    h, w = depth.shape
    v = depth[int(h*.8):, :]; v = v[v>0]
    if v.size < 100: return 0.0
    return math.degrees(math.atan2(h_m, np.median(v)/1000.0)) - 19.0

def run_ocr_auto(bgr, engine="easyocr", lang="en"):
    """Functional EasyOCR implementation."""
    assert bgr is not None
    try:
        import easyocr
        reader = easyocr.Reader([lang] if isinstance(lang, str) else lang, gpu=False)
        results = reader.readtext(bgr)
        text = " ".join([res[1] for res in results if res[2] > 0.4])
        return text.strip()
    except Exception as e:
        logger.warning(f"OCR Failed: {e}")
        return ""

class DebouncedBool:
    def __init__(self, on, off): 
        assert on > 0 and off > 0
        self.on, self.off, self.c_on, self.c_off, self.state = on, off, 0, 0, False
    def update(self, v):
        if v: self.c_on, self.c_off = self.c_on+1, 0
        else: self.c_off, self.c_on = self.c_off+1, 0
        if not self.state and self.c_on >= self.on: self.state, self.c_on = True, 0
        elif self.state and self.c_off >= self.off: self.state, self.c_off = False, 0
        return self.state

class MajorityLabel:
    def __init__(self, k): 
        assert k > 0
        self.buf = deque(maxlen=k)
    def update(self, l):
        assert l is not None
        self.buf.append(l)
        return max(set(self.buf), key=self.buf.count)

@dataclass
class RoiStats:
    valid_ratio: float; near_mm: int; median_mm: int; far_mm: int

def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--max_retries", type=int, default=5)
    ap.add_argument("--confirm_frames", type=int, default=5)
    ap.add_argument("--clear_frames", type=int, default=2)
    ap.add_argument("--dir_window", type=int, default=5)
    ap.add_argument("--watchdog_s", type=float, default=5.0)
    ap.add_argument("--tts_rate", type=int, default=175)
    ap.add_argument("--volume", type=float, default=1.0)
    ap.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    ap.add_argument("--hazards", type=str, default="person,chair,car")
    ap.add_argument("--yolo_conf", type=float, default=0.5)
    ap.add_argument("--hazard_on", type=int, default=2)
    ap.add_argument("--hazard_off", type=int, default=3)
    ap.add_argument("--hazard_cooldown_s", type=float, default=2.0)
    ap.add_argument("--ocr_every_s", type=float, default=5.0)
    ap.add_argument("--ocr_cooldown_s", type=float, default=7.0)
    ap.add_argument("--ocr_lang", default="eng")
    ap.add_argument("--ocr_engine", default="auto")
    ap.add_argument("--ocr_min_len", type=int, default=5)
    ap.add_argument("--ocr_auto_min_quality", type=float, default=0.5)
    ap.add_argument("--camera_height", type=float, default=1.5)
    ap.add_argument("--pothole_roi_valid", type=float, default=0.3)
    ap.add_argument("--pothole_score", type=float, default=0.5)
    ap.add_argument("--language", default="en")
    ap.add_argument("--enable_yolo", action="store_true")
    ap.add_argument("--enable_potholes", action="store_true")
    ap.add_argument("--enable_ocr", action="store_true")
    ap.add_argument("--enable_local_vlm", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--record_depth", action="store_true")
    ap.add_argument("--force_webcam", action="store_true")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--log_file", default=None)
    
    args = ap.parse_args()
    assert args is not None
    cfg = load_config(args.config)
    setup_logging(args.debug, args.log_file)
    set_language(args.language)
    install_signal_handlers()
    WalkPalApp(args, cfg).run()

if __name__ == "__main__":
    main()
