import numpy as np
import cv2
import logging
import math
from typing import Tuple, Optional, Dict, List

logger = logging.getLogger("walkingpal.depth")

class DepthProcessor:
    def __init__(self, width: int = 640, height: int = 400, hfov_deg: float = 72.0):
        assert width > 0 and height > 0
        assert hfov_deg > 0
        
        self.width = width
        self.height = height
        # Estimate focal length from HFOV
        self.fx = (width / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
        self.fy = self.fx # Approximate square pixels
        self.cx = width / 2.0
        self.cy = height / 2.0
        
        # Grid for reprojection (pre-calculated)
        self.u_grid, self.v_grid = np.meshgrid(np.arange(width), np.arange(height))
        
        # RANSAC params
        self.ransac_iters = 100
        self.ransac_thresh = 65.0
        self.min_inliers = 1500
        
        # Previous plane for temporal smoothing
        self.prev_plane = None # (a, b, c, d)
        
        # Visualization colors
        self.color_ground = (0, 255, 0)
        self.color_obs = (0, 0, 255)
        self.color_drop = (255, 0, 0) # Blue for dropoffs
        
        # Pre-allocated buffers (Rule 03)
        self.debug_img = np.zeros((height, width, 3), dtype=np.uint8)
        self.points_all = np.zeros((height * width, 3), dtype=np.float32)

    def reproject(self, depth_mm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert depth image to (N, 3) point cloud in Camera Frame."""
        assert depth_mm is not None and depth_mm.ndim == 2, "Invalid depth matrix"
        assert depth_mm.shape == (self.height, self.width), "Depth dimensions mismatch"
        
        mask = depth_mm > 0
        z = depth_mm[mask]
        u = self.u_grid[mask]
        v = self.v_grid[mask]
        
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        points = np.column_stack((x, y, z))
        assert points.shape[1] == 3, "Reprojection failed to produce 3D points"
        return points, mask

    def fit_plane_ransac(self, points: np.ndarray) -> Optional[np.ndarray]:
        """Robustly fit a plane (ax + by + cz + d = 0) to the points."""
        assert isinstance(points, np.ndarray)
        if points.shape[0] < self.min_inliers:
            return None
            
        n = points.shape[0]
        best_plane = None
        best_inliers = -1
        
        # Subsample for speed
        sample_points = points[np.random.choice(n, size=min(n, 5000), replace=False)]
        
        for _ in range(self.ransac_iters):
             idx = np.random.randint(0, sample_points.shape[0], 3)
             p1, p2, p3 = sample_points[idx]
             
             normal = np.cross(p2 - p1, p3 - p1)
             norm = np.linalg.norm(normal)
             if norm < 1e-6: continue
             
             normal = normal / norm
             d = -np.dot(normal, p1)
             
             dists = np.abs(np.dot(sample_points, normal) + d)
             count = np.count_nonzero(dists < self.ransac_thresh)
             
             if count > best_inliers:
                 best_inliers = count
                 best_plane = np.array([normal[0], normal[1], normal[2], d])

        assert best_plane is None or len(best_plane) == 4
        return best_plane

    def _update_temporal_plane(self, plane: np.ndarray) -> np.ndarray:
        """Apply consistency and smoothing to ground plane (Rule 04)."""
        assert plane is not None and len(plane) == 4
        # Camera Y is DOWN. Ground normal should point UP (negative Y).
        if plane[1] > 0: 
            plane = -plane
        
        if self.prev_plane is not None:
            self.prev_plane = 0.8 * self.prev_plane + 0.2 * plane
            n_norm = np.linalg.norm(self.prev_plane[:3])
            self.prev_plane /= n_norm
        else:
            self.prev_plane = plane
        return self.prev_plane

    def process_frame(self, depth_mm: np.ndarray) -> Dict:
        """Main pipeline (Rule 04 - Split for length)."""
        assert depth_mm is not None and depth_mm.size > 0
        h, w = depth_mm.shape
        self.debug_img.fill(0) # Reuse buffer (Rule 03)
        
        points, mask_valid = self.reproject(depth_mm)
        valid_u, valid_v = self.u_grid[mask_valid], self.v_grid[mask_valid]
        
        # 1. Fit Plane
        ground_mask = valid_v > (0.4 * h)
        plane = self.fit_plane_ransac(points[ground_mask])
        
        if plane is not None:
            final_plane = self._update_temporal_plane(plane)
            return self._analyze_navigation(points, valid_u, valid_v, final_plane)
        
        return self._empty_result()

    def _analyze_navigation(self, points: np.ndarray, u: np.ndarray, v: np.ndarray, plane: np.ndarray) -> Dict:
        """Classify pixels and generate nav hints (Rule 04)."""
        assert points.shape[0] == u.shape[0] == v.shape[0]
        assert len(plane) == 4
        
        dists = np.dot(points, plane[:3]) + plane[3]
        
        inliers = np.abs(dists) < 80
        obstacles = dists > 120
        dropoffs_deep = dists < -180
        
        # Colorize (Reuse buffer if possible, but boolean masks create new arrays)
        colors = np.zeros_like(points, dtype=np.uint8)
        colors[obstacles] = self.color_obs
        colors[inliers] = self.color_ground
        colors[dropoffs_deep] = self.color_drop
        
        self.debug_img[v, u] = colors
        
        # Logic analysis
        col_w = self.width // 3
        results = {k: self._check_col(points, obstacles, u, (i*col_w, (i+1)*col_w)) 
                  for i, k in enumerate(['L', 'C', 'R'])}
        
        # Better min_dist check
        min_dists = {k: self._get_min_z(points, obstacles, u, (i*col_w, (i+1)*col_w))
                    for i, k in enumerate(['L', 'C', 'R'])}
        
        is_dropoff = self._detect_dropoff_logic(u, v, dropoffs_deep, min_dists['C'])
        
        return {
            'plane': plane, 'debug_img': self.debug_img,
            'nav': results, 'dists': min_dists, 'dropoff': is_dropoff
        }

    def _check_col(self, pts, obs, u, range_u) -> str:
        mask = obs & (u >= range_u[0]) & (u < range_u[1])
        if np.count_nonzero(mask) > 50:
            near_z = np.min(pts[mask, 2])
            return 'blocked' if near_z < 1500 else 'free'
        return 'free'

    def _get_min_z(self, pts, obs, u, range_u) -> float:
        mask = obs & (u >= range_u[0]) & (u < range_u[1])
        return float(np.min(pts[mask, 2])) if np.any(mask) else 9999.0

    def _detect_dropoff_logic(self, u, v, drops, center_dist) -> bool:
        """Heuristic for dropoffs (Rule 04)."""
        h, w = self.height, self.width
        critical_mask = (u > w//3) & (u < 2*w//3) & (v > h*0.6)
        valid_crit = np.count_nonzero(critical_mask)
        drop_crit = np.count_nonzero(drops & critical_mask)
        
        if valid_crit > 1000 and (drop_crit / valid_crit > 0.05):
            return True
        # Implicit drop (missing data)
        if valid_crit / ( (w//3) * int(h*0.4) ) < 0.15 and center_dist > 1000:
            return True
        return False

    def _empty_result(self) -> Dict:
        return {
            'plane': None, 'debug_img': self.debug_img,
            'nav': {'L':'unknown', 'C':'unknown', 'R':'unknown'},
            'dists': {'L':0, 'C':0, 'R':0}, 'dropoff': False
        }
