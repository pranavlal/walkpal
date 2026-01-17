import numpy as np
import cv2
import logging
import math
from typing import Tuple, Optional, Dict, List

logger = logging.getLogger("walkingpal.depth")

class DepthProcessor:
    def __init__(self, width: int = 640, height: int = 400, hfov_deg: float = 72.0):
        self.width = width
        self.height = height
        # Estimate focal length from HFOV
        # tan(hfov/2) = (w/2) / fx
        self.fx = (width / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
        self.fy = self.fx # Approximate square pixels
        self.cx = width / 2.0
        self.cy = height / 2.0
        
        # Grid for reprojection (pre-calculated)
        self.u_grid, self.v_grid = np.meshgrid(np.arange(width), np.arange(height))
        
        # RANSAC params
        # RANSAC params
        self.ransac_iters = 100         # Robustness: Increased from 50 (laptop power is high)
        self.ransac_thresh = 65.0       # Robustness: Increased from 50mm to handle depth noise better
        self.min_inliers = 1500         # Robustness: Reduced slightly to allow fitting smaller valid patches

        
        # Previous plane for temporal smoothing
        self.prev_plane = None # (a, b, c, d)
        
        # Visualization colors
        self.color_ground = (0, 255, 0)
        self.color_obs = (0, 0, 255)
        self.color_drop = (50, 50, 50) # Gray/Blueish for "nothing"

    def reproject(self, depth_mm: np.ndarray) -> np.ndarray:
        """
        Convert depth image to (N, 3) point cloud in Camera Frame.
        Z is forward, X is right, Y is down (standard CV).
        Scale: mm
        """
        # Filter invalid depth
        mask = depth_mm > 0
        z = depth_mm[mask]
        u = self.u_grid[mask]
        v = self.v_grid[mask]
        
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        # Stack to (N, 3)
        points = np.column_stack((x, y, z))
        return points, mask

    def fit_plane_ransac(self, points: np.ndarray) -> Optional[np.ndarray]:
        """
        Robustly fit a plane (ax + by + cz + d = 0) to the points.
        Assumes majority of 'bottom' points are ground.
        Optimized vectorized NumPy implementation.
        """
        if points.shape[0] < self.min_inliers:
            return None
            
        n = points.shape[0]
        best_plane = None
        best_inliers = -1
        
        # Subsample for speed if cloud is huge
        if n > 5000:
            idx = np.random.choice(n, size=5000, replace=False)
            sample_points = points[idx]
        else:
            sample_points = points

        # We need 3 points to define a plane
        # Generate many sets of 3 random indices
        # Vectorized approach:
        # 1. Pick K sets of 3 points
        # 2. Compute normal vectors
        # 3. Compute D
        # 4. Check inliers for all planes against ALL sample points (or a subset)
        
        # Simplified iterative approach is often fast enough for 50 iters
        for _ in range(self.ransac_iters):
             # Pick 3 random points
             idx = np.random.randint(0, sample_points.shape[0], 3)
             p1, p2, p3 = sample_points[idx]
             
             # Vector 1: p2 - p1
             # Vector 2: p3 - p1
             v1 = p2 - p1
             v2 = p3 - p1
             
             # Cross product for normal
             normal = np.cross(v1, v2)
             norm = np.linalg.norm(normal)
             if norm == 0: continue
             
             normal = normal / norm
             a, b, c = normal
             d = -np.dot(normal, p1)
             
             # Heuristic: Ground plane normal should somewhat point "up" in world frame
             # In camera frame (Y down), ground normal usually points roughly (0, -1, 0) if camera is level,
             # or some mix of Y and Z if tilted.
             # If normal is horizontal (like a wall), ignore it.
             # We assume camera is < 45 deg tilt.
             # Y component (b) should be significant?
             # Actually, let's just count inliers first.
             
             # usage: distance = |Ax + By + Cz + D| (since normal is normalized)
             # Batch compute distances
             dists = np.abs(np.dot(sample_points, normal) + d)
             count = np.count_nonzero(dists < self.ransac_thresh)
             
             if count > best_inliers:
                 best_inliers = count
                 best_plane = np.array([a, b, c, d])

        return best_plane

    def process_frame(self, depth_mm: np.ndarray) -> Dict:
        """
        Main pipeline:
        1. Convert ROI (bottom half) to point cloud
        2. Fit Ground Plane
        3. Classify all pixels (Obstacle / Ground / Dropoff)
        4. Return navigation hints
        """
        h, w = depth_mm.shape
        
        # 1. Focus on bottom 2/3rds for ground fitting to saving compute and avoid ceilings
        roi_y = int(h * 0.4)
        depth_roi = depth_mm[roi_y:, :]
        
        # We need to adjust v_grid for the crop if we use the reproject method
        # Or just pass the crop offset to reproject logic?
        # Let's just pass the full image but mask it?
        # No, reproject uses self.v_grid. Let's make a temporary cloud generator for full img?
        # Efficient way:
        points, mask_valid = self.reproject(depth_mm) 
        
        # Filter points by Y coordinate (in image space? or 3D space?)
        # Let's rely on the mask we created.
        # mask_valid is boolean mask of shape (N_valid,).
        # We need to know which pixels correspond to which 3D point to reconstruct semantic image.
        
        # Let's simplify:
        # Just fit plane on subsampled bottom-half points.
        # But for OBSTACLE detection, we need to check the WHOLE image (or relevant volume).
        
        # 2. Fit Plane
        # Select candidate points for ground (y > something in image, i.e., lower half)
        # In flattened mask arrays, we need to know original U,V
        valid_u = self.u_grid[mask_valid]
        valid_v = self.v_grid[mask_valid]
        
        # Candidates for ground: lower 60% of image
        ground_candidates_mask = valid_v > (0.4 * h)
        ground_points = points[ground_candidates_mask]
        
        plane = self.fit_plane_ransac(ground_points)
        
        # Temporal smoothing of plane equation
        if plane is not None:
            # Enforce "up" normal consistency (flip if needed so B is negative, i.e., pointing up in Cam frame Y-down)
            # Normal (a,b,c). Camera Y is DOWN. Ground normal should point UP (negative Y).
            if plane[1] > 0: 
                plane = -plane
            
            if self.prev_plane is not None:
                # Lerp
                self.prev_plane = 0.8 * self.prev_plane + 0.2 * plane
                # Re-normalize normal
                n = self.prev_plane[:3]
                d = self.prev_plane[3]
                n_norm = np.linalg.norm(n)
                self.prev_plane[:3] = n / n_norm
                self.prev_plane[3] = d / n_norm
            else:
                self.prev_plane = plane
        
        final_plane = self.prev_plane
        
        debug_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 3. Classify
        if final_plane is not None:
            a, b, c, d = final_plane
            
            # Distance of ALL valid points to plane
            # signed distance: (Ax + By + Cz + D)
            # If normal points UP (negative Y), then:
            # - Points ON plane: dist ~ 0
            # - Points ABOVE plane (obstacles, negative Y): dist > 0?
            # Let's check: P=(0, -1000, 1000) (point in air). Normal=(0, -1, 0).
            # dot = 1000. D ~ height. 
            # We need to rely on absolute height check or signed distance.
            
            dists = np.dot(points, final_plane[:3]) + final_plane[3]
            
            # Ground Inliers: abs(dist) < 80mm
            inliers = np.abs(dists) < 80
            
            # Obstacles: Significantly ABOVE ground.
            # "Above" in camera frame (Y-down) means dot product is Positive (if normal points UP).
            # We increase threshold to 12cm to avoid carpet/bumps being obstacles.
            obstacles = dists > 120 # > 12cm above ground

            
            # Dropoffs:
            # This is trickier in point cloud. Dropoff means "missing ground" or "ground too far down".
            # Points that are significantly BELOW the plane (dist < -200mm) are effectively "holes" or lower stairs.
            dropoffs_deep = dists < -300
            
            # We also have "Missing Data" dropoffs (where Z suggests we should see ground but don't).
            # Those are harder to vectorise efficiently. 
            
            # Visualize
            # Map back to image pixels
            # valid_u, valid_v are coordinates
            
            # Optimization: Assign colors to flat array then fill image
            colors = np.zeros_like(points, dtype=np.uint8)
            
            # Obstacles = Red
            colors[obstacles] = self.color_obs
            
            # Ground = Green
            colors[inliers] = self.color_ground
            
            # Deep/Below = Blue
            colors[dropoffs_deep] = (255, 0, 0) # Blue
            
            # Fill debug image
            # Slicing is faster
            debug_img[valid_v, valid_u] = colors
            
            # 4. Generate Nav Logic
            # We need to aggregate this into "Left / Center / Right" safety
            # Divide into 3 columns
            col_w = w // 3
            
            sections = {
                'L': (0, col_w),
                'C': (col_w, 2*col_w),
                'R': (2*col_w, w)
            }
            results = {}
            min_dist = {}
            
            for key, (u1, u2) in sections.items():
                # Mask for this column
                col_mask = (valid_u >= u1) & (valid_u < u2)
                
                # Check obstacles in this column
                col_obs_mask = obstacles & col_mask
                
                # Check proximity
                # Only care about obstacles closer than 2.0m ?
                col_obs_indices = np.where(col_obs_mask)[0]
                
                if len(col_obs_indices) > 50: # Noise filter
                    near_obs_z = np.min(points[col_obs_indices, 2])
                    min_dist[key] = near_obs_z
                    results[key] = 'blocked' if near_obs_z < 1500 else 'free'
                else:
                    results[key] = 'free'
                    min_dist[key] = 9999.0
            
            # Dropoff detection (Global or per column?)
            # Global check: if significant number of "deep" pixels in bottom half
            # But deep pixels are "dists < -300".
            # We should check if they are "in front" of us (bottom center).
            
            # Mask for "bottom center" region (critical for walking)
            # u: w/3 to 2w/3. v: > h*0.6
            critical_mask = (valid_u > w//3) & (valid_u < 2*w//3) & (valid_v > h*0.6)
            critical_dropoffs = dropoffs_deep & critical_mask
            
            is_dropoff = np.count_nonzero(critical_dropoffs) > 200
            
            return {
                'plane': final_plane,
                'debug_img': debug_img,
                'nav': results,
                'dists': min_dist,
                'dropoff': is_dropoff
            }
        
        else:
            return {
                'plane': None,
                'debug_img': debug_img,
                'nav': {'L':'unknown', 'C':'unknown', 'R':'unknown'},
                'dists': {'L':0, 'C':0, 'R':0},
                'dropoff': False
            }

