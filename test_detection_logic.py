import numpy as np
from depth_processor import DepthProcessor

def test_dropoff_detection():
    print("Testing DepthProcessor Drop-off Logic...")
    
    # 1. Initialize Processor
    dp = DepthProcessor(width=640, height=400)
    
    # 2. Create Synthetic Ground Plane (Flat floor at y=1000mm below camera)
    # Camera assumed at (0,0,0). Y is Down. 
    # Ground plane: y = 1000.  (Normal: 0, 1, 0, D=-1000 ? No, ax+by+cz+d=0)
    # y - 1000 = 0  => 0*x + 1*y + 0*z - 1000 = 0.
    
    # Let's generate depth image where Z = 1000 / (y_norm_pixel) etc.
    # Simple approx: 
    # y_world = (v - cy) * z / fy
    # z = y_world * fy / (v - cy)
    # Let y_world = 1000mm (floor)
    
    h, w = 400, 640
    depth = np.zeros((h, w), dtype=np.uint16)
    
    # Fill bottom half with ground
    for v in range(h//2, h):
        if v == dp.cy: continue
        try:
            val = 1000.0 * dp.fy / (v - dp.cy)
            z = min(65000, max(0, int(val)))
        except:
            z = 0
        depth[v, :] = z
        
    # Scenario A: Perfect Ground
    res = dp.process_frame(depth)
    print(f"Scenario A (Perfect Ground): Dropoff={res['dropoff']} (Expected: False)")
    
    # Scenario B: Cliff (Simulate dropoff by masking bottom center)
    # Missing data in critical region
    depth_cliff = depth.copy()
    # Mask bottom center
    # roi: w/3 to 2w/3, > h*0.6
    depth_cliff[int(h*0.6):, w//3 : 2*w//3] = 0
    
    res_cliff = dp.process_frame(depth_cliff)
    print(f"Scenario B (Open Cliff/Missing Ground): Dropoff={res_cliff['dropoff']} (Expected: True)")
    
    # Scenario C: Deep Drop-off (Visible points but far away/deep)
    depth_deep = depth.copy()
    # Make the "ground" in front much deeper (e.g. 2000mm down i.e. 1m drop)
    # This changes Z values. 
    # If Y increases to 2000, Z changes too? 
    # Let's just manually set Z to be "plane Z + something" ? 
    # No, DepthProcessor uses reproject.
    # Let's set Z such that Y_calculated is > 1000 + 300mm.
    # We want points at y_world = 1300mm (30cm drop).
    for v in range(int(h*0.6), h):
         if v == dp.cy: continue
         val = 1300.0 * dp.fy / (v - dp.cy)
         z = min(65000, max(0, int(val)))
         depth_deep[v, w//3 : 2*w//3] = z

    res_deep = dp.process_frame(depth_deep)
    print(f"Scenario C (Visible Deep Drop-off 30cm): Dropoff={res_deep['dropoff']} (Expected: True)")

    # Scenario D: Curb (Small Drop-off 19cm)
    depth_curb = depth.copy()
    for v in range(int(h*0.6), h):
         if v == dp.cy: continue
         val = 1190.0 * dp.fy / (v - dp.cy)  # 19cm drop
         z = min(65000, max(0, int(val)))
         depth_curb[v, w//3 : 2*w//3] = z

    res_curb = dp.process_frame(depth_curb)
    print(f"Scenario D (Visible Curb 19cm): Dropoff={res_curb['dropoff']} (Expected: True)")

if __name__ == "__main__":
    test_dropoff_detection()
