import os
import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d

def fuse_scan(scan_id, pth_root, output_dir):
    scan_path = os.path.join(pth_root, "dtu", f"scan{scan_id}")
    
    if not os.path.exists(scan_path):
        print(f"Skipping {scan_path}, folder not found.")
        return

    all_points = []
    pth_files = sorted([f for f in os.listdir(scan_path) if f.endswith('.pth')])
    
    print(f"Fusing Scan {scan_id} ({len(pth_files)} views)...")
    
    for pth in pth_files:
        file_path = os.path.join(scan_path, pth)
        data = torch.load(file_path, weights_only=False)
        
        # 1. Prepare Depth and Confidence
        depth_tensor = data['pred_depth'].cpu().squeeze()
        target_h, target_w = depth_tensor.shape 
        conf_tensor = data['confidence'].cpu().squeeze()
        
        # Collapse 3D prob volume to 2D
        if conf_tensor.dim() == 3 and conf_tensor.shape[0] > 1:
            conf_2d = torch.max(conf_tensor, dim=0)[0] 
        else:
            conf_2d = conf_tensor.squeeze()

        # Resample confidence to match depth map
        conf_4d = conf_2d.unsqueeze(0).unsqueeze(0)
        conf_resized = F.interpolate(
            conf_4d, size=(target_h, target_w), mode='bilinear', align_corners=False
        ).squeeze()

        depth = depth_tensor.numpy()
        conf = conf_resized.numpy()

        # --- FIX: Debugging the zero-point issue ---
        if pth == pth_files[0]: # Print range for the first view of each scan
            print(f"  > View 0 Conf Range: Min={conf.min():.6f}, Max={conf.max():.6f}")

        # --- FIX: Use a much lower threshold to start ---
        # If your Max is 0.05, a threshold of 0.2 will result in 0 points.
        threshold = 0.01 
        mask = conf > threshold
        
        # 2. Generate Coordinates
        v, u = np.meshgrid(np.arange(target_h), np.arange(target_w), indexing='ij')
        valid_depth = depth[mask]
        valid_u = u[mask]
        valid_v = v[mask]
        
        # 3. Back-projection Logic (Metric: mm)
        focal_length = 2892.3 
        cx, cy = target_w / 2, target_h / 2
        
        z = valid_depth
        x = (valid_u - cx) * z / focal_length
        y = (valid_v - cy) * z / focal_length
        
        points = np.stack([x, y, z], axis=-1)
        if len(points) > 0:
            all_points.append(points)

    if all_points:
        combined_points = np.concatenate(all_points, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_points)
        
        # FIX: Only run outlier removal if we actually have a significant number of points
        if len(pcd.points) > 100:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        out_file = os.path.join(output_dir, f"scan{scan_id}.ply")
        o3d.io.write_point_cloud(out_file, pcd)
        print(f"Successfully saved: {out_file} (Total Points: {len(pcd.points)})")
    else:
        print(f"!! Warning: No points passed the threshold for Scan {scan_id}")

if __name__ == "__main__":
    PTH_ROOT = "./outputs"    
    OUT_DIR = "./fused_results" 
    os.makedirs(OUT_DIR, exist_ok=True)
    
    test_scans = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]
    for sid in test_scans:
        fuse_scan(sid, PTH_ROOT, OUT_DIR)