import os
import argparse
import torch
import numpy as np
import open3d as o3d

# ---------------------------------------------------------
# PFM writer
# ---------------------------------------------------------
def write_pfm(filename, image, scale=1.0):
    with open(filename, 'wb') as f:
        f.write(b'PF\n' if image.ndim == 3 else b'Pf\n')
        f.write(f"{image.shape[1]} {image.shape[0]}\n".encode())
        f.write(f"{scale}\n".encode())
        np.flipud(image).astype(np.float32).tofile(f)

# ---------------------------------------------------------
# DTU camera file loader
# ---------------------------------------------------------
def load_cam_txt(cam_file):
    with open(cam_file, "r") as f:
        lines = f.readlines()
    extrinsic = np.array([[float(x) for x in lines[i].split()] for i in range(1,5)], dtype=np.float32)
    intrinsic = np.array([[float(x) for x in lines[i].split()] for i in range(7,10)], dtype=np.float32)
    return intrinsic, extrinsic

# ---------------------------------------------------------
# Depth back-projection
# ---------------------------------------------------------
def depth_to_world_points(depth, intrinsic, extrinsic):
    H, W = depth.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    fx, fy = intrinsic[0,0], intrinsic[1,1]
    cx, cy = intrinsic[0,2], intrinsic[1,2]
    z = depth
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1).reshape(-1,4).T
    pts_world = np.linalg.inv(extrinsic) @ pts_cam
    return pts_world[:3].T

# ---------------------------------------------------------
# Process single view
# ---------------------------------------------------------
def process_view(pth_file, cam_dir, view_id, out_dir, min_depth, max_depth):
    # Load predicted depth
    data = torch.load(pth_file, map_location="cpu", weights_only=True)
    if "pred_depth" not in data:
        print(f"[WARN] No 'pred_depth' in {pth_file}, skipping")
        return

    depth = data["pred_depth"].squeeze().numpy() / 1000.0  # meters

    # File names
    pfm_path = os.path.join(out_dir, f"scan_view{view_id:02d}.pfm")
    ply_path = os.path.join(out_dir, f"scan_view{view_id:02d}.ply")

    # Save PFM
    write_pfm(pfm_path, depth)
    print(f"[INFO] Saved PFM: {pfm_path}")

    # Load camera
    cam_file = os.path.join(cam_dir, f"{view_id:08d}_cam.txt")
    if not os.path.isfile(cam_file):
        print(f"[WARN] Camera file missing: {cam_file}, skipping")
        return
    intrinsic, extrinsic = load_cam_txt(cam_file)

    # Back-project
    points = depth_to_world_points(depth, intrinsic, extrinsic)

    # Depth filtering
    mask = (points[:,2] > min_depth) & (points[:,2] < max_depth)
    points = points[mask]

    # Save PLY
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"[OK] Saved PLY: {ply_path} (Points: {len(points)})")

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth_dir", required=True, help="Directory with predicted .pth files")
    parser.add_argument("--cam_dir", required=True, help="DTU Cameras directory")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--min_depth", type=float, default=0.4)
    parser.add_argument("--max_depth", type=float, default=4.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Loop over all 49 views
    for view_id in range(49):
        pth_file = os.path.join(args.pth_dir, f"{view_id:05d}.pth")
        if not os.path.isfile(pth_file):
            print(f"[WARN] Missing .pth file: {pth_file}, skipping")
            continue
        process_view(pth_file, args.cam_dir, view_id, args.out_dir, args.min_depth, args.max_depth)

if __name__ == "__main__":
    main()
