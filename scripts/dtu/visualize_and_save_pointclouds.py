# -*- coding: utf-8 -*-

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
        f.write(b'Pf\n')
        f.write(f"{image.shape[1]} {image.shape[0]}\n".encode())
        f.write(f"{scale}\n".encode())
        np.flipud(image).astype(np.float32).tofile(f)

# ---------------------------------------------------------
# DTU camera loader
# ---------------------------------------------------------
def load_cam_txt(cam_file):
    with open(cam_file, "r") as f:
        lines = f.readlines()

    extrinsic = np.array(
        [[float(x) for x in lines[i].split()] for i in range(1, 5)],
        dtype=np.float32
    )

    intrinsic = np.array(
        [[float(x) for x in lines[i].split()] for i in range(7, 10)],
        dtype=np.float32
    )

    return intrinsic, extrinsic

# ---------------------------------------------------------
# Back-project depth to world
# ---------------------------------------------------------
def depth_to_world(depth, intrinsic, extrinsic):
    H, W = depth.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    z = depth.reshape(-1)
    x = (xs.reshape(-1) - cx) * z / fx
    y = (ys.reshape(-1) - cy) * z / fy

    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=0)
    pts_world = np.linalg.inv(extrinsic) @ pts_cam

    return pts_world[:3].T, z

# ---------------------------------------------------------
# Robust depth rescaling (CRITICAL FIX)
# ---------------------------------------------------------
def auto_rescale_depth(depth):
    """
    DTU scenes are ~0.5m  6m.
    If depth median is >> 10, scale it down robustly.
    """
    median = np.median(depth[depth > 0])
    if median > 10.0:
        scale = 1.0 / median
        print(f"[INFO] Auto-rescaling depth by factor {scale:.6f}")
        depth = depth * scale * 3.0  # bring median ~3m
    return depth

# ---------------------------------------------------------
# Process one view
# ---------------------------------------------------------
def process_view(pth_file, cam_dir, out_dir):
    view_id = os.path.splitext(os.path.basename(pth_file))[0]
    cam_id = view_id.zfill(8)
    cam_file = os.path.join(cam_dir, f"{cam_id}_cam.txt")

    if not os.path.isfile(cam_file):
        print(f"[WARN] Missing camera: {cam_file}")
        return

    intrinsic, extrinsic = load_cam_txt(cam_file)

    data = torch.load(pth_file, map_location="cpu", weights_only=True)
    if "pred_depth" not in data:
        print(f"[WARN] No pred_depth in {pth_file}")
        return

    depth = data["pred_depth"].squeeze().numpy().astype(np.float32)

    # Debug raw depth
    print(
        f"[DEBUG] {cam_id} RAW depth: "
        f"min={depth.min():.3f}, max={depth.max():.3f}, mean={depth.mean():.3f}"
    )

    # Auto-fix scale
    depth = auto_rescale_depth(depth)

    print(
        f"[DEBUG] {cam_id} FIXED depth: "
        f"min={depth.min():.3f}, max={depth.max():.3f}, mean={depth.mean():.3f}"
    )

    # Save PFM
    pfm_path = os.path.join(out_dir, f"{cam_id}.pfm")
    write_pfm(pfm_path, depth)
    print(f"[OK] Saved PFM: {pfm_path}")

    # Back-project
    points, depth_flat = depth_to_world(depth, intrinsic, extrinsic)

    # SAFE filter: keep all positive depth
    mask = depth_flat > 0
    points = points[mask]

    if len(points) == 0:
        print(f"[WARN] Still no valid points for view {cam_id}")
        return

    # Save PLY
    ply_path = os.path.join(out_dir, f"{cam_id}.ply")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(ply_path, pcd)

    print(f"[OK] Saved PLY: {ply_path} | Points: {len(points)}")

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth_dir", required=True)
    parser.add_argument("--cam_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pth_files = sorted(f for f in os.listdir(args.pth_dir) if f.endswith(".pth"))

    for f in pth_files:
        process_view(
            os.path.join(args.pth_dir, f),
            args.cam_dir,
            args.out_dir
        )

if __name__ == "__main__":
    main()
