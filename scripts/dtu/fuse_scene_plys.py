import open3d as o3d
import glob
import sys
import numpy as np
import os

def fuse_scene(ply_dir, out_ply):
    ply_files = sorted(glob.glob(os.path.join(ply_dir, "*.ply")))
    print(f"[INFO] Fusing {len(ply_files)} views")

    all_points = []

    for ply in ply_files:
        pcd = o3d.io.read_point_cloud(ply)
        pts = np.asarray(pcd.points)
        if pts.shape[0] == 0:
            continue
        all_points.append(pts)

    if len(all_points) == 0:
        print("[ERROR] No valid points found")
        return

    all_points = np.concatenate(all_points, axis=0)

    fused = o3d.geometry.PointCloud()
    fused.points = o3d.utility.Vector3dVector(all_points)

    # DTU standard voxel size: 2mm
    fused = fused.voxel_down_sample(voxel_size=0.002)

    o3d.io.write_point_cloud(out_ply, fused)
    print(f"[OK] Saved fused PLY: {out_ply}")
    print(f"[INFO] Final point count: {len(fused.points)}")

if __name__ == "__main__":
    fuse_scene(sys.argv[1], sys.argv[2])
