import open3d as o3d
import numpy as np
import sys
import os

def main(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)

    pts = np.asarray(pcd.points)
    print(f"[INFO] Num points: {pts.shape[0]}")

    print(f"[INFO] XYZ min: {pts.min(axis=0)}")
    print(f"[INFO] XYZ max: {pts.max(axis=0)}")
    print(f"[INFO] XYZ mean: {pts.mean(axis=0)}")

    bbox = pcd.get_axis_aligned_bounding_box()
    print(f"[INFO] Bounding box extent: {bbox.get_extent()}")

    # Density sanity
    nn_dist = pcd.compute_nearest_neighbor_distance()
    print(f"[INFO] Mean NN distance: {np.mean(nn_dist):.6f}")

if __name__ == "__main__":
    main(sys.argv[1])
