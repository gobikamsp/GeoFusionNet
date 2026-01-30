import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
import torch.nn.functional as F

# =====================================================
# USER CONFIG (EDIT ONLY THIS)
# =====================================================
DTU_ROOT = "/home/gobika/Research_Gobika/GeoFusionNet/datasets/dtu_training/mvs_training/dtu"

RECTIFIED_DIR = os.path.join(DTU_ROOT, "Rectified")
CAMERA_DIR    = os.path.join(DTU_ROOT, "Cameras")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# DTU CONSTANTS
# =====================================================
MAX_VIEWS = 64            # DTU has exactly 64 cameras
NUM_SRC_VIEWS = 5
NUM_DEPTH_PLANES = 128    # keep moderate for speed
DEPTH_MIN = 400.0         # mm
DEPTH_MAX = 1000.0        # mm
NCC_WIN = 3

# =====================================================
# CAMERA LOADER (GLOBAL DTU CAMERAS)
# =====================================================
def load_cam(view_id):
    cam_path = os.path.join(CAMERA_DIR, f"{view_id:08d}_cam.txt")
    if not os.path.exists(cam_path):
        raise FileNotFoundError(cam_path)

    with open(cam_path) as f:
        lines = f.readlines()

    extrinsics = np.array(
        [list(map(float, lines[i].split())) for i in range(1, 5)],
        dtype=np.float32
    )

    intrinsics = np.array(
        [list(map(float, lines[i].split())) for i in range(7, 10)],
        dtype=np.float32
    )

    return (
        torch.from_numpy(intrinsics).to(DEVICE),
        torch.from_numpy(extrinsics).to(DEVICE)
    )

# =====================================================
# SOURCE VIEW SELECTION (DTU SAFE)
# =====================================================
def select_src_views(ref_id, total, k):
    ids = list(range(total))
    ids.remove(ref_id)
    return ids[:k]

# =====================================================
# PLANE SWEEP WARP
# =====================================================
def warp(src_img, K_src, T_src, K_ref, T_ref, depth_planes):
    _, _, H, W = src_img.shape

    y, x = torch.meshgrid(
        torch.arange(H, device=DEVICE),
        torch.arange(W, device=DEVICE),
        indexing="ij"
    )

    pix = torch.stack([x, y, torch.ones_like(x)], dim=0).float().view(3, -1)
    cam_ref = torch.inverse(K_ref) @ pix

    warped_all = []

    for d in depth_planes:
        pts = cam_ref * d
        pts = torch.cat([pts, torch.ones(1, pts.shape[1], device=DEVICE)], dim=0)

        world = torch.inverse(T_ref) @ pts
        cam_src = T_src @ world

        proj = K_src @ cam_src[:3]
        proj[:2] /= proj[2:3].clamp(min=1e-6)

        u = proj[0].view(H, W)
        v = proj[1].view(H, W)

        grid = torch.stack([
            2 * (u / (W - 1)) - 1,
            2 * (v / (H - 1)) - 1
        ], dim=-1)

        warped = F.grid_sample(
            src_img,
            grid.unsqueeze(0),
            align_corners=True,
            padding_mode="zeros"
        )

        warped_all.append(warped)

    return torch.cat(warped_all, dim=0)

# =====================================================
# NCC COST
# =====================================================
def ncc(ref, src):
    ref_m = F.avg_pool2d(ref, NCC_WIN, 1, NCC_WIN // 2)
    src_m = F.avg_pool2d(src, NCC_WIN, 1, NCC_WIN // 2)

    ref_v = F.avg_pool2d(ref**2, NCC_WIN, 1, NCC_WIN // 2) - ref_m**2
    src_v = F.avg_pool2d(src**2, NCC_WIN, 1, NCC_WIN // 2) - src_m**2

    cov = F.avg_pool2d(ref * src, NCC_WIN, 1, NCC_WIN // 2) - ref_m * src_m
    return cov / (torch.sqrt(ref_v * src_v) + 1e-6)

# =====================================================
# PROCESS ONE SCAN
# =====================================================
def process_scan(scan_dir):
    imgs = sorted(glob(os.path.join(scan_dir, "*.png")))
    if len(imgs) == 0:
        print(f"? Skipping empty {scan_dir}")
        return

    num_views = min(len(imgs), MAX_VIEWS)

    out_dir = os.path.join(scan_dir, "input_depth")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n? {os.path.basename(scan_dir)} | views: {num_views}")
    print(f"  Found {len(imgs)} image files")

    depth_planes = torch.linspace(
        DEPTH_MIN, DEPTH_MAX, NUM_DEPTH_PLANES, device=DEVICE
    )

    for ref_id in tqdm(range(num_views), leave=False):
        out_path = os.path.join(out_dir, f"{ref_id:08d}.png")

        ref_img = cv2.imread(imgs[ref_id])
        if ref_img is None:
            raise RuntimeError(f"Failed to read {imgs[ref_id]}")

        ref_img = torch.from_numpy(ref_img).float().permute(2, 0, 1) / 255.
        ref_img = ref_img.unsqueeze(0).to(DEVICE)

        K_ref, T_ref = load_cam(ref_id)

        cost = torch.zeros(
            NUM_DEPTH_PLANES,
            ref_img.shape[2],
            ref_img.shape[3],
            device=DEVICE
        )

        src_ids = select_src_views(ref_id, num_views, NUM_SRC_VIEWS)

        for sid in src_ids:
            src_img = cv2.imread(imgs[sid])
            if src_img is None:
                raise RuntimeError(f"Failed to read {imgs[sid]}")

            src_img = torch.from_numpy(src_img).float().permute(2, 0, 1) / 255.
            src_img = src_img.unsqueeze(0).to(DEVICE)

            K_src, T_src = load_cam(sid)

            warped = warp(src_img, K_src, T_src, K_ref, T_ref, depth_planes)
            ref_rep = ref_img.repeat(NUM_DEPTH_PLANES, 1, 1, 1)

            cost += ncc(ref_rep, warped).mean(1)

        best = torch.argmax(cost, dim=0)
        depth = depth_planes[best].cpu().numpy()

        depth_u16 = np.clip(
            (depth - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN) * 65535,
            0, 65535
        ).astype(np.uint16)

        ok = cv2.imwrite(out_path, depth_u16)
        assert ok, f"FAILED TO WRITE {out_path}"

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    scans = sorted(glob(os.path.join(RECTIFIED_DIR, "scan*_train")))
    print(f"Found {len(scans)} DTU scans")

    for scan in scans:
        process_scan(scan)

    print("\n? DTU input depth generation completed successfully")
