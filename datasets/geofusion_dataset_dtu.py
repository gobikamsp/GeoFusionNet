# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
from PIL import Image

# ==================================================
# Utility: read PFM (GT depth, eval only)
# ==================================================
def read_pfm(file):
    with open(file, "rb") as f:
        header = f.readline().rstrip().decode("utf-8")
        color = header == "PF"

        width, height = map(int, f.readline().decode("utf-8").split())
        scale = float(f.readline().decode("utf-8"))
        endian = "<" if scale < 0 else ">"
        scale = abs(scale)

        data = np.fromfile(f, endian + "f")
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)  # Flip vertically

    return data, scale

# ==================================================
# DTU DATASET
# ==================================================
class GeoFusionDatasetDTU(Dataset):
    def __init__(
        self,
        datapath,
        listfile,
        nviews=2,
        ndepths=48,
        interval_scale=1.0,
        mode="train",
        use_input_depth=True,
        eval=False,
    ):
        super().__init__()

        self.datapath = datapath
        self.listfile = listfile
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.mode = mode
        self.use_input_depth = use_input_depth
        self.eval = eval

        self.rectified_dir = os.path.join(datapath, "Rectified")
        self.camera_dir = os.path.join(datapath, "Cameras")
        self.gt_depth_dir = os.path.join(datapath, "Depths")

        self.max_views = 64
        self.metas = self.build_list()
        self.scan_images = {}

        for scan, _ in self.metas:
            if scan not in self.scan_images:
                imgs = sorted(glob(os.path.join(self.rectified_dir, scan, "*.png")))
                if not imgs:
                    raise RuntimeError(f"No images found in {scan}")
                self.scan_images[scan] = imgs

    # --------------------------------------------------
    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = [line.strip() for line in f.readlines()]

        for scan in scans:
            imgs = sorted(glob(os.path.join(self.rectified_dir, scan, "*.png")))
            num_views = min(len(imgs), self.max_views)
            for ref_id in range(num_views):
                metas.append((scan, ref_id))
        return metas

    # --------------------------------------------------
    def load_cam(self, view_id):
        view_id = int(view_id)
        path_as_is = os.path.join(self.camera_dir, f"{view_id:08d}_cam.txt")
        path_minus_1 = os.path.join(self.camera_dir, f"{view_id-1:08d}_cam.txt")

        if os.path.exists(path_as_is):
            cam_path = path_as_is
        elif view_id > 0 and os.path.exists(path_minus_1):
            cam_path = path_minus_1
            print(f"[WARNING] Using fallback camera for view {view_id}")
        else:
            cam_path = os.path.join(self.camera_dir, "00000000_cam.txt")
            print(f"[WARNING] Using default camera for view {view_id}")

        with open(cam_path) as f:
            lines = [line.strip() for line in f.readlines()]

        extrinsic = np.fromstring(" ".join(lines[1:5]), sep=" ").reshape(4, 4)
        intrinsic = np.fromstring(" ".join(lines[7:10]), sep=" ").reshape(3, 3)
        depth_min, depth_interval = map(float, lines[11].split())

        # Validate shapes
        assert extrinsic.shape == (4, 4), f"Invalid extrinsic shape for view {view_id}"
        assert intrinsic.shape == (3, 3), f"Invalid intrinsic shape for view {view_id}"

        return intrinsic, extrinsic, depth_min, depth_interval

    # --------------------------------------------------
    def __getitem__(self, idx):
        scan, ref_id = self.metas[idx]
        img_list = self.scan_images[scan]

        # Select reference + source views
        offset = 1
        view_ids = [ref_id, (ref_id + offset) % len(img_list)]

        imgs = []
        proj_mats = []

        for i, vid in enumerate(view_ids):
            img_path = img_list[vid]
            img = np.array(Image.open(img_path).convert("RGB"), np.float32) / 255.0
            H, W = img.shape[:2]
            imgs.append(torch.from_numpy(img).permute(2, 0, 1))

            intrinsics, extrinsics, dmin, dint = self.load_cam(vid)

            # Scale intrinsics dynamically
            sx = W / 1600.0
            sy = H / 1200.0
            intrinsics = intrinsics.copy()
            intrinsics[0, 0] *= sx
            intrinsics[1, 1] *= sy
            intrinsics[0, 2] *= sx
            intrinsics[1, 2] *= sy

            K = torch.from_numpy(intrinsics).float()
            T = torch.from_numpy(extrinsics).float()

            proj_mats.append({"K": K, "T_c2w": T})

            if i == 0:
                depth_min = dmin / 1000.0
                depth_interval = dint / 1000.0

        # Depth hypotheses
        depth_hypos = torch.linspace(
            depth_min,
            depth_min + (self.ndepths - 1) * depth_interval,
            self.ndepths,
        ).clamp(min=0.0)

        # Input depth prior
        if self.use_input_depth:
            depth_path = os.path.join(self.rectified_dir, scan, "input_depth", f"{ref_id:08d}.png")
            if os.path.exists(depth_path):
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
                depth = torch.from_numpy(depth).unsqueeze(0)
                depth = torch.clamp(depth, min=0.0)  # <-- safe clamp
            else:
                depth = torch.full((1, H, W), depth_min)
        else:
            depth = torch.full((1, H, W), depth_min)

        # GT depth (eval only)
        gt_depth = torch.zeros_like(depth)
        gt_path = os.path.join(self.gt_depth_dir, scan, f"depth_map_{ref_id:04d}.pfm")
        if os.path.exists(gt_path):
            gt, _ = read_pfm(gt_path)
            gt_depth = torch.from_numpy(gt.copy()).unsqueeze(0)
            gt_depth = torch.nan_to_num(gt_depth, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "ref_img": imgs[0],        # [3,H,W]
            "src_img": imgs[1],        # [3,H,W]
            "depth": depth,            # [1,H,W]
            "proj_mats": proj_mats,    # [nviews, {"K", "T_c2w"}]
            "depth_hypos": depth_hypos,
            "gt_depth": gt_depth,
        }

    # --------------------------------------------------
    def __len__(self):
        return len(self.metas)
