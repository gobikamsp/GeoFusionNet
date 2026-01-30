# -*- coding: utf-8 -*-

import os
import torch
import torch.nn.functional as F
import numpy as np

# ----------------------------
# Config
# ----------------------------
OUTPUT_DIR = "outputs/dtu"
GT_DIR = "datasets/dtu_training/mvs_training/dtu/Depths"
TEST_LIST = "lists/dtu/test.txt"

# ----------------------------
# Utility: read PFM
# ----------------------------
def read_pfm(file):
    with open(file, "rb") as f:
        header = f.readline().rstrip().decode("ascii")
        color = header == "PF"

        width, height = map(int, f.readline().decode("ascii").split())
        scale = float(f.readline().decode("ascii"))
        endian = "<" if scale < 0 else ">"
        scale = abs(scale)

        data = np.fromfile(f, endian + "f")
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data).copy()

    return data, scale

# ----------------------------
# Load GT depth (meters)
# ----------------------------
def load_gt_depth(path):
    depth, _ = read_pfm(path)
    return torch.from_numpy(depth).float()

# ----------------------------
# Scale + shift alignment (DTU standard)
# ----------------------------
def align_depth_scale_shift(pred, gt):
    mask = gt > 0
    if mask.sum() < 50:
        return pred

    pred_v = pred[mask]
    gt_v = gt[mask]

    A = torch.stack([pred_v, torch.ones_like(pred_v)], dim=1)
    x, _, _, _ = torch.linalg.lstsq(A, gt_v)
    scale, shift = x[0], x[1]

    return pred * scale + shift

# ----------------------------
# MAE (meters)
# ----------------------------
def compute_mae(pred, gt):
    mask = gt > 0
    if mask.sum() == 0:
        return None
    return torch.mean(torch.abs(pred[mask] - gt[mask])).item()

# ----------------------------
# DTU metrics (ASCII safe)
# ----------------------------
def compute_dtu_metrics(pred, gt):
    mask = gt > 0
    if mask.sum() < 50:
        return None

    pred = pred[mask]
    gt = gt[mask]

    absrel = torch.mean(torch.abs(pred - gt) / gt)
    rmse = torch.sqrt(torch.mean((pred - gt) ** 2))

    ratio = torch.max(pred / gt, gt / pred)
    d1 = (ratio < 1.25).float().mean()
    d2 = (ratio < (1.25 ** 2)).float().mean()
    d3 = (ratio < (1.25 ** 3)).float().mean()

    return (
        absrel.item(),
        rmse.item(),
        d1.item(),
        d2.item(),
        d3.item()
    )

# ----------------------------
# Main
# ----------------------------
def main():
    with open(TEST_LIST) as f:
        scans = [line.strip() for line in f.readlines()]

    pred_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".pth")])
    pred_idx = 0

    mae_all = []
    absrel_all = []
    rmse_all = []
    d1_all = []
    d2_all = []
    d3_all = []

    for scan in scans:
        scan_gt_dir = os.path.join(GT_DIR, scan)
        if not os.path.isdir(scan_gt_dir):
            continue

        gt_files = sorted([
            f for f in os.listdir(scan_gt_dir)
            if f.startswith("depth_map_") and f.endswith(".pfm")
        ])

        for view_id in range(len(gt_files)):
            if pred_idx >= len(pred_files):
                break

            pred_file = pred_files[pred_idx]
            pred_idx += 1

            out = torch.load(
                os.path.join(OUTPUT_DIR, pred_file),
                map_location="cpu",
                weights_only=True
            )

            pred = out["pred_depth"].squeeze(0).squeeze(0) / 1000.0

            gt_path = os.path.join(scan_gt_dir, f"depth_map_{view_id:04d}.pfm")
            if not os.path.exists(gt_path):
                continue

            gt = load_gt_depth(gt_path)

            if pred.shape != gt.shape:
                gt = F.interpolate(
                    gt.unsqueeze(0).unsqueeze(0),
                    size=pred.shape,
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0).squeeze(0)

            pred = align_depth_scale_shift(pred, gt)

            mae = compute_mae(pred, gt)
            if mae is not None:
                mae_all.append(mae)

            metrics = compute_dtu_metrics(pred, gt)
            if metrics is not None:
                absrel, rmse, d1, d2, d3 = metrics
                absrel_all.append(absrel)
                rmse_all.append(rmse)
                d1_all.append(d1)
                d2_all.append(d2)
                d3_all.append(d3)

    print("\n========= DTU BASELINE RESULTS =========")
    print(f"MAE (m)        : {np.mean(mae_all):.4f}")
    print(f"AbsRel         : {np.mean(absrel_all):.4f}")
    print(f"RMSE (m)       : {np.mean(rmse_all):.4f}")
    print(f"d1 (<1.25)     : {np.mean(d1_all):.3f}")
    print(f"d2 (<1.25^2)   : {np.mean(d2_all):.3f}")
    print(f"d3 (<1.25^3)   : {np.mean(d3_all):.3f}")
    print("=======================================")

if __name__ == "__main__":
    main()
