import sys
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

# 1. Setup paths so it can find 'datasets' and 'train_geofusionformer'
BASE_DIR = "/home/gobika/Research_Gobika/GeoFusionNet"
sys.path.append(BASE_DIR)

from datasets.geofusion_dataset_dtu import GeoFusionDatasetDTU
from train_geofusionformer import Config

def compute_metrics(pred, gt):
    pred = pred.cpu().squeeze()
    gt = gt.cpu().squeeze()
    
    # Handle resolution mismatch (e.g., if model outputs 1/4 resolution)
    if pred.shape != gt.shape:
        pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), 
                             size=gt.shape, 
                             mode='bilinear', 
                             align_corners=False).squeeze()

    # Valid mask (GT > 0)
    mask = (gt > 0) & (pred > 0)
    if mask.sum() == 0:
        return [0] * 7

    pred = pred[mask]
    gt = gt[mask]

    mae = torch.mean(torch.abs(pred - gt)).item()
    rmse = torch.sqrt(torch.mean((pred - gt) ** 2)).item()
    absrel = torch.mean(torch.abs(pred - gt) / gt).item()
    rmse_log = torch.sqrt(torch.mean((torch.log(pred.clamp(min=1e-6)) - torch.log(gt.clamp(min=1e-6))) ** 2)).item()

    thresh = torch.max(pred / gt, gt / pred)
    d1 = (thresh < 1.25).float().mean().item()
    d2 = (thresh < 1.25 ** 2).float().mean().item()
    d3 = (thresh < 1.25 ** 3).float().mean().item()

    return mae, rmse, absrel, rmse_log, d1, d2, d3

def main():
    # --- PATH CONFIGURATION ---
    # Path to the actual test list from your screenshot
    test_list = "/home/gobika/Research_Gobika/GeoFusionNet/datasets/dtu-test-1200/scan_list_test.txt"
    # Path to where the original DTU depth maps (GT) are stored
    # CHANGE THIS to your actual DTU data folder (where 'Depths' exists)
    gt_data_root = "/home/gobika/Research_Gobika/GeoFusionNet/datasets/dtu_training/mvs_training/dtu/" 
    # Path to your .pth prediction files
    pred_root = Path("/home/gobika/Research_Gobika/GeoFusionNet/outputs/dtu")
    # --------------------------

    dataset = GeoFusionDatasetDTU(
        datapath=gt_data_root,
        listfile=test_list,
        nviews=3,
        use_input_depth=True,
        eval=True
    )

    print(f"Dataset loaded with {len(dataset)} evaluation items.")
    
    # Map dataset metas for fast scan/view lookup
    meta_map = {}
    for i, meta in enumerate(dataset.metas):
        # meta[0] is scan name, meta[1] is view index
        key = f"{meta[0]}_{int(meta[1])}"
        meta_map[key] = i

    all_metrics = []
    per_scan_results = {}

    # Iterate through folders in outputs/dtu (scan1, scan4, etc.)
    for scan_dir in sorted(pred_root.glob("scan*")):
        scan_name = scan_dir.name
        scan_metrics = []
        print(f"Evaluating {scan_name}...")

        # Iterate through view.pth files
        for view_file in sorted(scan_dir.glob("*.pth")):
            # Extract view index from file name (e.g., '00000001.pth' -> 1)
            try:
                view_idx = int(view_file.stem)
            except:
                # Handle names like 'view01.pth'
                view_idx = int(''.join(filter(str.isdigit, view_file.stem)))

            lookup_key = f"{scan_name}_{view_idx}"
            
            if lookup_key in meta_map:
                idx = meta_map[lookup_key]
                # dataset[idx] returns: imgs, cams, depth_min, depth_max, gt_depth
                _, _, _, _, gt_depth = dataset[idx]
                
                # Load prediction
                pred_data = torch.load(view_file, weights_only=False)
                pred_depth = pred_data['pred_depth'] if isinstance(pred_data, dict) else pred_data
                
                metrics = compute_metrics(pred_depth, gt_depth)
                if metrics[0] > 0:
                    scan_metrics.append(metrics)
                    all_metrics.append(metrics)
            else:
                continue

        if scan_metrics:
            per_scan_results[scan_name] = np.mean(scan_metrics, axis=0)

    # Output Results
    if not all_metrics:
        print("Error: No matches found. Check if view indices in .pth files match test.txt.")
        return

    print("\n" + "="*50)
    print(f"{'Scan':<10} | {'MAE':<8} | {'RMSE':<8} | {'d1':<8}")
    print("-" * 50)
    for name, m in per_scan_results.items():
        print(f"{name:<10} | {m[0]:<8.4f} | {m[1]:<8.4f} | {m[4]:<8.4f}")
    
    overall = np.mean(all_metrics, axis=0)
    print("-" * 50)
    print(f"{'OVERALL':<10} | {overall[0]:<8.4f} | {overall[1]:<8.4f} | {overall[4]:<8.4f}")
    print("="*50)

if __name__ == "__main__":
    main()