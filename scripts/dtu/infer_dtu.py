# -*- coding: utf-8 -*-

import os
import sys
import torch
from torch.utils.data import DataLoader

# -------------------------------------------------
# Project Root
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, PROJECT_ROOT)

# -------------------------------------------------
# Imports
# -------------------------------------------------
from train_geofusionformer import Config
from models.hybridfusionformer import HybridFusionFormer
from datasets.geofusion_dataset_dtu import GeoFusionDatasetDTU

def build_depth_hypos(batch_size, device, depth_num):
    """Build depth hypotheses required by HybridFusionFormer."""
    depth_min = 425.0
    depth_max = 935.0
    return torch.linspace(
        depth_min,
        depth_max,
        depth_num,
        device=device
    ).unsqueeze(0).repeat(batch_size, 1)

def main():
    cfg = Config(dataset="dtu")
    device = cfg.device
    os.makedirs(cfg.output_dir, exist_ok=True)

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------
    dataset = GeoFusionDatasetDTU(
        datapath=cfg.dataset_root,
        listfile=cfg.listfile,
        nviews=cfg.nviews,
        use_input_depth=True,
        eval=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------
    model = HybridFusionFormer(
        cfg,
        embed_dim=128,
        base_channels=32,
        safe_max_tokens=cfg.safe_max_tokens,
        safe_input_hw=cfg.safe_input_hw,
        nhead=cfg.nhead,
        skip_highres=True,
        debug=False,
    ).to(device)

    model.eval()

    # -------------------------------------------------
    # Load checkpoint
    # -------------------------------------------------
    ckpt_path = os.path.join(cfg.save_dir, "epoch_050.pth")  
    print(f"Loading checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)

    # -------------------------------------------------
    # Inference Loop
    # -------------------------------------------------
    # Official DTU Evaluation Scans
    test_scans = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]
    
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            
            # --- FIX: Calculation of Scan Index ---
            # 49 views per scan is the DTU standard
            scan_idx = idx // 49 
            
            # --- FIX: Stop after the 22 test scans to avoid IndexError ---
            if scan_idx >= len(test_scans):
                print(f"Reached end of 22 test scans (Index {idx}). Stopping.")
                break

            if isinstance(batch, dict):
                rgb = batch["image"]
                depth_in = batch["input_depth"]
                gt_depth = batch.get("gt_depth", None)
                
                # --- FIX: Robust Naming Logic ---
                if 'filename' in batch:
                    # Tries to get name from dataset path string
                    full_path = batch['filename'][0]
                    scan_name = full_path.split('/')[0]
                    view_id = os.path.basename(full_path).split('_')[0]
                else:
                    # Fallback to the manual list if filename isn't provided
                    scan_name = f"scan{test_scans[scan_idx]}"
                    view_id = f"{idx % 49:08d}"

                K = batch["K"]      # (B,3,3)
                T = batch["T"]      # (B,4,4)
                B = rgb.shape[0]

                proj_3x4 = torch.matmul(K, T[:, :3, :])
                bottom = torch.tensor([0, 0, 0, 1], dtype=proj_3x4.dtype, device=proj_3x4.device).view(1, 1, 4).repeat(B, 1, 1)
                proj_4x4 = torch.cat([proj_3x4, bottom], dim=1)
                proj_mats = [proj_4x4]

                depth_hypos = build_depth_hypos(B, device, cfg.depth_num)
            else:
                # Fallback for tuple-based batching
                rgb, depth_in, proj_mats, depth_hypos, gt_depth = batch
                scan_name = f"scan{test_scans[scan_idx]}"
                view_id = f"{idx % 49:08d}"

            # ---- move to device ----
            rgb = rgb.to(device)
            depth_in = depth_in.to(device)
            depth_hypos = depth_hypos.to(device)
            proj_mats = [p.to(device) for p in (proj_mats if isinstance(batch, dict) else proj_mats[0])]

            # ---- forward ----
            # We must capture 'prob' as confidence for filtering noisy points
            pred_depth, prob = model(rgb, depth_in, proj_mats, depth_hypos)

            # ---- SAVE ORGANIZED OUTPUT ----
            out = {
                "pred_depth": pred_depth.cpu(),
                "confidence": prob.cpu() # Vital for the 0.33mm accuracy target
            }
            if gt_depth is not None:
                out["gt_depth"] = gt_depth.cpu()

            # Create folder: output_dir/scan1/
            save_dir = os.path.join(cfg.output_dir, scan_name)
            os.makedirs(save_dir, exist_ok=True)
            
            # Save as: output_dir/scan1/00000000.pth
            out_path = os.path.join(save_dir, f"{view_id}.pth")
            torch.save(out, out_path)

            if idx % 10 == 0:
                print(f"[{idx+1}/{len(loader)}] Processing {scan_name} | Saved View {view_id}")

    print("Inference complete. Data organized for Fusion.")

if __name__ == "__main__":
    main()