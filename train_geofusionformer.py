# -*- coding: utf-8 -*-
import warnings
import logging
warnings.filterwarnings("ignore") # This stops the autocast warnings from breaking tqdm
logging.getLogger("torch").setLevel(logging.ERROR)

import os
import sys
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import glob
import re

# -------------------------------------------------------------
# Ensure project root is on PYTHONPATH for internal imports
# -------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Environment variables for debugging and stability
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Import models - specific paths from your directory structure
from models.hybridfusionformer import HybridFusionFormer
from datasets.geofusion_dataset import GeoFusionDataset
from datasets.geofusion_dataset_dtu import GeoFusionDatasetDTU


# -------------------------------------------------------------
# Config Class: Centralized Settings Management
# -------------------------------------------------------------
class Config:
    def __init__(self, dataset="dtu", exp_name="run_1"):
        # Basic Setup
        self.dataset = dataset.lower()
        self.exp_name = exp_name

        # Training Hyperparameters
        self.depth_num = 64
        self.decoder_in_channels = 128
        self.batch_size = 1
        self.num_epochs = 50
        self.learning_rate = 5e-5
        self.accum_steps = 1
        
        # Hardware optimization settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_workers = 4
        self.pin_memory = True

        # Organized Path Management
        # Simplified save_dir to ensure resume finds top-level .pth files
        self.save_dir = "checkpoints"
        self.log_dir = os.path.join("logs", self.dataset, self.exp_name)
        
        # Model Safety and Architecture Constraints
        self.safe_input_hw = (384, 512)
        self.safe_max_tokens = 4096
        self.nhead = 2

        # Create required directories automatically
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Dataset-specific path and view configurations
        if self.dataset == "blendedmvs":
            self.dataset_root = "datasets/blendedmvs/"
            self.listfile = "datasets/blendedmvs/training_list.txt"
            self.val_listfile = "datasets/blendedmvs/training_list.txt"
            self.nviews = 5
        elif self.dataset == "dtu":
            self.dataset_root = "datasets/dtu_training/mvs_training/dtu"
            self.listfile = "lists/dtu/train.txt"
            self.val_listfile = "lists/dtu/test.txt"
            self.nviews = 3	    
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

# -------------------------------------------------------------
# Geometric Loss Utilities
# -------------------------------------------------------------
def compute_normal_loss(pred, gt):
    """Computes surface normal consistency with resolution alignment."""
    # Ensure inputs are 4D [B, 1, H, W]
    if pred.dim() == 3: pred = pred.unsqueeze(1)
    if gt.dim() == 3: gt = gt.unsqueeze(1)
    
    # Handle multi-channel depth maps if they exist
    if pred.shape[1] > 1: pred = pred[:, 0:1]
    if gt.shape[1] > 1: gt = gt[:, 0:1]

    # Align resolution: Interpolate GT to match prediction resolution
    if pred.shape[-2:] != gt.shape[-2:]:
        gt = F.interpolate(gt, size=pred.shape[-2:], mode='nearest')

    def depth_to_normal(d):
        """Internal helper to convert depth maps to normal vectors."""
        dz_dx = F.pad(d[:, :, :, 1:] - d[:, :, :, :-1], (0, 1, 0, 0))
        dz_dy = F.pad(d[:, :, 1:, :] - d[:, :, :-1, :], (0, 0, 0, 1))
        # Normal vector calculation
        n = torch.cat([-dz_dx, -dz_dy, torch.ones_like(dz_dx)], dim=1)
        return F.normalize(n, dim=1)

    # Apply mask for valid depth values
    mask = (gt > 0).float()
    pred_n = depth_to_normal(pred)
    gt_n = depth_to_normal(gt)
    
    # Cosine similarity loss (1 - cos_theta)
    normal_cos = (1.0 - (pred_n * gt_n).sum(1, keepdim=True)) * mask
    return normal_cos.sum() / (mask.sum() + 1e-8)

def depth_l1_loss(pred_depth, gt_depth):
    """Standard L1 loss with shape safety and resolution alignment."""
    if pred_depth.dim() == 4:
        pred_depth = pred_depth[:, 0]
    if gt_depth.dim() == 4:
        gt_depth = gt_depth[:, 0]

    # Align resolution for EPE calculation
    if pred_depth.shape[-2:] != gt_depth.shape[-2:]:
        gt_depth = F.interpolate(gt_depth.unsqueeze(1), size=pred_depth.shape[-2:], mode='nearest').squeeze(1)

    mask = gt_depth > 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_depth.device, requires_grad=True)

    return F.l1_loss(pred_depth[mask], gt_depth[mask])

# -------------------------------------------------------------
# Validation Function (FIXED Resolution Mismatch)
# -------------------------------------------------------------
@torch.no_grad()
@torch.no_grad() # CRITICAL: Prevents memory accumulation
def validate(model, dataloader, cfg):
    """Evaluates the model on validation set with memory optimization."""
    model.eval()
    total_epe = 0.0
    count = 0
    
    # Use a descriptive bar that won't get hidden
    pbar = tqdm(dataloader, desc="--> Validating", leave=False, dynamic_ncols=True)
    
    for batch in pbar:
        # Move tensors to device
        rgb, depth_in, proj_mats, depth_hypos = [
            b.to(cfg.device) if isinstance(b, torch.Tensor) else b for b in batch
        ]
        proj_mats = [p.to(cfg.device) for p in proj_mats]

        # Use Mixed Precision for validation too (saves memory)
        with torch.amp.autocast('cuda', enabled=(cfg.device == "cuda")):
            outputs = model(rgb, depth_in, proj_mats, depth_hypos)
            pred_depth = outputs["stage1"]["depth"]

        # Formatting
        if pred_depth.dim() == 4: pred_depth = pred_depth.squeeze(1)
        if gt_depth.dim() == 4: gt_depth = gt_depth.squeeze(1)

        # Resolution alignment
        curr_gt = gt_depth
        if pred_depth.shape[-2:] != gt_depth.shape[-2:]:
            curr_gt = F.interpolate(
                gt_depth.unsqueeze(1), 
                size=pred_depth.shape[-2:], 
                mode='nearest'
            ).squeeze(1)

        # Metric calculation
        mask = curr_gt > 0
        if mask.sum() > 0:
            epe = torch.abs(pred_depth[mask] - curr_gt[mask]).mean()
            total_epe += epe.item()
            count += 1
            # Update the description live so you see it working
            pbar.set_description(f"Validating | Curr EPE: {epe.item():.4f}")

        # Explicitly clean up large tensors to free GPU cache
        del outputs, pred_depth, curr_gt, mask
            
    return total_epe / max(1, count)
# -------------------------------------------------------------
# Main Training Logic
# -------------------------------------------------------------
def train():
    parser = argparse.ArgumentParser(description="Train GeoFusionFormer")
    parser.add_argument("--dataset", type=str, default="dtu", help="dtu or blendedmvs")
    parser.add_argument("--exp_name", type=str, default="run_1", help="experiment identifier")
    args = parser.parse_args()
    
    # Init Configuration
    cfg = Config(dataset=args.dataset, exp_name=args.exp_name)

    # Dataset Loader Initialization
    if cfg.dataset == "dtu":
        train_dataset = GeoFusionDatasetDTU(
            datapath=cfg.dataset_root,
            listfile=cfg.listfile,
            nviews=3,
	    ndepths=cfg.depth_num,  # <--- ADDED HERE
            use_input_depth=True,
            eval=False
        )
        val_dataset = GeoFusionDatasetDTU(
            datapath=cfg.dataset_root,
            listfile=cfg.val_listfile,
            nviews=3,
            ndepths=cfg.depth_num,  # <--- ADDED HERE
            use_input_depth=True,
            eval=True
        )
    else:
        train_dataset = GeoFusionDataset(
            datapath=cfg.dataset_root,
            listfile=cfg.listfile,
            nviews=cfg.nviews,
            ndepths=cfg.depth_num
        )
        val_dataset = train_dataset

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers, 
        pin_memory=cfg.pin_memory
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
  
    # Initialize Model
    model = HybridFusionFormer(
        cfg, 
        embed_dim=128, 
        base_channels=32, 
        safe_max_tokens=cfg.safe_max_tokens,
        safe_input_hw=cfg.safe_input_hw,
        nhead=cfg.nhead,
        skip_highres=True,
        debug=False
    ).to(cfg.device)

    #3. Optimizer and Mixed Precision Scaler
    #optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    # --- UPDATED: Use AdamW for Transformer stability ---
    # Weight decay (1e-2) prevents the attention weights from exploding, 
    # helping bring that 91.5 EPE down.
    
    # --- UPDATED CODE (Point 4: Weight Decay Exclusion) ---
    # Separate parameters into two groups: 
# 1. Weights (apply decay) 
# 2. Biases and LayerNorm scales (do NOT apply decay)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Do not decay biases or 1D parameters like LayerNorm weights
        if len(param.shape) == 1 or name.endswith(".bias") or "layer_norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer_grouped_parameters = [
        {'params': decay, 'weight_decay': 1e-2},
        {'params': no_decay, 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg.learning_rate)
    
    # --- ADDED: Scheduler for fine-tuning ---
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.5)
    
    scaler = torch.amp.GradScaler(device='cuda', enabled=(cfg.device == "cuda"))

    # ---------------------------------------------------------
    # 4.RESUME LOGIC: Automatic State Restoration
    # ---------------------------------------------------------
    start_epoch = 1
    # Locate existing checkpoints
    ckpt_pattern = os.path.join(cfg.save_dir, f"{cfg.dataset}_epoch_*.pth")
    list_of_files = glob.glob(ckpt_pattern)

    if list_of_files:
        # Extract and sort by epoch number for chronological loading
        def get_epoch_num(f):
            match = re.search(r'epoch_(\d+)', f)
            return int(match.group(1)) if match else -1
            
        list_of_files.sort(key=get_epoch_num)
        latest_file = list_of_files[-1]
        
        print(f"--- ATTEMPTING RESUME FROM: {latest_file} ---")
        try:
            checkpoint = torch.load(latest_file, map_location=cfg.device)
            # Restore model and optimizer
            model.load_state_dict(checkpoint.get('model_state', checkpoint.get('model_state_dict')))
            optimizer.load_state_dict(checkpoint.get('optimizer_state', checkpoint.get('optimizer_state_dict')))
            
            # --- ADDED: Restore Scheduler state ---
            if 'scheduler_state' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state'])
            # 3. Restore GradScaler state (Prevents NaN spikes on restart)
            if 'scaler_state' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state'])
            # Start loop from next epoch
            start_epoch = int(checkpoint['epoch']) + 1
            print(f"--- RESUME SUCCESSFUL. STARTING AT EPOCH {start_epoch} ---")
        except Exception as e:
            print(f"--- RESUME FAILED: {e}. STARTING FROM FRESH ---")
    else:
        print("--- NO PREVIOUS CHECKPOINTS DETECTED ---")

    # ---------------------------------------------------------
    # 5.PRIMARY TRAINING LOOP
    # ---------------------------------------------------------
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        step = 0
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}/{cfg.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            # 1. Data Unpacking and GPU Transfer
            if isinstance(batch, dict):
                rgb = batch['rgb'].to(cfg.device, non_blocking=True)
                depth_in = batch['depth'].to(cfg.device, non_blocking=True)
                gt_depth = batch['gt_depth'].to(cfg.device, non_blocking=True)
                depth_hypos = batch['depth_hypos'].to(cfg.device, non_blocking=True)
                proj_mats = [p.to(cfg.device) for p in batch['proj_mats']] # Extract from dict
            else:
                rgb, depth_in, proj_mats, depth_hypos, gt_depth = [
                    b.to(cfg.device) if isinstance(b, torch.Tensor) else b for b in batch
                ]
                proj_mats = [p.to(cfg.device) for p in proj_mats]

            # 2. Unit Scaling (mm to meters)
            with torch.no_grad():
                
                if depth_hypos.mean() > 100:
                    depth_hypos = depth_hypos / 1000.0
	    # --- 2. THE SAFETY SKIP (Add this now!) ---
            # If the entire batch is empty, skip it to avoid NaN loss
            if not valid_gt_mask.any():
                pbar.set_description_str(f"Epoch {epoch} | SKIPPED (empty GT)")
                continue

            # 3. Debug Check (Epoch 1, Batch 0 only)
            if batch_idx == 0:
                valid_gt = gt_depth[gt_depth > 0]
                gt_mean = valid_gt.mean().item() if valid_gt.numel() > 0 else 0
                print(f"\n[DEBUG] GT Mean: {gt_mean:.2f} | Hypo: {depth_hypos.min().item():.2f}-{depth_hypos.max().item():.2f}")
		
		# Check for critical mismatch
                if gt_mean > 0 and (gt_mean < depth_hypos.min().item() or gt_mean > depth_hypos.max().item()):
                    print("!!! CRITICAL WARNING: GT Depth is outside Hypothesis Range !!!")

            # 4. Forward and Loss
            optimizer.zero_grad()
            try:
                with torch.amp.autocast('cuda', enabled=(cfg.device == "cuda")):
                    outputs = model(rgb, depth_in, proj_mats, depth_hypos)

                    # Align prediction and GT
                    pred_depth = outputs["stage1"]["depth"].squeeze(1) if outputs["stage1"]["depth"].dim() == 4 else outputs["stage1"]["depth"]
                    depth_gt_ms = {"stage1": gt_depth}
                    mask_ms = {"stage1": (gt_depth > 0).float()}

                    
                    if epoch > 5:
                        # Add your smooth/normal loss here if desired
                        pass

                scaler.scale(loss / cfg.accum_steps).backward()

                if (batch_idx + 1) % cfg.accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                # 5. METRIC UPDATES
                epoch_loss += loss.item()
                step += 1
                
                # We move the info to the FRONT of the bar where it won't be cut off
                current_loss = loss.item()
                pbar.set_description_str(f"Epoch {epoch} | Loss {current_loss:.4f} | EPE {epe.item():.3f}")
                
                # Keep this for the end-of-line metadata
                pbar.set_postfix(lr=f"{optimizer.param_groups[0]['lr']:.1e}")
		# If you STILL don't see it, this will print every 50 steps regardless
                if batch_idx % 50 == 0:
                    tqdm.write(f"Batch {batch_idx}: Loss {loss.item():.4f}")
		# If the bar is still empty, this will prove the code is working
                    print(f" -> Batch {batch_idx}: Loss {current_loss:.4f}", end='\r')

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e        # --- Post-Training Epoch Update ---
        scheduler.step()

        # --- POST-EPOCH TASKS ---
        avg_loss = epoch_loss / max(1, step)

        # 1. Save Checkpoint IMMEDIATELY after training (Fixes Restart Loop)
        ckpt_path = os.path.join(cfg.save_dir, f"{cfg.dataset}_epoch_{epoch:03d}.pth")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), # ADDED
	    "scaler_state": scaler.state_dict(),        # Added
	    "loss": avg_loss,
        }, ckpt_path)
        print(f"Progress Saved: {ckpt_path}")
	# --- ADD THESE TWO LINES HERE ---
        torch.cuda.empty_cache()
        gc.collect()

        # 2. Execute Validation with Error Reporting
        val_epe = 0.0
        try:
            val_epe = validate(model, val_loader, cfg)
            print(f"Epoch {epoch} COMPLETE | Loss: {avg_loss:.4f} | Validation EPE: {val_epe:.4f}")
        except Exception as e:
            # Report failure clearly but allow training to move on because we saved progress
            print(f"--- VALIDATION FAILED AT EPOCH {epoch} ---")
            print(f"Error Details: {e}")
            print("------------------------------------------")

        # 3. Record metrics in logs
        with open(os.path.join(cfg.log_dir, "metrics.txt"), "a") as f:
            f.write(f"{epoch},{avg_loss},{val_epe}\n")

    print("End of Training Run.")

if __name__ == "__main__":
    train()
