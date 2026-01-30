import os
import gc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Helpful environment knobs (optional)
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # uncomment if fragmentation issues persist
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Import model and dataset
from models.hybridfusionformer import HybridFusionFormer
from datasets.geofusion_dataset import GeoFusionDataset

# -------------------------------------------------------------
# Config class for all training parameters
# -------------------------------------------------------------
class Config:
    def __init__(self):
        self.dataset_root = "/path/to/DTU"              # <-- change this
        self.listfile = "lists/dtu/train.txt"           # <-- change this
        self.depth_num = 48
        self.decoder_in_channels = 128
        self.batch_size = 1
        self.num_epochs = 10
        self.learning_rate = 1e-4
        self.save_dir = "checkpoints"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # safety settings (should match your model defaults)
        self.safe_input_hw = (384, 512)
        self.safe_max_tokens = 4096
        self.nhead = 2
        # optional gradient accumulation: set >1 to accumulate
        self.accum_steps = 1


# -------------------------------------------------------------
# Simple depth regression loss
# -------------------------------------------------------------
def hybridfusionformer_loss(pred_depth, gt_depth, mask=None):
    """
    L1 loss on valid pixels.
    """
    if mask is None:
        mask = gt_depth > 0
    # avoid empty mask
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_depth.device, requires_grad=True)
    loss = F.l1_loss(pred_depth[mask], gt_depth[mask])
    return loss


# -------------------------------------------------------------
# Training function (AMP, safe defaults, OOM handling)
# -------------------------------------------------------------
def train():
    cfg = Config()

    # Create dataset and dataloader
    dataset = GeoFusionDataset(
        datapath=cfg.dataset_root,
        listfile=cfg.listfile,
        nviews=2,
        ndepths=cfg.depth_num
    )
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize model (the model file already contains skip-highres and safe defaults)
    model = HybridFusionFormer(cfg, embed_dim=128, base_channels=32,
                               safe_max_tokens=cfg.safe_max_tokens,
                               safe_input_hw=cfg.safe_input_hw,
                               nhead=cfg.nhead,
                               skip_highres=True,
                               debug=False).to(cfg.device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.device == "cuda"))

    # Create checkpoint directory
    os.makedirs(cfg.save_dir, exist_ok=True)

    print(f"?? Training HybridFusionFormer on {cfg.device}")
    print(f"Dataset: {len(dataset)} samples, Depth hypotheses (capped): {model.cfg.depth_num}")

    # ---------------------------------------------------------
    # Epoch loop
    # ---------------------------------------------------------
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        step = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg.num_epochs}", leave=True)
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device (dataset must return tensors)
            try:
                rgb, depth_in, proj_mats, depth_hypos, gt_depth = batch
            except ValueError:
                # If dataset returns different format, adapt here
                rgb, depth_in, proj_mats, depth_hypos, gt_depth = batch

            # Downsample inputs to safe size if needed (defensive)
            if rgb.dim() == 4:
                H, W = rgb.shape[-2], rgb.shape[-1]
                safe_h, safe_w = cfg.safe_input_hw
                if H > safe_h or W > safe_w:
                    rgb = F.interpolate(rgb, size=(safe_h, safe_w), mode='bilinear', align_corners=False)
                    depth_in = F.interpolate(depth_in, size=(safe_h, safe_w), mode='nearest')
                    gt_depth = F.interpolate(gt_depth, size=(safe_h, safe_w), mode='bilinear', align_corners=False)

            rgb = rgb.to(cfg.device, non_blocking=True)
            depth_in = depth_in.to(cfg.device, non_blocking=True)
            gt_depth = gt_depth.to(cfg.device, non_blocking=True)
            depth_hypos = depth_hypos.to(cfg.device, non_blocking=True)
            # proj_mats may be a list of tensors
            proj_mats = [p.to(cfg.device) for p in proj_mats]

            # Pre-forward maintenance
            gc.collect()
            torch.cuda.empty_cache()

            # Forward + backward with AMP
            try:
                with torch.cuda.amp.autocast(enabled=(cfg.device == "cuda")):
                    pred_depth = model(rgb, depth_in, proj_mats, depth_hypos)
                    loss = hybridfusionformer_loss(pred_depth, gt_depth)

                # gradient accumulation support
                loss = loss / cfg.accum_steps
                scaler.scale(loss).backward()

                if (batch_idx + 1) % cfg.accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * cfg.accum_steps
                step += 1

                pbar.set_postfix({"loss": f"{(epoch_loss / max(1, step)):.6f}"})
            except RuntimeError as e:
                # OOM handling: skip this batch but don't crash training
                if 'out of memory' in str(e).lower():
                    print(f"[OOM] Skipping batch {batch_idx} due to OOM. Clearing cache and continuing.")
                    torch.cuda.empty_cache()
                    gc.collect()
                    # zero grads and continue
                    optimizer.zero_grad()
                    continue
                else:
                    # re-raise other exceptions
                    raise

        # Epoch summary
        avg_loss = epoch_loss / max(1, step)
        print(f"? Epoch [{epoch}/{cfg.num_epochs}] - Avg Loss: {avg_loss:.6f}")

        # Save checkpoint
        ckpt_path = os.path.join(cfg.save_dir, f"epoch_{epoch:03d}.pth")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, ckpt_path)
        print(f"?? Checkpoint saved: {ckpt_path}")

    print("?? Training complete!")


# -------------------------------------------------------------
if __name__ == "__main__":
    train()
