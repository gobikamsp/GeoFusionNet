import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np

# ----------------------------
# User-configurable parameters
# ----------------------------
pred_root = Path("outputs/dtu_final_depths")        # fused predictions
input_root = Path("outputs/dtu_input_depths")      # optional: input depths
rgb_root = Path("datasets/dtu_training/dtu/Rectified")  # DTU RGB images
save_root = Path("outputs/qualitative")
save_root.mkdir(parents=True, exist_ok=True)

# Depth display settings
min_depth = 0          # minimum depth for clipping
max_depth = 1000       # maximum depth for clipping (mm or meters depending on dataset)
use_log_scale = True   # whether to apply log(1 + depth) scaling for visualization
# ----------------------------

scans = sorted(pred_root.glob("scan*"))

for scan_path in scans:
    scan_name = scan_path.name
    save_scan_dir = save_root / scan_name
    save_scan_dir.mkdir(exist_ok=True)

    view_files = sorted(scan_path.glob("*.pth"))

    for view_file in view_files:
        view_idx = int(view_file.stem.replace("view",""))

        # Load predicted depth
        pred = torch.load(view_file).squeeze().numpy()

        # Clip depth to avoid colormap saturation
        pred = np.clip(pred, min_depth, max_depth)

        # Optional log-scaling
        if use_log_scale:
            pred = np.log1p(pred)

        # Normalize to [0,1] for visualization
        pred_vis = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

        # Load input depth (optional)
        input_depth_file = input_root / scan_name / f"view{view_idx:02d}.pth"
        if input_depth_file.exists():
            input_depth = torch.load(input_depth_file).squeeze().numpy()
            input_depth = np.clip(input_depth, min_depth, max_depth)
            if use_log_scale:
                input_depth = np.log1p(input_depth)
            input_vis = (input_depth - input_depth.min()) / (input_depth.max() - input_depth.min() + 1e-8)
        else:
            input_vis = None

        # Load RGB image (optional)
        rgb_file = rgb_root / scan_name / f"rect_{view_idx:04d}.png"
        if rgb_file.exists():
            rgb = Image.open(rgb_file)
        else:
            rgb = None

        # Determine number of subplots
        num_plots = 1 + (input_vis is not None) + (rgb is not None)
        plt.figure(figsize=(5*num_plots,5))

        # Plot input depth
        idx = 1
        if input_vis is not None:
            plt.subplot(1,num_plots,idx)
            plt.imshow(input_vis, cmap="plasma")
            plt.title("Input Depth")
            plt.colorbar()
            idx += 1

        # Plot predicted depth
        plt.subplot(1,num_plots,idx)
        plt.imshow(pred_vis, cmap="plasma")
        plt.title("Predicted Depth")
        plt.colorbar()
        idx += 1

        # Plot RGB image
        if rgb is not None:
            plt.subplot(1,num_plots,idx)
            plt.imshow(rgb)
            plt.title("RGB")
            idx += 1

        # Save figure
        save_file = save_scan_dir / f"view{view_idx:02d}.png"
        plt.tight_layout()
        plt.savefig(save_file)
        plt.close()

        print(f"Saved qualitative visualization: {save_file}")
