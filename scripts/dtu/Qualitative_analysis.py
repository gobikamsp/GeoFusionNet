import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Paths
pred_root = Path("outputs/dtu_final_depths")        # fused predictions
input_root = Path("outputs/dtu_input_depths")      # optional: raw input depths
rgb_root = Path("datasets/dtu_training/dtu/Rectified")  # DTU RGB images
save_root = Path("outputs/qualitative")
save_root.mkdir(parents=True, exist_ok=True)

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

        # Load input depth (optional)
        input_file = input_root / scan_name / f"view{view_idx:02d}.pth"
        if input_file.exists():
            input_depth = torch.load(input_file).squeeze().numpy()
        else:
            input_depth = None

        # Load RGB image
        rgb_file = rgb_root / scan_name / f"rect_{view_idx:04d}.png"
        if rgb_file.exists():
            rgb = Image.open(rgb_file)
        else:
            rgb = None

        # Plot
        num_plots = 1 + (input_depth is not None) + (rgb is not None)
        plt.figure(figsize=(5*num_plots,5))

        idx = 1
        if input_depth is not None:
            plt.subplot(1,num_plots,idx)
            plt.imshow(input_depth, cmap="plasma")
            plt.title("Input Depth")
            plt.colorbar()
            idx += 1

        plt.subplot(1,num_plots,idx)
        plt.imshow(pred, cmap="plasma")
        plt.title("Predicted Depth")
        plt.colorbar()
        idx += 1

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
