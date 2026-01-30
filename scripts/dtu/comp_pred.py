import torch
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

# Paths to two prediction methods
pred_root1 = Path("outputs/dtu_final_depths_method1")
pred_root2 = Path("outputs/dtu_final_depths_method2")

# Scans
scans = sorted(pred_root1.glob("scan*"))

all_l1 = []
all_rmse = []
all_ssim = []

for scan_path in scans:
    scan_name = scan_path.name
    view_files1 = sorted((pred_root1 / scan_name).glob("*.pth"))
    view_files2 = sorted((pred_root2 / scan_name).glob("*.pth"))

    for vf1, vf2 in zip(view_files1, view_files2):
        pred1 = torch.load(vf1).squeeze().numpy()
        pred2 = torch.load(vf2).squeeze().numpy()

        # Mask valid pixels if needed
        mask = (pred1 > 0) & (pred2 > 0)
        pred1_masked = pred1[mask]
        pred2_masked = pred2[mask]

        # L1 / MAE
        l1_val = np.mean(np.abs(pred1_masked - pred2_masked))
        all_l1.append(l1_val)

        # RMSE
        rmse_val = np.sqrt(np.mean((pred1_masked - pred2_masked) ** 2))
        all_rmse.append(rmse_val)

        # SSIM
        ssim_val = ssim(pred1, pred2, data_range=pred2.max() - pred2.min())
        all_ssim.append(ssim_val)

# Print overall metrics
print("Comparison between method1 and method2 (no GT):")
print(f"L1 / MAE: {np.mean(all_l1):.4f}")
print(f"RMSE    : {np.mean(all_rmse):.4f}")
print(f"SSIM    : {np.mean(all_ssim):.4f}")
