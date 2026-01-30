# -*- coding: utf-8 -*-
import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. Path fix: ensure script can find 'datasets' folder from root
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)

from datasets.geofusion_dataset_dtu import GeoFusionDatasetDTU

def visualize_sample(datapath, listfile):
    # 2. Initialize the dataset
    dataset = GeoFusionDatasetDTU(
        datapath=datapath,
        listfile=listfile,
        use_input_depth=True
    )

    # 3. Pull the first sample (usually scan1_train or scan100_train)
    ref_img, input_depth, proj, hypos, gt_depth = dataset[0]

    # Convert tensors to numpy for plotting
    img = ref_img.permute(1, 2, 0).numpy()
    # Squeeze out channel dimension (1, H, W) -> (H, W)
    in_depth_np = input_depth.squeeze().numpy()
    gt_depth_np = gt_depth.squeeze().numpy()

    # 4. Plotting
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title("Reference Image")
    plt.imshow(img)
    plt.axis('off')

    # Use 'jet' colormap and set fixed range to 425-935mm for consistency
    plt.subplot(1, 3, 2)
    plt.title("Input Depth (Scaled Prior)")
    plt.imshow(in_depth_np, cmap='jet', vmin=425, vmax=935)
    plt.colorbar(label='Millimeters (mm)')

    plt.subplot(1, 3, 3)
    plt.title("Ground Truth (PFM)")
    plt.imshow(gt_depth_np, cmap='jet', vmin=425, vmax=935)
    plt.colorbar(label='Millimeters (mm)')

    plt.tight_layout()
    plt.show()

    # Numerical verification printout
    print(f"Depth Range Check:")
    print(f"Input Min/Max: {in_depth_np.min():.2f} / {in_depth_np.max():.2f} mm")
    print(f"GT Min/Max:    {gt_depth_np.min():.2f} / {gt_depth_np.max():.2f} mm")

if __name__ == "__main__":
    DATA_PATH = "datasets/dtu_training/mvs_training/dtu"
    LIST_FILE = "lists/dtu/train.txt"
    visualize_sample(DATA_PATH, LIST_FILE)