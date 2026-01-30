import sys
import os
sys.path.append(os.path.abspath("."))

import torch
from torch.utils.data import DataLoader
from datasets.blendedmvs import BlendedMVSDataset  # Adjust import path if needed

import matplotlib.pyplot as plt
import numpy as np
os.makedirs("viz", exist_ok=True)


def visualize_sample(sample):
    rgb_imgs = sample["imgs"]
    input_depths = sample["depth_list"]

    for i in range(len(rgb_imgs)):
        rgb = rgb_imgs[i][0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        rgb = rgb / 255.0 if rgb.max() > 1 else rgb

        depth = input_depths[i][0][0].cpu().numpy()  # (H, W)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(rgb)
        plt.title(f"RGB View {i}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(depth, cmap="viridis")
        plt.title(f"Input Depth View {i}")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
        plt.savefig(f"viz/sample_{i}_view_{i}.png")
        plt.close()

def test_loader():
    dataset = BlendedMVSDataset(
        root_dir="datasets/blendedmvs",
        list_file="datasets/blendedmvs/training_list.txt",
        split="train",
        n_views=5,
        img_wh=(768, 576),
        robust_train=False,
        augment=False
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for i, sample in enumerate(dataloader):
        print(f"Sample {i}")
        print(" - imgs:", [img.shape for img in sample["imgs"]])
        print(" - proj_matrices:", {k: v.shape for k, v in sample["proj_matrices"].items()})
        print(" - depth:", {k: v.shape for k, v in sample["depth"].items()})
        print(" - mask:", {k: v.shape for k, v in sample["mask"].items()})
        print(" - filename:", sample["filename"])
        if "depth_list" in sample:
            print(" - depth_list:", [d.shape for d in sample["depth_list"]])
        
        visualize_sample(sample)
        break


if __name__ == "__main__":
    test_loader()
