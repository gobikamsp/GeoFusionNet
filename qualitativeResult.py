import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# --------- PFM Loader ---------
def load_pfm(filename):
    with open(filename, "rb") as f:
        header = f.readline().decode("utf-8").rstrip()
        color = header == "PF"
        if not color and header != "Pf":
            raise Exception("Not a PFM file.")
        
        dim_line = f.readline().decode("utf-8").strip()
        while dim_line.startswith("#"):  # Skip comments
            dim_line = f.readline().decode("utf-8").strip()
        width, height = map(int, dim_line.split())

        scale = float(f.readline().decode("utf-8").strip())
        endian = "<" if scale < 0 else ">"
        scale = abs(scale)

        data = np.fromfile(f, endian + "f")
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data

# --------- Paths ---------
out_dir = "outputs"  # your output folder
save_dir = "viz_results"
os.makedirs(save_dir, exist_ok=True)

# --------- File matching ---------
jpg_files = sorted(glob(os.path.join(out_dir, "**", "*_ref.jpg"), recursive=True))

for jpg_path in jpg_files:
    base_name = os.path.splitext(os.path.basename(jpg_path))[0].replace("_ref", "")
    depth_path = jpg_path.replace("_ref.jpg", "_depth_est.pfm")
    conf_path = jpg_path.replace("_ref.jpg", "_confidence.pfm")

    if not (os.path.exists(depth_path) and os.path.exists(conf_path)):
        print(f"Skipping {jpg_path} (missing depth/conf)")
        continue

    # Load RGB
    rgb_img = cv2.cvtColor(cv2.imread(jpg_path), cv2.COLOR_BGR2RGB)

    # Load depth + normalize for colormap
    depth_map = load_pfm(depth_path)
    depth_norm = (depth_map - np.nanmin(depth_map)) / (np.nanmax(depth_map) - np.nanmin(depth_map) + 1e-8)

    # Load confidence + normalize
    conf_map = load_pfm(conf_path)
    conf_norm = (conf_map - np.nanmin(conf_map)) / (np.nanmax(conf_map) - np.nanmin(conf_map) + 1e-8)

    # --------- Plot & Save ---------
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(rgb_img)
    axs[0].set_title("Reference RGB")
    axs[0].axis("off")

    axs[1].imshow(depth_norm, cmap="viridis")
    axs[1].set_title("Predicted Depth")
    axs[1].axis("off")

    axs[2].imshow(conf_norm, cmap="inferno")
    axs[2].set_title("Confidence Map")
    axs[2].axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{base_name}_triplet.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")



