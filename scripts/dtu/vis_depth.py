import os
import torch
import matplotlib.pyplot as plt

PRED_DIR = "outputs_dtu"
OUT_DIR = "vis_dtu"
os.makedirs(OUT_DIR, exist_ok=True)

def visualize(pred_depth, out_path):
    depth = pred_depth[0, 0].numpy()

    plt.figure(figsize=(5, 4))
    plt.imshow(depth, cmap="plasma")
    plt.colorbar()
    plt.title("Predicted Depth")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    files = sorted(f for f in os.listdir(PRED_DIR) if f.endswith(".pt"))

    for f in files:
        pred = torch.load(os.path.join(PRED_DIR, f))
        out_path = os.path.join(OUT_DIR, f.replace(".pt", ".png"))
        visualize(pred, out_path)

    print(f"Saved depth visualizations to {OUT_DIR}")

if __name__ == "__main__":
    main()
