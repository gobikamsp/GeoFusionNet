import os
import torch
import torch.nn.functional as F
import argparse
import csv

def compute_per_pixel_mae(pred, gt, valid_mask=None):
    """
    Robust MAE computation for DTU-style evaluation.
    - pred is resized to GT resolution
    - mask is resized to GT resolution
    - safe for any resolution mismatch
    """

    pred = pred.float()
    gt = gt.float()

    # 1?? Resize prediction to GT resolution
    if pred.shape[2:] != gt.shape[2:]:
        pred = F.interpolate(
            pred,
            size=gt.shape[2:],
            mode="bilinear",
            align_corners=False
        )

    # 2?? Prepare mask
    if valid_mask is not None:
        valid_mask = valid_mask.float()
        if valid_mask.shape[2:] != gt.shape[2:]:
            valid_mask = F.interpolate(
                valid_mask,
                size=gt.shape[2:],
                mode="nearest"
            )
    else:
        valid_mask = torch.ones_like(gt)

    # 3?? Masked MAE (NO boolean indexing)
    diff = torch.abs(pred - gt) * valid_mask
    mae = diff.sum() / valid_mask.sum()

    return mae.item()

def evaluate_pth_file(pth_path):
    data = torch.load(pth_path, map_location="cpu")

    # Common key guesses (robust)
    pred_keys = ["pred_depth", "prediction", "pred"]
    gt_keys = ["gt_depth", "depth_gt", "gt"]

    pred = next((data[k] for k in pred_keys if k in data), None)
    gt = next((data[k] for k in gt_keys if k in data), None)

    if pred is None:
        raise KeyError("Prediction not found")
    if gt is None:
        raise KeyError("Ground truth not found")

    valid_mask = data.get("valid_mask", None)

    return compute_per_pixel_mae(pred, gt, valid_mask)

def main(pth_dir, output_csv):
    results = []
    maes = []

    files = sorted(f for f in os.listdir(pth_dir) if f.endswith(".pth"))

    for fname in files:
        path = os.path.join(pth_dir, fname)
        try:
            mae = evaluate_pth_file(path)
            print(f"{fname}: MAE = {mae:.6f}")
            maes.append(mae)
            results.append({"file": fname, "MAE": mae, "error": ""})
        except Exception as e:
            print(f"{fname}: ERROR -> {e}")
            results.append({"file": fname, "MAE": None, "error": str(e)})

    # Save CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "MAE", "error"])
        writer.writeheader()
        writer.writerows(results)

    # Final baseline MAE
    if maes:
        avg_mae = sum(maes) / len(maes)
        print("\n==============================")
        print(f"Baseline MAE (mean over {len(maes)} views): {avg_mae:.6f}")
        print("==============================")
    else:
        print("\nNo valid files found for MAE computation.")

    print(f"\nSaved results to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth_dir", required=True)
    parser.add_argument("--output_csv", default="baseline_results.csv")
    args = parser.parse_args()

    main(args.pth_dir, args.output_csv)
