import os
import torch
from pathlib import Path
from tqdm import tqdm

# Paths
input_dir = Path("outputs/dtu")  # folder with 7616 .pth files
output_dir = Path("outputs/dtu_final_depths")  # clean output
output_dir.mkdir(parents=True, exist_ok=True)

# Scans and number of views
scans = [
    "scan1", "scan4", "scan9", "scan10", "scan11", "scan12", "scan13",
    "scan15", "scan23", "scan24", "scan29", "scan32", "scan33", "scan34",
    "scan48", "scan49", "scan62", "scan75", "scan77", "scan110", "scan114", "scan118"
]
num_views = 49  # each scan has 49 views

# Sort files
all_files = sorted(input_dir.glob("*.pth"))

if len(all_files) != len(scans) * num_views:
    print(f"Warning: Expected {len(scans) * num_views} files, found {len(all_files)}.")

# Map files to scans/views and extract pred_depth
file_idx = 0
for scan in scans:
    scan_dir = output_dir / scan
    scan_dir.mkdir(exist_ok=True)
    
    for view in range(num_views):
        if file_idx >= len(all_files):
            print(f"Ran out of files at scan {scan}, view {view}")
            break
        
        pth_file = all_files[file_idx]
        file_idx += 1
        
        data = torch.load(pth_file, map_location="cpu")
        pred_depth = data['pred_depth']  # only predicted depth
        
        out_file = scan_dir / f"view{view:02d}.pth"
        torch.save(pred_depth, out_file)

print("Done! Clean predicted depth maps saved in:", output_dir)
