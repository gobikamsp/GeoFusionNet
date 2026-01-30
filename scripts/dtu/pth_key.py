import torch
from pathlib import Path
from collections import Counter

# Path to your .pth files
input_dir = Path("outputs/dtu")  # adjust if needed
all_files = sorted(input_dir.glob("*.pth"))

key_counter = Counter()
file_keys_map = {}

for pth_file in all_files:
    try:
        data = torch.load(pth_file, map_location="cpu")
        if isinstance(data, dict):
            keys = list(data.keys())
        else:
            keys = ["<tensor_only>"]  # not a dict, just a tensor
        file_keys_map[pth_file.name] = keys
        key_counter.update(keys)
    except Exception as e:
        print(f"Error loading {pth_file.name}: {e}")

# Print summary
print("Unique keys across all files:")
for k, count in key_counter.items():
    print(f"{k}: {count} files")

# Optional: print per-file keys
print("\nSample mapping (first 20 files):")
for i, (fname, keys) in enumerate(file_keys_map.items()):
    if i >= 20:
        break
    print(f"{fname}: {keys}")
