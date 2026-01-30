import os
import sys
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from datasets.geofusion_dataset_dtu import GeoFusionDatasetDTU
from models.hybridfusionformer import HybridFusionFormer


# --------------------------------------------------
# Minimal config object (only what model needs)
# --------------------------------------------------
class Cfg:
    depth_num = 48                 # will be capped to 32 internally
    decoder_in_channels = 128      # MUST match training config


# --------------------------------------------------
# Dataset (DTU)
# --------------------------------------------------
dataset = GeoFusionDatasetDTU(
    datapath="datasets/dtu_training/mvs_training/dtu",
    listfile="lists/dtu/train.txt",
    nviews=3,
    use_input_depth=True,
    eval=False,
)

sample = dataset[0]

# --------------------------------------------------
# Build model
# --------------------------------------------------
cfg = Cfg()
model = HybridFusionFormer(cfg).cuda()
model.eval()

# --------------------------------------------------
# Prepare inputs
# --------------------------------------------------
rgb = sample["image"].unsqueeze(0).cuda()          # [1, 3, H, W]
depth_in = sample["input_depth"].unsqueeze(0).cuda()

K = sample["K"].cuda()      # [3, 3]
T = sample["T"].cuda()      # [4, 4]

# --------------------------------------------------
# Build projection matrix (K @ [R|t])
# --------------------------------------------------
proj = torch.zeros((1, 4, 4), device="cuda")
proj[:, :3, :4] = torch.matmul(K, T[:3, :4])
proj[:, 3, 3] = 1.0

# GeoFusion expects list-like proj_mats: [ref, src1, src2]
proj_mats = [proj, proj.clone(), proj.clone()]

# --------------------------------------------------
# Depth hypotheses
# --------------------------------------------------
depth_values = torch.linspace(
    0.5, 5.0, cfg.depth_num, device="cuda"
).unsqueeze(0)   # [1, D]

# --------------------------------------------------
# Forward pass
# --------------------------------------------------
with torch.no_grad():
    out = model(
        rgb,
        depth_in,
        proj_mats,
        depth_values
    )

print("Forward sanity passed ?")

if torch.is_tensor(out):
    print("Output shape:", out.shape)
elif isinstance(out, (list, tuple)):
    for i, o in enumerate(out):
        if torch.is_tensor(o):
            print(f"Output[{i}] shape:", o.shape)
