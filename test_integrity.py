import torch

# Import your model
from models.hybridfusionformer import HybridFusionFormer

# Minimal config simulation
class Config:
    def __init__(self):
        self.depth_num = 16
        self.decoder_in_channels = 128

cfg = Config()
model = HybridFusionFormer(cfg)

print("? HybridFusionFormer initialized successfully!")

# Mock inputs
rgb = torch.randn(1, 3, 128, 160)
depth = torch.randn(1, 1, 128, 160)
proj_mats = [torch.eye(4).unsqueeze(0), torch.eye(4).unsqueeze(0)]
depth_hypos = torch.linspace(0.5, 10.0, steps=cfg.depth_num).unsqueeze(0)

try:
    # Forward pass (will raise if any missing argument mismatch)
    depth_map = model(rgb, depth, proj_mats, depth_hypos)
    if isinstance(depth_map, tuple):
        depth_map = depth_map[0]
    print("? Forward pass completed!")
    print("Output depth shape:", depth_map.shape)
except Exception as e:
    print("?? Runtime error:", e)
