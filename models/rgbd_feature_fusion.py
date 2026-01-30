import torch
import torch.nn as nn
from models.submodules import FPN

class RGBDFeatureNet(nn.Module):
    def __init__(self, base_channels=8):
        super().__init__()
        self.rgb_encoder = FPN(base_channels=base_channels, in_channels=3)
        self.depth_encoder = FPN(base_channels=base_channels, in_channels=1)

        self.fusion_blocks = nn.ModuleDict({
            "stage1": self._fusion_block(base_channels * 8),
            "stage2": self._fusion_block(base_channels * 4),
            "stage3": self._fusion_block(base_channels * 2),
            "stage4": self._fusion_block(base_channels),
        })

    def _fusion_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb, depth):
        rgb_feats = self.rgb_encoder(rgb)       # Dict of features per stage
        depth_feats = self.depth_encoder(depth)

        fused_feats = {}
        for stage in rgb_feats:
	    # Ensure both feature maps have same spatial size before fusion
            if rgb_feats[stage].shape[2:] != depth_feats[stage].shape[2:]:
                depth_feats[stage] = torch.nn.functional.interpolate(
                depth_feats[stage],
                size=rgb_feats[stage].shape[2:],  # match H, W of RGB features
                mode='bilinear',
                align_corners=False
            )
            fused_input = torch.cat([rgb_feats[stage], depth_feats[stage]], dim=1)
            fused_feats[stage] = self.fusion_blocks[stage](fused_input)
        return fused_feats
