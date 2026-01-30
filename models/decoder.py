# models/decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# 3D Cost Volume Regularizer
# ----------------------------
class CostRegressor3D(nn.Module):
    """
    Simple 3D convolutional regularization stack.
    Input: cost_volume (B, C, D, H, W)
    Output: aggregated cost (B, D, H, W)
    """
    def __init__(self, in_channels, hidden_channels=64, num_layers=3, debug=False):
        super().__init__()
        layers = []
        cur_ch = in_channels
        for i in range(num_layers):
            out_ch = hidden_channels if i < num_layers - 1 else hidden_channels
            layers.append(nn.Conv3d(cur_ch, out_ch, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm3d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            cur_ch = out_ch
        # final projection to single-channel per-hypothesis
        layers.append(nn.Conv3d(cur_ch, 1, kernel_size=3, padding=1, bias=False))
        self.net = nn.Sequential(*layers)
        self.debug = debug

    def forward(self, cost_volume):
        """
        cost_volume: (B, C, D, H, W)
        returns: aggregated_cost: (B, D, H, W)
        """
        out = self.net(cost_volume)               # (B, 1, D, H, W)
        out = out.squeeze(1)                      # (B, D, H, W)

        # Guard numeric issues: replace NaN/Inf, clamp extreme values
        if torch.isnan(out).any() or torch.isinf(out).any():
            if self.debug:
                print("[CostRegressor3D] Warning: aggregated cost contains NaN/Inf; applying nan_to_num.")
            out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
            out = torch.clamp(out, min=-1e6, max=1e6)

        return out


# ----------------------------
# Soft-argmin regression
# ----------------------------
def soft_argmin(cost_volume, depth_values):
    """
    cost_volume: (B, D, H, W)  -- lower cost = better match
    depth_values: (B, D) or (D,)  depth hypothesis values
    Returns:
        depth_map: (B, 1, H, W)
        prob_volume: (B, D, H, W)
    """
    # Convert cost -> similarity (higher = better) by negating cost
    # score: (B, D, H, W)
    score = -cost_volume

    B, D, H, W = score.shape

    # reshape for softmax over depth dimension
    score_flat = score.view(B, D, -1)  # (B, D, H*W)

    # Numeric stability: subtract per-column max before softmax (broadcastable)
    # max over depth dim (dim=1)
    max_vals = torch.max(score_flat, dim=1, keepdim=True)[0]  # (B,1,H*W)
    score_flat_stable = score_flat - max_vals

    # Guard against NaN/Inf in score
    if torch.isnan(score_flat_stable).any() or torch.isinf(score_flat_stable).any():
        score_flat_stable = torch.nan_to_num(score_flat_stable, nan=-1e6, posinf=1e6, neginf=-1e6)

    # compute softmax
    prob_flat = F.softmax(score_flat_stable, dim=1)  # softmax over D -> (B, D, H*W)

    # reshape back
    prob = prob_flat.view(B, D, H, W)  # (B, D, H, W)

    # Depth values to broadcastable shape
    if isinstance(depth_values, torch.Tensor):
        if depth_values.dim() == 1:
            depth_vals = depth_values.view(1, D, 1, 1)   # (1, D, 1, 1)
        else:
            depth_vals = depth_values.view(B, D, 1, 1)   # (B, D, 1, 1)
    else:
        # convert to tensor and clamp
        depth_values = torch.as_tensor(depth_values, dtype=score.dtype, device=score.device)
        if depth_values.dim() == 1:
            depth_vals = depth_values.view(1, D, 1, 1)
        else:
            depth_vals = depth_values.view(B, D, 1, 1)

    # Ensure depth_vals is clamped to avoid zeros/negatives
    depth_vals = torch.clamp(depth_vals, min=1e-4)

    # Weighted sum
    depth_map = torch.sum(prob * depth_vals, dim=1, keepdim=True)  # (B, 1, H, W)

    # Final numeric guards
    if torch.isnan(prob).any() or torch.isinf(prob).any():
        prob = torch.nan_to_num(prob, nan=0.0, posinf=1e6, neginf=-1e6)
        # renormalize probabilities to sum to 1 along D
        prob_sum = prob.sum(dim=1, keepdim=True)  # (B,1,H,W)
        prob_sum = torch.clamp(prob_sum, min=1e-8)
        prob = prob / prob_sum

    if torch.isnan(depth_map).any() or torch.isinf(depth_map).any():
        depth_map = torch.nan_to_num(depth_map, nan=0.0, posinf=1e6, neginf=-1e6)
        depth_map = torch.clamp(depth_map, min=1e-4, max=1e6)

    return depth_map, prob


# ----------------------------
# 2D Depth Refinement Block
# ----------------------------
class DepthRefinementBlock(nn.Module):
    """
    Refine coarse depth map using high-resolution fused features (e.g., o1 from encoders).
    Input:
      - coarse_depth: (B,1,H,W)
      - fused_highres: (B,C,H,W)
    Output:
      - refined_depth: (B,1,H,W)
    """
    def __init__(self, feat_channels=128, hidden=64):
        super().__init__()
        # combine depth and features
        self.conv1 = nn.Sequential(
            nn.Conv2d(feat_channels + 1, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True)
        )
        self.conv_out = nn.Conv2d(hidden, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, coarse_depth, fused_highres):
        # ensure shapes match
        if coarse_depth.shape[-2:] != fused_highres.shape[-2:]:
            # upsample coarse depth to match feature resolution
            coarse_depth = F.interpolate(coarse_depth, size=fused_highres.shape[-2:], mode="bilinear", align_corners=False)

        # Concatenate and refine
        x = torch.cat([coarse_depth, fused_highres], dim=1)  # (B, C+1, H, W)
        x = self.conv1(x)
        x = self.conv2(x)
        res = self.conv_out(x)
        refined = coarse_depth + res

        # Guard numeric issues
        if torch.isnan(refined).any() or torch.isinf(refined).any():
            # Use a safe scalar fallback derived from coarse_depth mean (safe for multi-element tensors)
            try:
                fallback = float(coarse_depth.mean().detach().cpu().item())
            except Exception:
                fallback = 0.0
            refined = torch.nan_to_num(refined, nan=fallback, posinf=1e6, neginf=-1e6)
            refined = torch.clamp(refined, min=1e-4, max=1e6)

        return refined


# ----------------------------
# Depth Decoder (Full)
# ----------------------------
class DepthDecoder(nn.Module):
    """
    DepthDecoder:
      - cost_volume: (B, C, D, H, W) -> 3D reg -> (B, D, H, W)
      - soft-argmin -> coarse depth (B,1,H,W)
      - optional refinement using fused_feat_highres (B, C, H, W)
    """
    def __init__(self, in_channels=128, reg_hidden=64, depth_num=48, refine=True, debug=False):
        super().__init__()
        # in_channels: channels of cost_volume (C)
        self.regressor = CostRegressor3D(in_channels, hidden_channels=reg_hidden, num_layers=3, debug=debug)
        self.refine = refine
        if refine:
            # use feature channels same as unified embed_dim from encoders
            self.refinement = DepthRefinementBlock(feat_channels=in_channels, hidden=reg_hidden)
        self.debug = debug

    def forward(self, cost_volume, depth_values, fused_highres=None):
        """
        Args:
            cost_volume: (B, C, D, H, W)   - output from CostVolumeConstructor
            depth_values: (B, D) or (D,)   - depth hypotheses
            fused_highres: (B, C, H, W)    - optional high-res fused feature for refinement (level 1)
        Returns:
            depth_map: (B, 1, H, W)
            prob_volume: (B, D, H, W)
        """
        # Basic shape checks
        if cost_volume.dim() != 5:
            raise ValueError("cost_volume must be (B, C, D, H, W)")

        # 3D regularization -> aggregated cost (B, D, H, W)
        aggregated_cost = self.regressor(cost_volume)

        # Guard aggregated_cost
        if torch.isnan(aggregated_cost).any() or torch.isinf(aggregated_cost).any():
            if self.debug:
                print("[DepthDecoder] Warning: aggregated_cost contains NaN/Inf; applying nan_to_num.")
            aggregated_cost = torch.nan_to_num(aggregated_cost, nan=0.0, posinf=1e6, neginf=-1e6)
            aggregated_cost = torch.clamp(aggregated_cost, min=-1e6, max=1e6)

        # soft-argmin regression -> coarse depth + prob
        depth_map, prob = soft_argmin(aggregated_cost, depth_values)

        # optional refinement
        if self.refine and fused_highres is not None:
            # ensure feature channel sizes are consistent: if not, project fused_highres to expected channels
            if fused_highres.shape[1] != cost_volume.shape[1]:
                # attempt simple 1x1 conv projection on the fly
                if self.debug:
                    print("[DepthDecoder] Info: projecting fused_highres channels to match cost_volume channels for refinement.")
                proj = nn.Conv2d(fused_highres.shape[1], cost_volume.shape[1], kernel_size=1, bias=False).to(fused_highres.device)
                fused_highres = proj(fused_highres)

            depth_map = self.refinement(depth_map, fused_highres)

        # final guards
        if torch.isnan(depth_map).any() or torch.isinf(depth_map).any():
            if self.debug:
                print("[DepthDecoder] Warning: final depth_map contains NaN/Inf; applying nan_to_num.")
            depth_map = torch.nan_to_num(depth_map, nan=0.0, posinf=1e6, neginf=-1e6)
            depth_map = torch.clamp(depth_map, min=1e-4, max=1e6)

        if torch.isnan(prob).any() or torch.isinf(prob).any():
            if self.debug:
                print("[DepthDecoder] Warning: final prob contains NaN/Inf; applying nan_to_num and renormalizing.")
            prob = torch.nan_to_num(prob, nan=0.0, posinf=1e6, neginf=-1e6)
            prob_sum = prob.sum(dim=1, keepdim=True).clamp(min=1e-8)
            prob = prob / prob_sum

        return depth_map, prob


# ----------------------------
# Quick local test
# ----------------------------
if __name__ == "__main__":
    # small sanity check
    B, C, D, H, W = 1, 128, 16, 64, 80
    cost = torch.randn(B, C, D, H, W)
    depth_vals = torch.linspace(0.5, 10.0, D)  # (D,)
    fused_highres = torch.randn(B, C, H, W)

    decoder = DepthDecoder(in_channels=C, reg_hidden=64, depth_num=D, refine=True, debug=True)
    depth_map, prob = decoder(cost, depth_vals, fused_highres=fused_highres)
    print("Depth map shape:", depth_map.shape)
    print("Prob shape:", prob.shape)
