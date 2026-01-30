# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp

def _ensure_4x4(proj, device=None, batch_size=1):
    """Standardizes projection matrices to (B, 4, 4) tensors."""
    if not isinstance(proj, torch.Tensor):
        proj = torch.as_tensor(proj)
    if device is not None:
        proj = proj.to(device)
    if proj.dim() == 2:
        proj = proj.unsqueeze(0)
    if proj.shape[0] == 1 and batch_size > 1:
        proj = proj.expand(batch_size, -1, -1).contiguous()
    if proj.shape[-2:] == (3, 4):
        pad = proj.new_tensor([0.0, 0.0, 0.0, 1.0]).view(1, 1, 4).expand(proj.shape[0], -1, -1)
        proj = torch.cat([proj, pad], dim=1)
    return proj

def _safe_inverse(mat):
    """Inverts matrices with pseudo-inverse fallback for numerical stability."""
    try:
        inv = torch.inverse(mat)
        if torch.isnan(inv).any() or torch.isinf(inv).any():
            inv = torch.linalg.pinv(mat)
    except Exception:
        inv = torch.linalg.pinv(mat)
    return torch.nan_to_num(inv, nan=0.0)

def homography_warp(src_feat, src_proj, ref_proj, depth_values, depth_chunk=8):
    """
    Warps source features to reference view. 
    Uses dynamic H, W detection to ensure the grid matches the input feature resolution.
    """
    device = src_feat.device
    B, C, H, W = src_feat.shape
    dtype = src_feat.dtype

    if depth_values.dim() == 1:
        depth_values = depth_values.unsqueeze(0)
    depth_values = depth_values.to(device)
    k = depth_values.shape[-1]

    srcP = _ensure_4x4(src_proj, device, B)
    refP = _ensure_4x4(ref_proj, device, B)
    
    # Relative transformation: ref -> src
    relP = srcP @ _safe_inverse(refP)
    R = relP[:, :3, :3].to(dtype)
    t = relP[:, :3, 3:4].to(dtype)

    # Generate pixel grid matching the current feature resolution (H, W)
    y, x = torch.meshgrid(
        torch.arange(0, H, device=device, dtype=torch.float32),
        torch.arange(0, W, device=device, dtype=torch.float32),
        indexing="ij"
    )
    # xyz: (3, H*W)
    xyz = torch.stack([x, y, torch.ones_like(x)], dim=0).view(3, -1)

    warped = torch.zeros((B, C, k, H, W), device=device, dtype=dtype)

    # Process in depth chunks to prevent OOM
    for i in range(0, k, depth_chunk):
        curr_k = min(depth_chunk, k - i)
        d = depth_values[:, i:i+curr_k].view(B, curr_k, 1).to(dtype)

        for b in range(B):
            # Back-project reference pixels to 3D space at current depth
            scaled_xyz = xyz.unsqueeze(1) * d[b].unsqueeze(0) 
            # Project 3D points into source camera coordinates
            proj_xyz = R[b] @ scaled_xyz.view(3, -1) + t[b]
            proj_xyz = proj_xyz.view(3, curr_k, H, W)

            # Convert to normalized coordinates [-1, 1] for grid_sample
            X = proj_xyz[0] / (proj_xyz[2] + 1e-6)
            Y = proj_xyz[1] / (proj_xyz[2] + 1e-6)
            
            norm_grid = torch.stack([
                (X / (W - 1) - 0.5) * 2.0,
                (Y / (H - 1) - 0.5) * 2.0
            ], dim=-1)

            feat_b = src_feat[b:b+1].repeat(curr_k, 1, 1, 1)
            
            with amp.autocast('cuda', enabled=True):
                sampled = F.grid_sample(
                    feat_b, norm_grid, mode='bilinear', 
                    padding_mode='zeros', align_corners=True
                )
            
            warped[b, :, i:i+curr_k] = sampled.permute(1, 0, 2, 3)

    return torch.nan_to_num(warped, nan=0.0)



class CostVolumeConstructor(nn.Module):
    """
    Constructs a cost volume by aggregating warped source features.
    Ensures all input features are spatially aligned to the reference scale.
    """
    def __init__(self, depth_num=32, fusion_mode="variance", debug=False):
        super().__init__()
        self.depth_num = depth_num
        self.fusion_mode = fusion_mode
        self.debug = debug

    def forward(self, fused_feats, proj_mats, depth_hypos):
        # 1. Use the first feature in the list to define the reference resolution
        ref_feat = fused_feats[0]
        B, C, H, W = ref_feat.shape
        device = ref_feat.device
        dtype = ref_feat.dtype

        # 2. Sanitize and clip depth hypotheses
        if isinstance(depth_hypos, torch.Tensor):
            depth_hypos = depth_hypos.to(device=device)
        else:
            depth_hypos = torch.as_tensor(depth_hypos, device=device, dtype=torch.float32)
        
        if depth_hypos.dim() == 1:
            depth_hypos = depth_hypos.unsqueeze(0)
            
        D = min(self.depth_num, depth_hypos.shape[-1])
        depth_hypos = depth_hypos[:, :D]

        # 3. Initialize accumulation buffers at reference resolution (H, W)
        sum_vol = torch.zeros((B, C, D, H, W), device=device, dtype=dtype)
        sq_sum_vol = torch.zeros((B, C, D, H, W), device=device, dtype=dtype)

        # 4. Reference view contribution
        ref_vol = ref_feat.unsqueeze(2).expand(-1, -1, D, -1, -1)
        sum_vol += ref_vol
        sq_sum_vol += ref_vol ** 2

        # 5. Loop through source views
        ref_proj = proj_mats[0]
        
        for i in range(1, len(proj_mats)):
            # Pick the corresponding feature map or fallback to the first one
            src_feat = fused_feats[i] if i < len(fused_feats) else fused_feats[0]
            
            # THE FIX: If source feature resolution doesn't match the reference, resize it.
            # This prevents "RuntimeError: size of tensor a must match size of tensor b"
            if src_feat.shape[-2:] != (H, W):
                if self.debug:
                    print(f"[CostVolume] Resizing src_feat {i} from {src_feat.shape[-2:]} to {(H, W)}")
                src_feat = F.interpolate(src_feat, size=(H, W), mode='bilinear', align_corners=True)
            
            # Warp current source feature into reference coordinates
            warped = homography_warp(src_feat, proj_mats[i], ref_proj, depth_hypos)
            
            # Accumulate into buffers
            sum_vol += warped
            sq_sum_vol += warped ** 2
            del warped

        # 6. Calculate variance across views
        num_views = len(proj_mats)
        # Var(X) = E[X^2] - (E[X])^2
        cost_vol = (sq_sum_vol / num_views) - (sum_vol / num_views) ** 2
        
        # Numeric guard: variance cannot be negative
        cost_vol = cost_vol.clamp(min=0.0)

        if self.fusion_mode == "concat":
            ref_exp = ref_feat.unsqueeze(2).expand(-1, -1, D, -1, -1)
            cost_vol = torch.cat([cost_vol, ref_exp], dim=1)
            
        return torch.nan_to_num(cost_vol, nan=0.0)