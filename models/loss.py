# -*- coding: utf-8 -*-
# @Description: Loss Functions (Sec 3.4 in the paper).
# @Author: Zhe Zhang (doublez@stu.pku.edu.cn)
# @LastEditDate: 2026-01-07
# @Integration: Fixed IndexError by forcing 3D shapes for Boolean indexing.

import torch
import torch.nn as nn
import torch.nn.functional as F

def geomvsnet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    """
    Complete Multi-stage loss integration.
    Calculates Pixel-wise Cross Entropy and Depth Distribution Similarity.
    """
    stage_lw = kwargs.get("stage_lw", [1, 1, 1, 1])
    depth_values = kwargs.get("depth_values")
    depth_min, depth_max = depth_values[:, 0], depth_values[:, -1]
    
    # Initialize total loss on the correct device
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=True)
    pw_loss_stages = []
    dds_loss_stages = []
    
    # Loop through all stages (e.g., stage1, stage2, stage3)
    relevant_stages = [(inputs[k], k) for k in inputs.keys() if "stage" in k]
    
    for stage_idx, (stage_inputs, stage_key) in enumerate(relevant_stages):
        # Extract stage-specific tensors
        depth = stage_inputs['depth']
        prob_volume = stage_inputs['prob_volume']
        depth_value = stage_inputs['depth_hypo']
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]

        # 1. Pixel-wise Cross Entropy Loss (Supervises Probability Volume)
        pw_loss = pixel_wise_loss(prob_volume, depth_gt, mask, depth_value)
        pw_loss_stages.append(pw_loss)

        # 2. Depth Distribution Similarity Loss (Supervises Regressed Depth)
        dds_loss = depth_distribution_similarity_loss(depth, depth_gt, mask, depth_min, depth_max)
        dds_loss_stages.append(dds_loss)
        
        # 3. Aggregate total loss with stage weights and lambda balancing
        # Based on GeoMVSNet: lam1 (PW) = 0.8, lam2 (DDS) = 0.2
        lam1, lam2 = 0.8, 0.2
        total_loss = total_loss + stage_lw[stage_idx] * (lam1 * pw_loss + lam2 * dds_loss)

    # Calculate metrics for the final stage prediction
    epe = cal_metrics(depth, depth_gt, mask, depth_min, depth_max)
    
    return total_loss, epe, pw_loss_stages, dds_loss_stages

def pixel_wise_loss(prob_volume, depth_gt, mask, depth_value):
    """
    Calculates Cross Entropy between the probability volume and the one-hot GT depth index.
    """
    b, d, h, w = prob_volume.shape
    device = prob_volume.device

    # Shape-safe Interpolation: Force to 4D for F.interpolate
    if depth_gt.dim() == 3: 
        depth_gt = depth_gt.unsqueeze(1)
    if mask.dim() == 3: 
        mask = mask.unsqueeze(1)

    depth_gt_s = F.interpolate(depth_gt, size=(h, w), mode='nearest').squeeze(1)
    mask_s = F.interpolate(mask.float(), size=(h, w), mode='nearest').squeeze(1) > 0.5
    
    valid_pixel_num = torch.sum(mask_s, dim=[1, 2]) + 1e-12

    # Prepare depth hypotheses
    if depth_value.dim() == 2:
        depth_value_mat = depth_value.view(b, d, 1, 1)
    else:
        depth_value_mat = F.interpolate(depth_value, size=(h, w), mode='nearest')

    # Find the hypothesis index closest to the Ground Truth per pixel
    gt_index_image = torch.argmin(torch.abs(depth_value_mat - depth_gt_s.unsqueeze(1)), dim=1)

    # Mask and prepare indices
    gt_index_image = torch.mul(mask_s.float(), gt_index_image.type(torch.float))
    indices = torch.round(gt_index_image).type(torch.long).reshape(b, 1, h, w)

    # Convert indices to one-hot volume
    gt_index_volume = torch.zeros(b, d, h, w, device=device).scatter_(1, indices, 1)
    
    # Negative Log Likelihood Calculation
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume + 1e-12), dim=1)
    masked_cross_entropy_image = torch.mul(mask_s.float(), cross_entropy_image)
    
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])
    return torch.mean(masked_cross_entropy / valid_pixel_num)

def depth_distribution_similarity_loss(depth, depth_gt, mask, depth_min, depth_max):
    """
    Measures similarity between predicted and GT depth using normalized scales.
    """
    # FORCE 3D [B, H, W] to avoid IndexError during Boolean indexing
    if depth.dim() == 4: 
        depth = depth.squeeze(1)
    if depth_gt.dim() == 4: 
        depth_gt = depth_gt.squeeze(1)
    if mask.dim() == 4: 
        mask = mask.squeeze(1)

    # Fix resolution mismatch
    if depth.shape[-2:] != depth_gt.shape[-2:]:
        depth_gt = F.interpolate(depth_gt.unsqueeze(1), size=depth.shape[-2:], mode='nearest').squeeze(1)
        mask = F.interpolate(mask.float().unsqueeze(1), size=depth.shape[-2:], mode='nearest').squeeze(1)

    mask = mask.bool()
    d_min = depth_min.view(-1, 1, 1)
    d_max = depth_max.view(-1, 1, 1)

    # Normalize depth values
    depth_norm = depth * 128 / (d_max - d_min + 1e-12)
    depth_gt_norm = depth_gt * 128 / (d_max - d_min + 1e-12)

    M_bins = 48
    kl_min = torch.min(depth_gt)
    kl_max = torch.max(depth_gt)
    bins = torch.linspace(kl_min, kl_max, steps=M_bins, device=depth.device)

    kl_divs = []
    for i in range(len(bins) - 1):
        bin_mask = (depth_gt >= bins[i]) & (depth_gt < bins[i+1])
        merged_mask = mask & bin_mask 

        if merged_mask.any():
            # Indexing safe: both depth_norm and merged_mask are now aligned
            p = depth_norm[merged_mask]
            q = depth_gt_norm[merged_mask]
            kl_div = torch.mean(torch.abs(p - q)) 
            kl_divs.append(kl_div)

    if len(kl_divs) == 0:
        return torch.tensor(0.0, device=depth.device, requires_grad=True)
        
    return sum(kl_divs) / len(kl_divs)

def cal_metrics(depth_pred, depth_gt, mask, depth_min, depth_max):
    """
    Calculates End-Point Error (EPE) on normalized scale.
    """
    # FORCE 3D [B, H, W]
    if depth_pred.dim() == 4: 
        depth_pred = depth_pred.squeeze(1)
    if depth_gt.dim() == 4: 
        depth_gt = depth_gt.squeeze(1)
    if mask.dim() == 4: 
        mask = mask.squeeze(1)

    if depth_pred.shape[-2:] != depth_gt.shape[-2:]:
        depth_gt = F.interpolate(depth_gt.unsqueeze(1), size=depth_pred.shape[-2:], mode='nearest').squeeze(1)
        mask = F.interpolate(mask.float().unsqueeze(1), size=depth_pred.shape[-2:], mode='nearest').squeeze(1)

    mask = mask.bool()
    d_min = depth_min.view(-1, 1, 1)
    d_max = depth_max.view(-1, 1, 1)

    depth_pred_norm = depth_pred * 128 / (d_max - d_min + 1e-12)
    depth_gt_norm = depth_gt * 128 / (d_max - d_min + 1e-12)

    if not mask.any():
        return torch.tensor(0.0, device=depth_pred.device)

    abs_err = torch.abs(depth_pred_norm[mask] - depth_gt_norm[mask])
    return abs_err.mean()