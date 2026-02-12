# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from .hybrid_encoders import HybridRGBEncoder, HybridDepthEncoder
from .fusion_block import MultiScaleFusion
from .cost_volume import CostVolumeConstructor
from .decoder import DepthDecoder


class HybridFusionFormer(nn.Module):
    """
    Fully fixed HybridFusionFormer with:
      - OOM-safe downsampling
      - Numeric stability safeguards
      - Proj_mats sanitization
      - Depth hypothesis clamping
      - NaN/Inf safety for cost volume and decoder
      - Returns dict compatible with loss function
    """

    def __init__(self, cfg,
                 embed_dim: int = 128,
                 base_channels: int = 32,
                 safe_max_tokens: int = 4096,
                 safe_input_hw: tuple = (384, 512),
                 nhead: int = 2,
                 skip_highres: bool = True,
                 debug: bool = False):
        super().__init__()

        # Config
        self.cfg = cfg
        self.cfg.depth_num = int(min(getattr(cfg, "depth_num", 48), 32))
        self.embed_dim = embed_dim
        self.base_channels = base_channels
        self.safe_max_tokens = safe_max_tokens
        self.safe_input_hw = safe_input_hw
        self.skip_highres = skip_highres
        self.debug = debug

        # Encoders
        self.rgb_encoder = HybridRGBEncoder(
            in_ch=3, base_channels=base_channels, embed_dim=embed_dim, nhead=nhead)
        self.depth_encoder = HybridDepthEncoder(
            in_ch=1, base_channels=base_channels, embed_dim=embed_dim, nhead=nhead)

        # Multi-scale fusion
        self.fusion = MultiScaleFusion(
            embed_dim=embed_dim,
            nhead=nhead,
            levels=4,
            encoder_channels=[embed_dim] * 4,
            debug=debug
        )

        # Cost volume + decoder
        self.cost_volume = CostVolumeConstructor(
            depth_num=self.cfg.depth_num,
            fusion_mode="variance"
        )
        self.decoder = DepthDecoder(in_channels=cfg.decoder_in_channels)

    # -------------------------------
    def _maybe_downsample_inputs(self, rgb: torch.Tensor, depth: torch.Tensor):
        """Downscale inputs if larger than safe_input_hw."""
        _, _, H, W = rgb.shape
        target_h, target_w = self.safe_input_hw
        if H > target_h or W > target_w:
            if self.debug:
                print(f"[HybridFusionFormer] Downsampling inputs {H}x{W} -> {target_h}x{target_w}")
            rgb = F.interpolate(rgb, size=(target_h, target_w), mode="bilinear", align_corners=False)
            depth = F.interpolate(depth, size=(target_h, target_w), mode="nearest")
            return rgb, depth, (target_h / H, target_w / W)
        return rgb, depth, (1.0, 1.0)

    # -------------------------------
    def _ensure_proj_mats_device(self, proj_mats, device):
        """Convert proj_mats to list of tensors on `device`."""
        if proj_mats is None:
            return None

        if isinstance(proj_mats, (list, tuple)):
            return [torch.as_tensor(p, device=device, dtype=torch.float32) if not isinstance(p, torch.Tensor) else p.to(device) for p in proj_mats]
        elif isinstance(proj_mats, torch.Tensor):
            if proj_mats.dim() == 4 and proj_mats.shape[1] > 1:
                return [proj_mats[:, i].to(device) for i in range(proj_mats.shape[1])]
            else:
                return [proj_mats.to(device)]
        else:
            return [torch.as_tensor(proj_mats, device=device, dtype=torch.float32)]

    # -------------------------------
    def _sanitize_and_normalize_proj_mats(self, proj_mats_list, device, batch_size):
        """Sanitize and normalize proj_mats: shape -> (B,4,4) and clamp homogeneous scale."""
        cleaned = []
        eps = 1e-8
        for i, p in enumerate(proj_mats_list):
            if p is None:
                cleaned.append(None)
                continue
            tp = p if isinstance(p, torch.Tensor) else torch.as_tensor(p, device=device, dtype=torch.float32)
            tp = tp.to(device)

            # squeeze singleton dims (B,1,4,4) -> (B,4,4)
            while tp.dim() > 3:
                if tp.shape[1] == 1:
                    tp = tp.squeeze(1)
                elif tp.shape[0] == 1:
                    tp = tp.squeeze(0)
                else:
                    break

            # Promote (4,4) -> (B,4,4)
            if tp.dim() == 2 and tp.shape == (4, 4):
                tp = tp.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
            elif tp.dim() == 3 and tp.shape[0] == 1 and batch_size > 1:
                tp = tp.expand(batch_size, -1, -1).contiguous()

            # Final check
            if tp.dim() != 3 or tp.shape[1:] != (4, 4):
                try:
                    tp = tp.reshape(batch_size, 4, 4)
                except Exception:
                    raise ValueError(f"proj_mats element {i} has unsupported shape {tp.shape}")

            # Normalize homogeneous scale
            denom = tp[..., -1, -1].clamp(min=eps)
            tp = tp / denom.unsqueeze(-1).unsqueeze(-1)

            # NaN/Inf safety
            if torch.isnan(tp).any() or torch.isinf(tp).any():
                if self.debug:
                    print(f"[HybridFusionFormer] Warning: proj_mats element {i} contains NaN/Inf; applying nan_to_num")
                tp = torch.nan_to_num(tp, nan=0.0, posinf=1e6, neginf=-1e6)
            cleaned.append(tp)
        return cleaned

    # -------------------------------
    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, proj_mats, depth_hypos):
        """
        Args:
            rgb: (B,3,H,W)
            depth: (B,1,H,W)
            proj_mats: list of projection matrices (ref first)
            depth_hypos: (B,D) or (D,)
        Returns:
            dict with stage1: {"depth", "prob_volume", "depth_hypo"}
        """

        # 1) Downsample inputs if needed
        rgb_safe, depth_safe, scale_factors = self._maybe_downsample_inputs(rgb, depth)
        depth_safe = torch.clamp(depth_safe, min=1e-4)

        # 2) Ensure proj_mats are on device and sanitized
        proj_mats = self._ensure_proj_mats_device(proj_mats, rgb_safe.device)
        proj_mats = self._sanitize_and_normalize_proj_mats(proj_mats, rgb_safe.device, batch_size=rgb_safe.shape[0])

        # 3) Encode
        amp_ctx = autocast if torch.cuda.is_available() else (lambda *a, **k: (lambda x: x))
        with amp_ctx():
            rgb_feats = self.rgb_encoder(rgb_safe)
            depth_feats = self.depth_encoder(depth_safe)

            if self.debug:
                for i, f in enumerate(rgb_feats):
                    print(f"[HybridFusionFormer] RGB feat[{i}] shape: {tuple(f.shape)}")
                for i, f in enumerate(depth_feats):
                    print(f"[HybridFusionFormer] Depth feat[{i}] shape: {tuple(f.shape)}")

            # 4) Fusion
            fused_feats = self.fusion(rgb_feats, depth_feats,
                                      max_tokens=self.safe_max_tokens,
                                      skip_highres=self.skip_highres)

            if self.debug:
                for i, f in enumerate(fused_feats):
                    print(f"[HybridFusionFormer] Fused feat[{i}] shape: {tuple(f.shape)}")

            # 5) Prepare depth hypotheses
            if depth_hypos is None:
                raise ValueError("depth_hypos must be provided")
            depth_hypos = torch.as_tensor(depth_hypos, device=rgb_safe.device, dtype=torch.float32)
            if depth_hypos.dim() == 1:
                depth_hypos = depth_hypos.unsqueeze(0)
            if depth_hypos.shape[-1] > self.cfg.depth_num:
                depth_hypos = depth_hypos[:, :self.cfg.depth_num]
            depth_hypos = torch.clamp(depth_hypos, min=1e-4)
	    # FIX-3: normalize depth hypotheses (GeoMVSNet style)
            depth_hypos = depth_hypos / depth_hypos.mean(dim=-1, keepdim=True)


            # 6) Build cost volume
            try:
                cost_volume = self.cost_volume(fused_feats=[fused_feats[0]],proj_mats=[proj_mats[0]], depth_hypos=depth_hypos)
            except Exception as e:
                if self.debug:
                    print("[HybridFusionFormer] Error building cost volume:", e)
                raise

            # NaN/Inf safety
            if torch.isnan(cost_volume).any() or torch.isinf(cost_volume).any():
                cost_volume = torch.nan_to_num(cost_volume, nan=0.0, posinf=1e6, neginf=-1e6)
                cost_volume = torch.clamp(cost_volume, min=-1e6, max=1e6)

            # 7) Decode depth
            fused_highres = fused_feats[0]
            try:
                depth_map, prob_volume = self.decoder(cost_volume, depth_hypos, fused_highres=fused_highres)
            except Exception as e:
                if self.debug:
                    print("[HybridFusionFormer] Error during decoding:", e)
                raise

        # 8) Upsample if downsampled
        if scale_factors != (1.0, 1.0):
            _, _, orig_H, orig_W = rgb.shape
            depth_map = F.interpolate(depth_map, size=(orig_H, orig_W), mode="bilinear", align_corners=False)

        return {
            "stage1": {
                "depth": depth_map,
                "prob_volume": prob_volume,
                "depth_hypo": depth_hypos
            }
        }


# -------------------------------
# Optional: lightweight config for testing
# -------------------------------
class Config:
    def __init__(self):
        self.depth_num = 48
        self.decoder_in_channels = 64


# -------------------------------
# Quick test
# -------------------------------
if __name__ == "__main__":
    """
    Sanity test that mimics the real DTU pipeline:
    Dataset ? Model ? Stage1 output
    """

    from datasets.geofusion_dataset_dtu import GeoFusionDatasetDTU
    from torch.utils.data import DataLoader

    class Config:
        depth_num = 48
        decoder_in_channels = 64

    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Dataset ----
    dataset = GeoFusionDatasetDTU(
        datapath="datasets/dtu_training/mvs_training/dtu",
        listfile="datasets/dtu_training/lists/train.txt",
        nviews=2,
        ndepths=cfg.depth_num,
        mode="train",
        use_input_depth=True,
        eval=False
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # ---- Model ----
    model = HybridFusionFormer(cfg, debug=True).to(device)
    model.eval()

    # ---- One forward pass ----
    sample = next(iter(loader))

    rgb = sample["ref_img"].unsqueeze(0).to(device)     # (1,3,H,W)
    depth = sample["depth"].to(device)                  # (1,1,H,W)
    proj_mats = sample["proj_mats"]                      # list of dicts
    depth_hypos = sample["depth_hypos"].to(device)      # (D,)

    # Convert dataset proj_mats dict ? tensor (c2w only)
    #proj_mats = [p["T_c2w"].unsqueeze(0).to(device) for p in proj_mats]
    proj_mats_fixed = []
    for p in proj_mats:
        K = p["K"].to(device)
        T_c2w = p["T_c2w"].to(device)
        T_w2c = torch.inverse(T_c2w)

        P = K @ T_w2c[:3, :]
        P4 = torch.eye(4, device=device)
        P4[:3, :] = P

        proj_mats_fixed.append(P4.unsqueeze(0))
    proj_mats = proj_mats_fixed


    with torch.no_grad():
        out = model(rgb, depth, proj_mats, depth_hypos)

    print("? Forward pass successful")
    print("Depth:", out["stage1"]["depth"].shape)
    print("Prob volume:", out["stage1"]["prob_volume"].shape)
