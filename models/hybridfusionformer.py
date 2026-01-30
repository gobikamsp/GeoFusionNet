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
    OOM-optimized HybridFusionFormer with numeric-stability safeguards.

    Changes from original:
      - clamp depth inputs and depth hypothesis to avoid zero divisions
      - sanitize proj_mats shapes (squeeze stray dims, promote 4x4 -> (B,4,4))
      - normalize proj matrices by bottom-right element with clamp
      - nan/inf checks after cost volume and before decoder; replace with safe values
      - Multi-Stage Output: returns (depth_map, prob_volume) for Entropy/Normal losses
      - preserve debug printing
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

        # store config and safety params
        self.cfg = cfg
        # cap depth hypotheses to save memory during cost-volume construction
        self.cfg.depth_num = int(min(getattr(cfg, "depth_num", 48), 32))

        self.embed_dim = embed_dim
        self.base_channels = base_channels
        self.safe_max_tokens = safe_max_tokens
        self.safe_input_hw = safe_input_hw  # (H, W)
        self.skip_highres = skip_highres
        self.debug = debug

        # ----- Encoders (keep nhead small to save memory) -----
        self.rgb_encoder = HybridRGBEncoder(
            in_ch=3, base_channels=base_channels, embed_dim=embed_dim, nhead=nhead)
        self.depth_encoder = HybridDepthEncoder(
            in_ch=1, base_channels=base_channels, embed_dim=embed_dim, nhead=nhead)

        # ----- Fusion: adapter channels must match encoder unified output (embed_dim) -----
        self.fusion = MultiScaleFusion(
            embed_dim=embed_dim,
            nhead=nhead,
            levels=4,
            encoder_channels=[embed_dim] * 4,  # encoder unify layers must output embed_dim channels
            debug=debug
        )

        # ----- Cost volume and decoder -----
        # CostVolumeConstructor expects proj_mats as list-like: [ref_proj, src_proj1, ...]
        self.cost_volume = CostVolumeConstructor(depth_num=self.cfg.depth_num, fusion_mode="variance")
        self.decoder = DepthDecoder(in_channels=cfg.decoder_in_channels)

    def _maybe_downsample_inputs(self, rgb: torch.Tensor, depth: torch.Tensor):
        """Downscale inputs to safe_input_hw if larger. Returns rgb, depth, scale_factors."""
        _, _, H, W = rgb.shape
        target_h, target_w = self.safe_input_hw
        if H > target_h or W > target_w:
            if self.debug:
                print(f"[HybridFusionFormer] Downsampling inputs {H}x{W} -> {target_h}x{target_w}")
            rgb = F.interpolate(rgb, size=(target_h, target_w), mode="bilinear", align_corners=False)
            depth = F.interpolate(depth, size=(target_h, target_w), mode="nearest")
            return rgb, depth, (target_h / H, target_w / W)
        return rgb, depth, (1.0, 1.0)

    def _ensure_proj_mats_device(self, proj_mats, device):
        """
        Ensure proj_mats are tensors on `device`.
        Accepts:
          - list/tuple of tensors
          - single tensor shaped (B, N, 4, 4) or (B, 4, 4) / (B, 3, 4)
        Returns list of per-view projection tensors (ref first).
        """
        if proj_mats is None:
            return None

        if isinstance(proj_mats, (list, tuple)):
            out = []
            for p in proj_mats:
                if isinstance(p, torch.Tensor):
                    out.append(p.to(device))
                else:
                    out.append(torch.tensor(p, device=device) if not p is None else None)
            return out
        elif isinstance(proj_mats, torch.Tensor):
            # If shape is (B, nview, 4, 4), split into list: first is ref, then sources
            if proj_mats.dim() == 4 and proj_mats.shape[1] > 1:
                # (B, Nviews, 4, 4) -> list of (B, 4, 4)
                pcs = [proj_mats[:, i].to(device) for i in range(proj_mats.shape[1])]
                return pcs
            else:
                return [proj_mats.to(device)]
        else:
            # try converting to tensor then to device
            try:
                t = torch.as_tensor(proj_mats, device=device)
                if t.dim() == 4 and t.shape[1] > 1:
                    return [t[:, i] for i in range(t.shape[1])]
                return [t]
            except Exception:
                raise ValueError("proj_mats must be list/tuple of tensors or a tensor")

    def _sanitize_and_normalize_proj_mats(self, proj_mats_list, device, batch_size):
        """
        Make proj_mats_list a clean list of tensors shaped (B,4,4).
        Steps:
          - convert to tensor on device
          - squeeze singleton dims like (B,1,4,4) -> (B,4,4)
          - if a single (4,4) is provided, promote to (B,4,4)
          - normalize by bottom-right element with clamp to avoid very small denominators
        Returns: cleaned list of (B,4,4) tensors (ref first)
        """
        cleaned = []
        eps = 1e-8
        for i, p in enumerate(proj_mats_list):
            if p is None:
                cleaned.append(None)
                continue
            tp = p
            # ensure tensor and device
            if not isinstance(tp, torch.Tensor):
                tp = torch.as_tensor(tp, device=device, dtype=torch.float32)
            else:
                tp = tp.to(device)

            # Squeeze singleton dims until we reach (B,4,4) or (4,4)
            # but avoid accidentally squeezing batch dim if batch>1
            while tp.dim() > 3:
                # prefer to squeeze singleton middle dims (e.g., (B,1,4,4) -> (B,4,4))
                if tp.shape[1] == 1:
                    tp = tp.squeeze(1)
                elif tp.shape[0] == 1:
                    tp = tp.squeeze(0)
                else:
                    break

            # If shape is (4,4) -> promote to (B,4,4)
            if tp.dim() == 2 and tp.shape == (4, 4):
                tp = tp.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

            # If shape is (B,4,4) and B==1 but actual batch_size>1, expand
            if tp.dim() == 3 and tp.shape[0] == 1 and batch_size > 1:
                tp = tp.expand(batch_size, -1, -1).contiguous()

            # Final check
            if tp.dim() != 3 or tp.shape[1:] != (4, 4):
                # As a last resort, attempt reshape if possible
                try:
                    tp = tp.reshape(batch_size, 4, 4)
                except Exception:
                    raise ValueError(f"proj_mats element {i} has unsupported shape after sanitization: {tuple(tp.shape)}")

            # Normalize homogeneous scale (bottom-right element)
            denom = tp[..., -1, -1].clamp(min=eps)
            tp = tp / denom.unsqueeze(-1).unsqueeze(-1)

            # Final NaN/Inf guard
            if torch.isnan(tp).any() or torch.isinf(tp).any():
                if self.debug:
                    print(f"[HybridFusionFormer] Warning: proj_mats element {i} contains NaN/Inf after sanitization; replacing via nan_to_num.")
                tp = torch.nan_to_num(tp, nan=0.0, posinf=1e6, neginf=-1e6)

            cleaned.append(tp)
        return cleaned

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, proj_mats, depth_hypos):
        """
        Args:
            rgb: (B,3,H,W)
            depth: (B,1,H,W)
            proj_mats: list/structure required by CostVolumeConstructor (ref first)
            depth_hypos: (B, D) or (D,) depth hypothesis values (will respect cfg.depth_num)
        Returns:
            depth_map: (B,1,H_out,W_out) predicted depth
            prob_volume: (B,D,H/4,W/4) probability volume for uncertainty/entropy loss
        """
        # 1) safety preprocessing: downsample if input is very large
        rgb_safe, depth_safe, scale_factors = self._maybe_downsample_inputs(rgb, depth)

        # clamp input depths to avoid zeros (which cause division by zero in projections)
        depth_safe = torch.clamp(depth_safe, min=1e-4)

        # 1.b) ensure proj_mats are on the same device as rgb_safe and sanitized
        proj_mats = self._ensure_proj_mats_device(proj_mats, rgb_safe.device)
        proj_mats = self._sanitize_and_normalize_proj_mats(proj_mats, rgb_safe.device, batch_size=rgb_safe.shape[0])

        # 2) Encode (use original encoders)
        # Note: encoders are expected to return lists of 4 feature maps (o1..o4)
        # Use autocast only if CUDA is available
        amp_ctx = autocast if torch.cuda.is_available() else (lambda *a, **k: (lambda x: x))
        with amp_ctx():
            rgb_feats = self.rgb_encoder(rgb_safe)
            depth_feats = self.depth_encoder(depth_safe)

            if self.debug:
                for i, f in enumerate(rgb_feats):
                    print(f"[HybridFusionFormer] RGB feat[{i}] shape: {tuple(f.shape)}")
                for i, f in enumerate(depth_feats):
                    print(f"[HybridFusionFormer] Depth feat[{i}] shape: {tuple(f.shape)}")

            # 3) Multi-scale fusion (OOM-safe parameters)
            fused_feats = self.fusion(
                rgb_feats,
                depth_feats,
                max_tokens=self.safe_max_tokens,
                skip_highres=self.skip_highres
            )

            if self.debug:
                for i, f in enumerate(fused_feats):
                    print(f"[HybridFusionFormer] Fused feat[{i}] shape: {tuple(f.shape)}")

            # 4) Prepare depth_hypos: ensure (B, D) and cap to cfg.depth_num
            if depth_hypos is None:
                raise ValueError("depth_hypos must be provided")
            if isinstance(depth_hypos, torch.Tensor):
                if depth_hypos.dim() == 1:
                    depth_hypos = depth_hypos.unsqueeze(0)
                if depth_hypos.shape[-1] > self.cfg.depth_num:
                    depth_hypos = depth_hypos[:, :self.cfg.depth_num]
            else:
                # allow list/ndarray input
                depth_hypos = torch.as_tensor(depth_hypos, device=rgb_safe.device, dtype=torch.float32)
                if depth_hypos.dim() == 1:
                    depth_hypos = depth_hypos.unsqueeze(0)
                if depth_hypos.shape[-1] > self.cfg.depth_num:
                    depth_hypos = depth_hypos[:, :self.cfg.depth_num]

            # clamp depth hypotheses to avoid zeros / negatives
            depth_hypos = torch.clamp(depth_hypos, min=1e-4)

            # 5) Build cost volume (cost_volume expects proj_mats list with ref first)
            # Add try/except and NaN-checks around cost volume construction since many ops here use projection
            try:
                cost_volume = self.cost_volume(fused_feats, proj_mats, depth_hypos)
            except Exception as e:
                # Provide helpful debug info and re-raise for developer
                if self.debug:
                    print("[HybridFusionFormer] Error while building cost volume:", e)
                    print("proj_mats sanitized shapes:", [p.shape if p is not None else None for p in proj_mats])
                    print("depth_hypos:", depth_hypos.shape, "min/max:", depth_hypos.min().item(), depth_hypos.max().item())
                raise

            # If cost_volume contains NaNs/Infs, try to salvage it safely
            if torch.isnan(cost_volume).any() or torch.isinf(cost_volume).any():
                if self.debug:
                    print("[HybridFusionFormer] Warning: cost_volume contains NaN/Inf. Applying torch.nan_to_num and clamping.")
                    # print some stats
                    try:
                        print(" cost_volume min/max before:", torch.nanmin(cost_volume).item(), torch.nanmax(cost_volume).item())
                    except Exception:
                        pass
                cost_volume = torch.nan_to_num(cost_volume, nan=0.0, posinf=1e6, neginf=-1e6)
                # small clamp to reasonable range to avoid huge values exploding in decoder
                cost_volume = torch.clamp(cost_volume, min=-1e6, max=1e6)

            # 6) Decode depth
            fused_highres = fused_feats[0]
            try:
                # Decoder returns (depth_prediction, probability_volume)
                depth_map, prob_volume = self.decoder(cost_volume, depth_hypos, fused_highres=fused_highres)
            except Exception as e:
                if self.debug:
                    print("[HybridFusionFormer] Error during depth decoding:", e)
                raise

        # 7) If we downsampled inputs, upsample prediction back to original input size
        if scale_factors != (1.0, 1.0):
            # original size
            _, _, orig_H, orig_W = rgb.shape
            depth_map = F.interpolate(depth_map, size=(orig_H, orig_W), mode="bilinear", align_corners=False)

       
       #return depth_map, prob_volume
    # Wrap in a dictionary to "talk" to your loss function correctly
            return {
                "stage1": {
                    "depth": depth_map,
                    "prob_volume": prob_volume,
                    "depth_hypo": depth_hypos
                    } }   


# ---------------------------------------------------------------
# Optional: lightweight config class for standalone testing
# ---------------------------------------------------------------
class Config:
    def __init__(self):
        self.depth_num = 48
        self.decoder_in_channels = 64


if __name__ == "__main__":
    # Quick structural test (use smaller image during dev if needed)
    import torch
    cfg = Config()
    model = HybridFusionFormer(cfg, debug=True).cuda()
    model.eval()

    # simulate larger input (model will downsample to safe size)
    rgb = torch.randn(1, 3, 576, 768).cuda()
    depth = torch.randn(1, 1, 576, 768).cuda()
    # create sane proj_mats: ref and one source (each (4,4))
    ref = torch.eye(4).cuda()
    src = torch.eye(4).cuda() * 1.0
    proj_mats = [ref, src]
    depth_hypos = torch.linspace(0.5, 10.0, steps=cfg.depth_num).cuda()

    try:
        out_depth, out_prob = model(rgb, depth, proj_mats, depth_hypos)
        print("Output depth shape:", out_depth.shape)
        print("Output prob shape:", out_prob.shape)
    except Exception as e:
        print("Model run failed:", e)
