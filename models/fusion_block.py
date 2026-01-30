# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    """
    Bidirectional cross-attention block using batch_first=True and robust unpacking.
    Expects inputs (B, C, H, W) with C == embed_dim.
    """
    def __init__(self, embed_dim=128, nhead=2, dropout=0.1, debug=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.debug = debug

        # per-direction q/k/v
        self.query_rgb = nn.Linear(embed_dim, embed_dim)
        self.key_depth = nn.Linear(embed_dim, embed_dim)
        self.value_depth = nn.Linear(embed_dim, embed_dim)

        self.query_depth = nn.Linear(embed_dim, embed_dim)
        self.key_rgb = nn.Linear(embed_dim, embed_dim)
        self.value_rgb = nn.Linear(embed_dim, embed_dim)

        # MultiheadAttention expects (B, N, C) when batch_first=True
        self.attn_rgb = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.attn_depth = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)

        self.proj = nn.Linear(embed_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        assert embed_dim % nhead == 0, "embed_dim must be divisible by nhead"

    def forward(self, feat_rgb, feat_depth):
        B, C, H, W = feat_rgb.shape
        if C != self.embed_dim:
            raise RuntimeError(f"CrossAttentionBlock: channel mismatch (C={C}) != embed_dim({self.embed_dim})")

        # (B, N, C)
        rgb_tokens = feat_rgb.flatten(2).permute(0, 2, 1)
        depth_tokens = feat_depth.flatten(2).permute(0, 2, 1)

        if self.debug:
            print(f"[CrossAttention] B={B}, N={rgb_tokens.shape[1]}, C={C}")

        # RGB queries Depth
        q_rgb = self.query_rgb(rgb_tokens)
        k_depth = self.key_depth(depth_tokens)
        v_depth = self.value_depth(depth_tokens)
        out_rgb = self.attn_rgb(q_rgb, k_depth, v_depth)
        rgb2depth = out_rgb[0] if isinstance(out_rgb, tuple) else out_rgb

        # Depth queries RGB
        q_depth = self.query_depth(depth_tokens)
        k_rgb = self.key_rgb(rgb_tokens)
        v_rgb = self.value_rgb(rgb_tokens)
        out_depth = self.attn_depth(q_depth, k_rgb, v_rgb)
        depth2rgb = out_depth[0] if isinstance(out_depth, tuple) else out_depth

        # combine, project, norm
        fused = torch.cat([rgb2depth, depth2rgb], dim=-1)  # (B, N, 2C)
        fused = self.proj(fused)                           # (B, N, C)
        fused = self.norm(fused)

        # back to (B, C, H, W)
        fused = fused.permute(0, 2, 1).contiguous().view(B, C, H, W)
        # sanitize
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e6, neginf=-1e6)
        return fused


class MultiScaleFusion(nn.Module):
    """
    Multi-scale fusion wrapper with:
      - configurable max_tokens (token budget for attention)
      - adapters to project encoder channels -> embed_dim
      - forced downsample per-level
      - safe adaptive pooling
      - skip_highres option to avoid attention at level 0
    """
    def __init__(self, embed_dim=128, nhead=2, levels=4, encoder_channels=None,
                 debug=False, downsample_factors=(4, 2, 1, 1), max_tokens=4096):
        super().__init__()
        self.levels = levels
        self.embed_dim = embed_dim
        self.debug = debug
        self.max_tokens = int(max_tokens)

        # fusion blocks
        self.fusion_blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim=embed_dim, nhead=nhead, debug=debug)
            for _ in range(levels)
        ])

        if encoder_channels is None:
            encoder_channels = [embed_dim] * levels
        assert len(encoder_channels) == levels

        # adapters: project encoder channel -> embed_dim
        self.adapters = nn.ModuleList()
        for in_ch in encoder_channels:
            if in_ch == embed_dim:
                self.adapters.append(nn.Identity())
            else:
                self.adapters.append(nn.Sequential(
                    nn.Conv2d(in_ch, embed_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(num_groups=min(8, embed_dim), num_channels=embed_dim),
                    nn.ReLU(inplace=True)
                ))

        # forced downsamplers (conv stride)
        self.downsample_factors = list(downsample_factors)
        assert len(self.downsample_factors) == levels
        self.downsamplers = nn.ModuleList()
        for f in self.downsample_factors:
            if f <= 1:
                self.downsamplers.append(nn.Identity())
            else:
                # operates on embed_dim channels (after adapter)
                self.downsamplers.append(nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=f, padding=1, bias=False),
                    nn.GroupNorm(num_groups=min(8, embed_dim), num_channels=embed_dim),
                    nn.ReLU(inplace=True)
                ))

        # refine convs after fusion
        self.refine_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=min(8, embed_dim), num_channels=embed_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(levels)
        ])

    def forward(self, rgb_feats, depth_feats, max_tokens=None, skip_highres=True):
        fused_feats = []
        # allow per-call override but default to module's max_tokens
        token_budget = self.max_tokens if max_tokens is None else int(max_tokens)

        for i in range(self.levels):
            # project encoder channels -> embed_dim
            f_rgb = self.adapters[i](rgb_feats[i])
            f_depth = self.adapters[i](depth_feats[i])
            B, C, H, W = f_rgb.shape
            N = H * W

            if self.debug:
                print(f"[MultiScaleFusion] level={i} in {f_rgb.shape} N={N}")

            # skip attention at full-res (level 0) if requested
            if skip_highres and i == 0:
                fused = self.refine_blocks[i](f_rgb + f_depth)
                fused_feats.append(fused)
                continue

            # forced downsample to reduce token counts (conv stride)
            ds = self.downsample_factors[i]
            if ds > 1:
                f_rgb_ds = self.downsamplers[i](f_rgb)
                f_depth_ds = self.downsamplers[i](f_depth)
            else:
                f_rgb_ds = f_rgb
                f_depth_ds = f_depth

            Hp, Wp = f_rgb_ds.shape[-2:]
            Np = Hp * Wp

            # further adaptive pooling if still too large
            if Np > token_budget:
                scale = math.sqrt(Np / token_budget)
                out_h = max(1, math.floor(Hp / scale))
                out_w = max(1, math.floor(Wp / scale))
                f_rgb_p = F.adaptive_avg_pool2d(f_rgb_ds, (out_h, out_w))
                f_depth_p = F.adaptive_avg_pool2d(f_depth_ds, (out_h, out_w))
                if self.debug:
                    print(f"[MultiScaleFusion] pooled to {(out_h, out_w)} Np={out_h*out_w}")
            else:
                f_rgb_p, f_depth_p = f_rgb_ds, f_depth_ds

            # attention on reduced maps
            fused_p = self.fusion_blocks[i](f_rgb_p, f_depth_p)

            # upsample back to original resolution if needed
            if fused_p.shape[-2:] != (H, W):
                fused = F.interpolate(fused_p, size=(H, W), mode='bilinear', align_corners=False)
            else:
                fused = fused_p

            fused = self.refine_blocks[i](fused)

            # Best-effort free of temporaries to reduce peak memory
            try:
                # drop references to large intermediates
                fused_p = None
                f_rgb_p = None
                f_depth_p = None
                f_rgb_ds = None
                f_depth_ds = None
                if torch.cuda.is_available():
                    # hint to allocator — optional but helpful
                    torch.cuda.empty_cache()
            except Exception:
                pass

            fused = torch.nan_to_num(fused)
            fused_feats.append(fused)

        return fused_feats