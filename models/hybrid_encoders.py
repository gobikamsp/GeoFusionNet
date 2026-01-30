# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Basic convolutional modules
# ---------------------------

class ConvBNReLU(nn.Sequential):
    """Convolution -> BatchNorm -> ReLU block."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )


class ResidualBlock(nn.Module):
    """Residual block with two ConvBNReLU layers."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBNReLU(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += x
        return self.relu(out)


# ---------------------------
# Transformer helper modules
# ---------------------------

class PatchEmbed(nn.Module):
    """Flatten 2D feature maps into token embeddings."""
    def __init__(self, in_ch, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).permute(2, 0, 1)   # (S, B, embed_dim)
        return x, (H, W)


class Unpatchify(nn.Module):
    """Restore 2D feature maps from token embeddings."""
    def forward(self, x, hw):
        S, B, C = x.shape
        H, W = hw
        x = x.permute(1, 2, 0).contiguous().view(B, C, H, W)
        return x


class TransformerBlockSeq(nn.Module):
    """Lightweight Transformer encoder operating on token sequences."""
    def __init__(self, embed_dim, nhead=4, num_layers=1, dim_feedforward=256, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation='relu', batch_first=False  # <-- you used S,B,E format
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)


# ---------------------------
# Hybrid RGB Encoder
# ---------------------------

class HybridRGBEncoder(nn.Module):
    def __init__(self, in_ch=3, base_channels=32, embed_dim=128, nhead=8):
        super().__init__()

        self.conv1 = nn.Sequential(
            ConvBNReLU(in_ch, base_channels),
            ConvBNReLU(base_channels, base_channels)
        )

        self.down1 = nn.Sequential(ConvBNReLU(base_channels, base_channels*2, stride=2),
                                   ResidualBlock(base_channels*2))
        self.down2 = nn.Sequential(ConvBNReLU(base_channels*2, base_channels*4, stride=2),
                                   ResidualBlock(base_channels*4))
        self.down3 = nn.Sequential(ConvBNReLU(base_channels*4, base_channels*8, stride=2),
                                   ResidualBlock(base_channels*8))

        self.patch_embed2 = PatchEmbed(base_channels*2, embed_dim)
        self.patch_embed3 = PatchEmbed(base_channels*4, embed_dim)
        self.patch_embed4 = PatchEmbed(base_channels*8, embed_dim)

        self.trans2 = TransformerBlockSeq(embed_dim, nhead=nhead)
        self.trans3 = TransformerBlockSeq(embed_dim, nhead=nhead)
        self.trans4 = TransformerBlockSeq(embed_dim, nhead=nhead)

        self.proj2 = nn.Conv2d(embed_dim, base_channels*2, 1, bias=False)
        self.proj3 = nn.Conv2d(embed_dim, base_channels*4, 1, bias=False)
        self.proj4 = nn.Conv2d(embed_dim, base_channels*8, 1, bias=False)

        self.unify1 = nn.Conv2d(base_channels, embed_dim, 1)
        self.unify2 = nn.Conv2d(base_channels*2, embed_dim, 1)
        self.unify3 = nn.Conv2d(base_channels*4, embed_dim, 1)
        self.unify4 = nn.Conv2d(base_channels*8, embed_dim, 1)

    def debug_print(self, name, t):
        print(f"\n===== DEBUG {name} =====")
        print("Input shape:", t.shape)
        print("dtype:", t.dtype)
        print("device:", t.device)
        print("embed_dim:", t.shape[-1])
        print("num_heads:", self.trans2.encoder.layers[0].self_attn.num_heads)
        print("embed_dim % num_heads:", t.shape[-1] % self.trans2.encoder.layers[0].self_attn.num_heads)
        print("========================\n")

    def forward(self, x, debug=False):
        f1 = self.conv1(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)

        t2_in, hw2 = self.patch_embed2(f2)
        if debug: self.debug_print("RGB t2 before transformer", t2_in)
        t2 = Unpatchify()(self.trans2(t2_in), hw2)

        t3_in, hw3 = self.patch_embed3(f3)
        if debug: self.debug_print("RGB t3 before transformer", t3_in)
        t3 = Unpatchify()(self.trans3(t3_in), hw3)

        t4_in, hw4 = self.patch_embed4(f4)
        if debug: self.debug_print("RGB t4 before transformer", t4_in)
        t4 = Unpatchify()(self.trans4(t4_in), hw4)

        t2 = self.proj2(t2)
        t3 = self.proj3(t3)
        t4 = self.proj4(t4)

        o1 = self.unify1(f1)
        o2 = self.unify2(t2)
        o3 = self.unify3(t3)
        o4 = self.unify4(t4)

        return [o1, o2, o3, o4]


# ---------------------------
# Hybrid Depth Encoder
# ---------------------------

class HybridDepthEncoder(nn.Module):
    def __init__(self, in_ch=1, base_channels=32, embed_dim=128, nhead=8):
        super().__init__()

        self.conv1 = nn.Sequential(
            ConvBNReLU(in_ch, base_channels),
            ConvBNReLU(base_channels, base_channels)
        )

        self.down1 = nn.Sequential(ConvBNReLU(base_channels, base_channels*2, stride=2),
                                   ResidualBlock(base_channels*2))
        self.down2 = nn.Sequential(ConvBNReLU(base_channels*2, base_channels*4, stride=2),
                                   ResidualBlock(base_channels*4))
        self.down3 = nn.Sequential(ConvBNReLU(base_channels*4, base_channels*8, stride=2),
                                   ResidualBlock(base_channels*8))

        self.patch_embed2 = PatchEmbed(base_channels*2, embed_dim)
        self.patch_embed3 = PatchEmbed(base_channels*4, embed_dim)
        self.patch_embed4 = PatchEmbed(base_channels*8, embed_dim)

        self.trans2 = TransformerBlockSeq(embed_dim, nhead=nhead)
        self.trans3 = TransformerBlockSeq(embed_dim, nhead=nhead)
        self.trans4 = TransformerBlockSeq(embed_dim, nhead=nhead)

        self.proj2 = nn.Conv2d(embed_dim, base_channels*2, 1, bias=False)
        self.proj3 = nn.Conv2d(embed_dim, base_channels*4, 1, bias=False)
        self.proj4 = nn.Conv2d(embed_dim, base_channels*8, 1, bias=False)

        self.unify1 = nn.Conv2d(base_channels, embed_dim, 1)
        self.unify2 = nn.Conv2d(base_channels*2, embed_dim, 1)
        self.unify3 = nn.Conv2d(base_channels*4, embed_dim, 1)
        self.unify4 = nn.Conv2d(base_channels*8, embed_dim, 1)

    def debug_print(self, name, t):
        print(f"\n===== DEBUG {name} =====")
        print("Input shape:", t.shape)
        print("dtype:", t.dtype)
        print("device:", t.device)
        print("embed_dim:", t.shape[-1])
        print("num_heads:", self.trans2.encoder.layers[0].self_attn.num_heads)
        print("embed_dim % num_heads:", t.shape[-1] % self.trans2.encoder.layers[0].self_attn.num_heads)
        print("========================\n")

    def forward(self, x, debug=False):
        f1 = self.conv1(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)

        t2_in, hw2 = self.patch_embed2(f2)
        if debug: self.debug_print("Depth t2 before transformer", t2_in)
        t2 = Unpatchify()(self.trans2(t2_in), hw2)

        t3_in, hw3 = self.patch_embed3(f3)
        if debug: self.debug_print("Depth t3 before transformer", t3_in)
        t3 = Unpatchify()(self.trans3(t3_in), hw3)

        t4_in, hw4 = self.patch_embed4(f4)
        if debug: self.debug_print("Depth t4 before transformer", t4_in)
        t4 = Unpatchify()(self.trans4(t4_in), hw4)

        t2 = self.proj2(t2)
        t3 = self.proj3(t3)
        t4 = self.proj4(t4)

        o1 = self.unify1(f1)
        o2 = self.unify2(t2)
        o3 = self.unify3(t3)
        o4 = self.unify4(t4)

        return [o1, o2, o3, o4]