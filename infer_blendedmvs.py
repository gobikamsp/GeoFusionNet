# infer_blendedmvs.py  (FULL + ORGANIZED OUTPUTS)
# -*- coding: utf-8 -*-

import os
import torch
import cv2
import numpy as np

from models.hybridfusionformer import HybridFusionFormer
from datasets.geofusion_dataset import GeoFusionDataset


# --------------------------------------------------------
# SAFE UINT8 CONVERTER
# --------------------------------------------------------
def to_uint8(img):
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()

    img = np.nan_to_num(img, nan=0.0)
    mn = float(np.nanmin(img))
    mx = float(np.nanmax(img))

    if mx - mn < 1e-8:
        out = np.clip(img - mn, 0, 1.0)
    else:
        out = (img - mn) / (mx - mn)

    return (out * 255).astype(np.uint8)


# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------
class Config:
    def __init__(self):
        self.dataset_root = "datasets/blendedmvs/"
        self.listfile = "datasets/blendedmvs/validation_list.txt"
        self.depth_num = 48
        self.decoder_in_channels = 128
        self.batch_size = 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()

CKPT_PATH = "checkpoints/epoch_010.pth"
OUTDIR = "inference_debug_outputs"
os.makedirs(OUTDIR, exist_ok=True)

device = cfg.device


# --------------------------------------------------------
# LOAD MODEL
# --------------------------------------------------------
model = HybridFusionFormer(cfg, debug=False).to(device)

# enable internal debug flags
for part in ["cost_volume", "decoder", "fusion"]:
    try:
        setattr(getattr(model, part), "debug", True)
    except Exception:
        pass

ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["model_state"], strict=False)
model.eval()


# --------------------------------------------------------
# LOAD DATASET
# --------------------------------------------------------
dataset = GeoFusionDataset(
    datapath=cfg.dataset_root,
    listfile=cfg.listfile,
    nviews=2,
    ndepths=cfg.depth_num
)


# ========================================================
# MAIN INFERENCE LOOP
# ========================================================
for idx in range(5):

    print("\n====================================================")
    print(f"                 SAMPLE {idx}")
    print("====================================================")

    # ----------------------------------------------------
    # LOAD SAMPLE
    # ----------------------------------------------------
    rgb, depth_in, proj_mats, depth_hypos, gt_depth = dataset[idx]

    rgb = rgb.unsqueeze(0).to(device)
    depth_in = depth_in.unsqueeze(0).to(device)
    depth_in = torch.clamp(depth_in, min=1e-4)

    # depth hypotheses
    if isinstance(depth_hypos, torch.Tensor):
        depth_hypos = depth_hypos.unsqueeze(0).to(device)
    else:
        depth_hypos = torch.as_tensor(depth_hypos, device=device).unsqueeze(0)

    depth_hypos = torch.clamp(depth_hypos, min=1e-4)

    # projection matrices
    if isinstance(proj_mats, (list, tuple)):
        proj_stack = torch.stack([p for p in proj_mats], dim=0)
        if proj_stack.dim() == 3:
            proj_stack = proj_stack.unsqueeze(0)
        proj_mats_input = proj_stack.to(device)
    else:
        proj_mats_input = proj_mats.to(device)

    # ----------------------------------------------------
    # INTERNAL SAFE DOWNSAMPLING
    # ----------------------------------------------------
    rgb_safe, depth_safe, scale_factors = model._maybe_downsample_inputs(
        rgb, depth_in
    )
    depth_safe = torch.clamp(depth_safe, min=1e-4)

    proj_list = model._ensure_proj_mats_device(
        proj_mats_input, rgb_safe.device
    )
    proj_list = model._sanitize_and_normalize_proj_mats(
        proj_list,
        rgb_safe.device,
        batch_size=rgb_safe.shape[0]
    )

    # ----------------------------------------------------
    # CREATE OUTPUT FOLDERS
    # ----------------------------------------------------
    sample_dir = os.path.join(OUTDIR, f"sample{idx}")
    costs_dir = os.path.join(sample_dir, "costslices")
    probs_dir = os.path.join(sample_dir, "prob_slices")

    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(costs_dir, exist_ok=True)
    os.makedirs(probs_dir, exist_ok=True)

    # ----------------------------------------------------
    # FORWARD PASS
    # ----------------------------------------------------
    with torch.no_grad():

        # -------------------------------
        # 1) ENCODERS
        # -------------------------------
        rgb_feats = model.rgb_encoder(rgb_safe)
        depth_feats = model.depth_encoder(depth_safe)

        # -------------------------------
        # 2) FUSION
        # -------------------------------
        fused_feats = model.fusion(
            rgb_feats,
            depth_feats,
            max_tokens=model.safe_max_tokens,
            skip_highres=model.skip_highres
        )

        print(f"\n[Sample {idx}] Fused features:")
        for i, f in enumerate(fused_feats):
            print(
                f" level {i}: {tuple(f.shape)}, "
                f"min={float(f.min()):.6f}, "
                f"max={float(f.max()):.6f}, "
                f"mean={float(f.mean()):.6f}, "
                f"std={float(f.std()):.6f}"
            )

        # -------------------------------
        # 3) COST VOLUME
        # -------------------------------
        cost_volume = model.cost_volume(
            fused_feats,
            proj_list,
            depth_hypos
        )

        print(f"\n[Sample {idx}] cost_volume: {tuple(cost_volume.shape)}")
        print(
            " stats:",
            float(cost_volume.min()),
            float(cost_volume.max()),
            float(cost_volume.mean()),
            float(cost_volume.std())
        )

        # per-depth stats
        cost_mean = cost_volume.mean(dim=(1, 3, 4))[0].cpu().numpy()
        cost_std = cost_volume.std(dim=(1, 3, 4))[0].cpu().numpy()

        print(" Depth mean (first 8):", cost_mean[:8].tolist())
        print(" Depth std  (first 8):", cost_std[:8].tolist())

        # -------------------------------
        # SAVE COST VOLUME SLICES
        # -------------------------------
        per_depth_map = cost_volume[0]
        if per_depth_map.shape[0] > 1:
            per_depth_map = per_depth_map.mean(dim=0)

        per_depth_map = per_depth_map.cpu().numpy()
        dcount = per_depth_map.shape[0]

        slice_idxs = [
            0,
            dcount // 2,
            dcount - 1
        ]

        for j, sid in enumerate(slice_idxs):
            cv2.imwrite(
                os.path.join(costs_dir, f"costslice_{j}_d{sid}.png"),
                to_uint8(per_depth_map[sid])
            )

        # -------------------------------
        # 4) DECODER
        # -------------------------------
        depth_map, prob = model.decoder(
            cost_volume,
            depth_hypos,
            fused_highres=fused_feats[0]
        )

        print(
            f"\n[Sample {idx}] depth_map stats:"
            f" min={float(depth_map.min())},"
            f" max={float(depth_map.max())},"
            f" mean={float(depth_map.mean())},"
            f" std={float(depth_map.std())}"
        )

        print(
            f"[Sample {idx}] prob shape={tuple(prob.shape)}, "
            f"min={float(prob.min())}, max={float(prob.max())}"
        )

        # save prob slices
        prob_np = prob[0].cpu().numpy()
        for j, sid in enumerate(slice_idxs):
            cv2.imwrite(
                os.path.join(probs_dir, f"prob_slice_{j}_d{sid}.png"),
                to_uint8(prob_np[sid])
            )

        # save final depth map
        depth_np = depth_map[0, 0].cpu().numpy()
        cv2.imwrite(
            os.path.join(sample_dir, "pred_depth.png"),
            to_uint8(depth_np)
        )

        print(f"[Sample {idx}] outputs saved to:")
        print(f"  {sample_dir}/")
        print(f"    +- costslices/")
        print(f"    +- prob_slices/")
        print(f"    +- pred_depth.png")
