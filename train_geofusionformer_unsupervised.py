# -*- coding: utf-8 -*-
import os, sys, gc, argparse, warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("torch").setLevel(logging.ERROR)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast

# ------------------------------------------------------------------
# CUDA SAFETY
# ------------------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ------------------------------------------------------------------
# PROJECT PATH
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from models.hybridfusionformer import HybridFusionFormer
from datasets.geofusion_dataset_dtu import GeoFusionDatasetDTU

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
class Config:
    def __init__(self, dataset="dtu", exp_name="unsup"):

        self.dataset = dataset.lower()
        self.exp_name = exp_name

        # training
        self.batch_size = 1
        self.num_epochs = 50
        self.learning_rate = 5e-5
        self.depth_num = 16
        self.accum_steps = 1

        # loss weights
        self.photo_weight = 1.0
        self.smooth_weight = 0.1
        self.consistency_weight = 0.05

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_workers = 4
        self.pin_memory = True

        # model
        self.embed_dim = 128
        self.base_channels = 32
        self.decoder_in_channels = self.embed_dim
        self.nhead = 2
        self.safe_input_hw = (192, 256)
        self.safe_max_tokens = 4096

        # paths
        self.save_dir = "checkpoints"
        os.makedirs(self.save_dir, exist_ok=True)

        # dataset
        self.dataset_root = "datasets/dtu_training/mvs_training/dtu"
        self.listfile = "lists/dtu/train.txt"
        self.val_listfile = "lists/dtu/test.txt"
        self.nviews = 2


# ------------------------------------------------------------------
# GEOMETRY
# ------------------------------------------------------------------
def backproject(depth, Kinv):
    """
    depth: [B,1,H,W]
    Kinv:  [B,3,3] or [3,3]
    """
    B, _, H, W = depth.shape
    device, dtype = depth.device, depth.dtype

    # Create pixel grid as float
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij"
    )
    pix = torch.stack([x, y, torch.ones_like(x)], 0).reshape(3, -1)  # [3,H*W]
    pix = pix.unsqueeze(0).repeat(B, 1, 1)  # [B,3,H*W]

    # Ensure Kinv has batch dimension
    if Kinv.dim() == 2:
        Kinv = Kinv.unsqueeze(0)
    if Kinv.shape[0] == 1 and B > 1:
        Kinv = Kinv.repeat(B, 1, 1)

    # Convert Kinv to same dtype/device as depth
    Kinv = Kinv.to(dtype=dtype, device=device)

    cam = torch.bmm(Kinv, pix)              # [B,3,H*W]
    cam = cam * depth.view(B, 1, -1)       # scale by depth
    return cam

def make_homogeneous(T):
    """
    T: (B, 3, 4)
    Returns: (B, 4, 4) in same dtype/device as T
    """
    B, device, dtype = T.shape[0], T.device, T.dtype
    bottom = torch.tensor([0, 0, 0, 1], device=device, dtype=dtype).view(1, 1, 4)
    bottom = bottom.repeat(B, 1, 1)
    return torch.cat([T, bottom], dim=1)  # [B,4,4]


def project(points, K, T):
    """
    points: [B,3,N]
    K: [B,3,3]
    T: [B,4,4] or [B,3,4]
    """
    B, _, N = points.shape
    dtype, device = points.dtype, points.device

    # make T 4x4 homogeneous if needed
    if T.shape[1] == 3:
        T = make_homogeneous(T)

    # convert K and T to same dtype/device as points
    K = K.to(dtype=dtype, device=device)
    T = T.to(dtype=dtype, device=device)

    ones = torch.ones(B, 1, N, device=device, dtype=dtype)
    pts_h = torch.cat([points, ones], 1)  # [B,4,N]
    cam = torch.bmm(T, pts_h)             # [B,4,N]
    proj_mat = K.bmm(cam[:, :3, :])       # [B,3,N]
    xy = proj_mat[:, :2] / (proj_mat[:, 2:3] + 1e-8)
    return xy


def split_proj_matrix(P):
    if P.dim() == 2:
        P = P.unsqueeze(0)
    K = P[:, :3, :3]
    T = P[:, :3, :]
    return K, T


# ------------------------------------------------------------------
# LOSSES
# ------------------------------------------------------------------
def photometric_loss(ref, src, proj):
    """
    ref, src: [B,3,H,W]
    """
    B, _, H, W = ref.shape

    cam = backproject(proj["depth"], torch.inverse(proj["Kref"]))
    pix = project(cam, proj["Ksrc"], proj["Tsrc"])

    x = (pix[:, 0] / (W - 1) - 0.5) * 2
    y = (pix[:, 1] / (H - 1) - 0.5) * 2
    grid = torch.stack([x, y], -1).view(B, H, W, 2)

    warped = F.grid_sample(src, grid, align_corners=True)
    return F.l1_loss(warped, ref)


def smoothness(depth, img):
    dx = torch.abs(depth[..., 1:] - depth[..., :-1])
    dy = torch.abs(depth[..., 1:, :] - depth[..., :-1, :])
    img_dx = torch.mean(torch.abs(img[..., 1:] - img[..., :-1]), 1, keepdim=True)
    img_dy = torch.mean(torch.abs(img[..., 1:, :] - img[..., :-1, :]), 1, keepdim=True)
    return (dx * torch.exp(-img_dx)).mean() + (dy * torch.exp(-img_dy)).mean()


def depth_consistency(depths):
    loss = 0.0
    for v in range(1, depths.shape[1]):
        loss += torch.mean(torch.abs(depths[:, 0] - depths[:, v]))
    return loss / max(1, depths.shape[1] - 1)


# ------------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, cfg):
    model.eval()
    err, n = 0.0, 0

    for batch in loader:
        rgb, depth_in, proj, hypos, gt = batch
        rgb = rgb.to(cfg.device)
        depth_in = depth_in.to(cfg.device)
        hypos = hypos.to(cfg.device)
        gt = gt.to(cfg.device)
        proj = [p.to(cfg.device) for p in proj]

        out = model(rgb, depth_in, proj, hypos)
        assert "stage1" in out and "depth" in out["stage1"]

        pred = out["stage1"]["depth"][:, 0]

        if pred.shape != gt.shape:
            gt = F.interpolate(gt.unsqueeze(1), pred.shape[-2:], mode="nearest").squeeze(1)

        mask = gt > 0
        if mask.any():
            err += torch.abs(pred[mask] - gt[mask]).mean().item()
            n += 1

    return err / max(1, n)


# ------------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------------
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dtu")
    parser.add_argument("--exp_name", default="unsup")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    cfg = Config(args.dataset, args.exp_name)

    train_set = GeoFusionDatasetDTU(
        cfg.dataset_root, cfg.listfile,
        nviews=cfg.nviews, ndepths=cfg.depth_num,
        use_input_depth=True, eval=False
    )
    val_set = GeoFusionDatasetDTU(
        cfg.dataset_root, cfg.val_listfile,
        nviews=cfg.nviews, ndepths=cfg.depth_num,
        use_input_depth=True, eval=True
    )

    train_loader = DataLoader(
        train_set, cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )
    val_loader = DataLoader(val_set, 1, shuffle=False)

    model = HybridFusionFormer(
        cfg,
        embed_dim=cfg.embed_dim,
        base_channels=cfg.base_channels,
        safe_input_hw=cfg.safe_input_hw,
        safe_max_tokens=cfg.safe_max_tokens,
        nhead=cfg.nhead,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 40], 0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.device == "cuda"))

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=cfg.device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"=> Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        model.train()
        total = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)

        for step, batch in enumerate(pbar, 1):
            rgb, depth_in, proj, hypos, _ = batch
            rgb = rgb.to(cfg.device)
            depth_in = depth_in.to(cfg.device)
            hypos = hypos.to(cfg.device)
            proj = [p.to(cfg.device) for p in proj]

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(cfg.device == "cuda")):
                out = model(rgb, depth_in, proj, hypos)
                depth_pred = out["stage1"]["depth"]
                depth_ref = depth_pred[:, 0:1]

                Kref, Tref = split_proj_matrix(proj[0])

                l_photo = 0.0
                for v in range(1, rgb.shape[1]):
                    Ksrc, Tsrc = split_proj_matrix(proj[v])
                    # Convert both source and reference to homogeneous 4x4
                    Tref_h = make_homogeneous(Tref)
                    Tsrc_h = make_homogeneous(Tsrc)

		    # Now compute relative pose safely
                    Trel = torch.bmm(Tsrc_h, torch.inverse(Tref_h))

                    proj_data = {
                        "Kref": Kref,
                        "Ksrc": Ksrc,
                        "Tsrc": Trel,
                        "depth": depth_ref,
                    }
                    ref = rgb[:, 0:1] if rgb.dim() == 4 else rgb[:, 0:1, :, :]
                    src = rgb[:, v:v+1] if rgb.dim() == 4 else rgb[:, v:v+1, :, :]
                    l_photo += photometric_loss(ref, src, proj_data)

                l_photo /= max(1, rgb.shape[1] - 1)
                l_smooth = smoothness(depth_ref, rgb[:, 0])
                l_cons = depth_consistency(depth_pred)

                loss = (
                    cfg.photo_weight * l_photo +
                    cfg.smooth_weight * l_smooth +
                    cfg.consistency_weight * l_cons
                )

            if not torch.isfinite(loss):
                print("[WARN] Non-finite loss, skipping")
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total += loss.item()
            if step % 20 == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}"
                )

        scheduler.step()
        avg = total / len(train_loader)

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "loss": avg,
        }, f"{cfg.save_dir}/{cfg.dataset}_epoch_{epoch:03d}.pth")

        val_epe = validate(model, val_loader, cfg)
        print(f"Epoch {epoch} | Loss {avg:.4f} | Val EPE {val_epe:.4f}")

        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    train()
