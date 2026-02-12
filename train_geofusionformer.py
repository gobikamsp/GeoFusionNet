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
        self.learning_rate = 5e-5 #2e-4
        self.depth_num = 8 #16
        self.accum_steps = 1

        # loss weights
        self.photo_weight = 1.0
        self.smooth_weight = 0.01
        self.consistency_weight = 0.0

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_workers = 4
        self.pin_memory = True

        # model
        self.embed_dim = 128
        self.base_channels = 32
        self.decoder_in_channels = self.embed_dim
        self.nhead = 2
        self.safe_input_hw = (160, 224) #(192, 256)
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
    """
    Accepts:
      - DTU-style dict with K / T_w2c / T_c2w
      - GeoMVSNet-style tensor (3x4 or 4x4)
    Returns:
      K, T_w2c
    """

    # -------- DTU dataset path --------
    if isinstance(P, dict):
        if "K" in P and "T_w2c" in P:
            return P["K"], P["T_w2c"]

        if "K" in P and "T_c2w" in P:
            return P["K"], torch.inverse(P["T_c2w"])

        raise ValueError(f"[split_proj_matrix] Unknown proj_mats dict keys: {P.keys()}")

    # -------- Tensor path (GeoMVSNet) --------
    if P.dim() == 2:
        P = P.unsqueeze(0)

    K = P[:, :3, :3]
    T = P[:, :3, 3:]
    # DTU camera translation is in millimeters ? convert to meters
    #T = T / 1000.0
    return K, T

def relative_pose_w2c(T_ref_w2c, T_src_w2c):
    """
    Returns transform from reference camera ? source camera
    """
    #T_src_w2c = torch.inverse(T_src_c2w)
    #return torch.bmm(T_src_w2c, T_ref_c2w)
    """
    Compute transform from reference camera to source camera
    when extrinsics are World ? Camera
    """
    T_rel = torch.bmm(T_src_w2c, torch.inverse(T_ref_w2c))
    return T_rel


# ------------------------------------------------------------------
# LOSSES
# ------------------------------------------------------------------


def photometric_loss(ref_img, src_img, depth_ref, K_ref, K_src, T_rel, eps=1e-6, debug=False):
    """
    Computes photometric loss between ref_img and src_img given depth and relative pose.

    Args:
        ref_img: [B, 1, H, W] or [B, H, W]
        src_img: [B, 1, H, W] or [B, H, W]
        depth_ref: [B, H, W] reference depth
        K_ref: [B, 3, 3] reference intrinsics
        K_src: [B, 3, 3] source intrinsics
        T_rel: [B, 4, 4] relative pose from ref to src
        eps: small constant for stability
        debug: print debug info if True
    Returns:
        Scalar photometric loss
    """
    if ref_img.dim() == 3:
        ref_img = ref_img.unsqueeze(1)
    if src_img.dim() == 3:
        src_img = src_img.unsqueeze(1)
# --- [ADD THIS FIX HERE] ---
    if K_ref.dim() == 2:
        K_ref = K_ref.unsqueeze(0)                
    if T_rel.dim() == 2:
        T_rel = T_rel.unsqueeze(0)
# ---------------------------

    B, C, H, W = ref_img.shape
    device = ref_img.device

    if depth_ref.dim() == 4:
        depth_ref = depth_ref.squeeze(1)

    # --- Pixel grid --
    dtype = depth_ref.dtype
    device = depth_ref.device

    y, x = torch.meshgrid(
    torch.arange(H, device=device, dtype=dtype),
    torch.arange(W, device=device, dtype=dtype),
    indexing="ij"
)   
    ones = torch.ones_like(x, device=device, dtype=dtype)
    pix = torch.stack([x, y, ones], dim=0)
    pix = pix.unsqueeze(0).repeat(B,1,1,1).view(B,3,-1)            # [B,3,HW]

    # --- Backproject to 3D ---
    K_ref_inv = torch.inverse(K_ref)
    cam_points = K_ref_inv @ pix
    cam_points *= depth_ref.view(B,1,-1)
    cam_points = torch.cat([cam_points, torch.ones(B,1,cam_points.shape[-1], device=device)], dim=1)  # [B,4,HW]

    # --- Transform to source camera ---
    #cam_points_src = torch.inverse(T_rel) @ cam_points
    cam_points_src = T_rel @ cam_points
    
    cam_points_src = cam_points_src[:, :3]
    z = cam_points_src[:, 2:3].clamp(min=1e-3)

    # --- Project to source image ---
    proj = K_src @ cam_points_src
    u = proj[:,0] / (proj[:,2] + eps)
    v = proj[:,1] / (proj[:,2] + eps)

    # --- Normalize for grid_sample ---
    x_norm = 2.0 * u / (W - 1) - 1.0
    y_norm = 2.0 * v / (H - 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).view(B,H,W,2)

    print("x_norm min/max:", x_norm.min().item(), x_norm.max().item())
    print("y_norm min/max:", y_norm.min().item(), y_norm.max().item())
    print("grid min/max:", grid.min().item(), grid.max().item())

    # --- Sample source image ---
    warped_src = F.grid_sample(src_img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

   # --- Valid mask (FIXED) ---
    eps = 1e-6

    valid_xy = (
    (x_norm >= -1.0 - eps) & (x_norm <= 1.0 + eps) &
    (y_norm >= -1.0 - eps) & (y_norm <= 1.0 + eps)
    )   

    valid_z = (z > 1e-6)

    valid = valid_xy.view(B, 1, H, W) & valid_z.view(B, 1, H, W)
    valid = valid.float()
    if valid.sum() < 1000:
        if debug: print("[FATAL] valid pixels too low:", valid.sum().item())
        return torch.zeros((), device=ref_img.device, dtype=ref_img.dtype)    # --- Photometric error ---
    l1 = torch.abs(ref_img - warped_src)
    l_ssim = ssim(ref_img, warped_src)
    photo = 0.85 * l_ssim + 0.15 * l1
    loss = (photo * valid).mean()
    

    if debug:
        print(f"[DEBUG] valid_pixels={valid.sum().item()}/{valid.numel()}, "
              f"x_norm min/max={x_norm.min().item():.3f}/{x_norm.max().item():.3f}, "
              f"y_norm min/max={y_norm.min().item():.3f}/{y_norm.max().item():.3f}, "
              f"z min/max={z.min().item():.3f}/{z.max().item():.3f}, loss={loss.item():.6f}")

    return loss

def ssim(x, y, C1=0.01**2, C2=0.03**2):
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)

    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    return torch.clamp((1 - ssim_n / ssim_d) / 2, 0, 1)


def smoothness(depth, img):
    # depth: [B,1,H,W]
    depth_dx = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])
    depth_dy = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])

    img_dx = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]), 1, keepdim=True)
    img_dy = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]), 1, keepdim=True)

    return (depth_dx * torch.exp(-img_dx)).mean() + \
           (depth_dy * torch.exp(-img_dy)).mean()


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
    """
    Validation loop computing:
      - Mean Absolute Error (EPE) with GT depth
      - Photometric loss
      - Smoothness loss
    """
    model.eval()
    total_epe, total_photo, total_smooth = 0.0, 0.0, 0.0
    count = 0

    for batch in loader:
        # Move inputs to device
        ref_img = batch["ref_img"].to(cfg.device)
        depth_in = batch["depth"].to(cfg.device)
        proj_mats = [p.to(cfg.device) for p in batch["proj_mats"]]
        depth_hypos = batch["depth_hypos"].to(cfg.device)
        gt_depth = batch.get("gt_depth", None)
        if gt_depth is not None:
            gt_depth = gt_depth.to(cfg.device)

        # Forward pass
        out = model(ref_img, depth_in, proj_mats, depth_hypos)
        if "stage1" not in out or "depth" not in out["stage1"]:
            raise RuntimeError("Model output missing stage1 depth")
        depth_pred = out["stage1"]["depth"][:, 0]  # [B,H,W]

        # Upsample depth to match RGB size
        if depth_pred.shape[-2:] != ref_img.shape[-2:]:
            depth_pred = F.interpolate(depth_pred.unsqueeze(1), size=ref_img.shape[-2:],
                                       mode="bilinear", align_corners=False).squeeze(1)

        # ---- Compute EPE / MAE ----
        if gt_depth is not None:
            if gt_depth.dim() == 3:
                gt_depth = gt_depth.unsqueeze(1)
            if gt_depth.shape[-2:] != depth_pred.shape[-2:]:
                gt_depth = F.interpolate(gt_depth, size=depth_pred.shape[-2:], mode="nearest")
            gt_depth = gt_depth.squeeze(1)
            mask = gt_depth > 0
            if mask.any():
                total_epe += torch.abs(depth_pred[mask] - gt_depth[mask]).mean().item()
                count += 1

        # ---- Compute photometric loss per view ----
        l_photo = 0.0
        num_views = len(proj_mats)
        Kref, Tref = split_proj_matrix(proj_mats[0])
        Kref = Kref.float().to(cfg.device)
        Tref_h = make_homogeneous(Tref).float().to(cfg.device)
        depth_metric = depth_pred.clamp(min=0.1, max=50.0)
        photo_losses = []

        for v in range(1, num_views):
            Ksrc, Tsrc = split_proj_matrix(proj_mats[v])
            Ksrc = Ksrc.float().to(cfg.device)
            Tsrc_h = make_homogeneous(Tsrc).float().to(cfg.device)
            # Relative pose
            
            Trel =  relative_pose_w2c(Tref, Tsrc)
            assert Trel.shape[-2:] == (4,4)
            assert torch.isfinite(Trel).all()


            # Extract view images
            ref_view = ref_img[:, 0:1]
            src_view = ref_img[:, v:v+1]

            l_view = photometric_loss(ref_view, src_view, depth_metric, Kref, Ksrc, Trel,debug=True)
            if torch.isfinite(l_view) and l_view.item() > 0:
                photo_losses.append(l_view)
                if len(photo_losses) > 0:
    	            l_photo = torch.stack(photo_losses).mean()
                else:
    	            l_photo = torch.zeros((), device=depth_metric.device, requires_grad=True)


        if num_views > 1:
            l_photo /= (num_views - 1)
        total_photo += l_photo.item()

        # ---- Compute smoothness ----
        l_smooth = smoothness(depth_metric, ref_img[:, 0:1])
        total_smooth += l_smooth.item()

    n = max(1, count)
    return {
        "epe": total_epe / n,
        "photometric_loss": total_photo / len(loader),
        "smoothness_loss": total_smooth / len(loader)
    }

def find_latest_checkpoint(ckpt_dir, dataset):
    if not os.path.exists(ckpt_dir):
        return None

    ckpts = [
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.startswith(dataset) and f.endswith(".pth")
    ]
    if len(ckpts) == 0:
        return None

    ckpts.sort()
    return ckpts[-1]

def compute_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------
    # DATA
    # ---------------------------
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
        train_set,
        cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )
    val_loader = DataLoader(val_set, 1, shuffle=False)

    # ---------------------------
    # MODEL
    # ---------------------------
    model = HybridFusionFormer(
        cfg,
        embed_dim=cfg.embed_dim,
        base_channels=cfg.base_channels,
        safe_input_hw=cfg.safe_input_hw,
        safe_max_tokens=cfg.safe_max_tokens,
        nhead=cfg.nhead,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    # ================= LR SCHEDULERS =================
    warmup_epochs = 5

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.1,
    total_iters=warmup_epochs
    )

    main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[20, 40],
    gamma=0.5
    )
# ================================================

    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 40], 0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.device == "cuda"))

    start_epoch = 1

    resume_path = None
    if args.resume:
        resume_path = args.resume
    else:
        resume_path = find_latest_checkpoint(cfg.save_dir, cfg.dataset)

    if resume_path is not None and os.path.isfile(resume_path):
        print(f"=> Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=cfg.device)

        model.load_state_dict(ckpt["model_state"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state"])

        
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])

        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"=> Resumed training from epoch {start_epoch}")
    else:
        print("=> No checkpoint found. Training from scratch.")

    # ===========================
    # TRAIN LOOP
    # ===========================
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        model.train()
        total_loss = 0.0
        best_val_epe = float("inf")

        optimizer.zero_grad(set_to_none=True)   # <-- HERE
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)

        for step, batch in enumerate(pbar, 1):
            # batch is a dict
            ref_img = batch["ref_img"].to(device)      # [B,3,H,W]
            src_img = batch["src_img"].to(device)      # [B,3,H,W]          
            depth_hypos = batch["depth_hypos"].to(device)  # [B,D]
            gt_depth = batch.get("gt_depth", None)

            ref_img = batch["ref_img"].to(device)    # [B,3,H,W]
            src_img = batch["src_img"].to(device)    # [B,3,H,W]
            depth_in = batch["depth"].to(device)     # [B,1,H,W]

            if ref_img.max() > 1.5:
                ref_img = ref_img / 255.0
                src_img = src_img / 255.0

            # ---------------------------
            # FORWARD (AMP SAFE)
            # ---------------------------
            with autocast(enabled=(cfg.device == "cuda")):
                proj_mats = batch["proj_mats"]   # already List[(B,4,4)]

                #proj_mats = [p.to(cfg.device) for p in proj_mats]
                proj_mats = [
                    p["T_c2w"].to(cfg.device) if isinstance(p, dict) else p.to(cfg.device) 
                    for p in proj_mats
		]


                out = model(ref_img,depth_in,proj_mats,depth_hypos)

                depth_pred = out["stage1"]["depth"]
	    # ---- extract reference depth ----
            depth_ref = depth_pred[:, 0:1]
	     
            depth_ref = depth_ref.float()

            if step == 1 and epoch == start_epoch:
                print(
        	"[DEPTH CHECK]",
        	depth_ref.min().item(),
        	depth_ref.max().item(),
        	depth_ref.mean().item()
    		)
            # ---- depth must be FP32 + clamped
            # depth_pred is low-res ? upsample to RGB resolution
            if depth_ref.shape[-2:] != ref_img.shape[-2:]:
                depth_ref = F.interpolate(
        	depth_ref,
        	size=ref_img.shape[-2:],
        	mode="bilinear",
        	align_corners=False
    		)
            _, _, H_img, W_img = ref_img.shape
            _, _, H_k, W_k = depth_ref.shape

            
	     # 3. Remove NaN / Inf (non-negotiable)
            depth_ref = torch.nan_to_num(
    		depth_ref,
    		nan=0.0,
    		posinf=0.0,
    		neginf=0.0
			)
		# --------------------------------------------------
		# ?? SPLIT DEPTH HERE (CRITICAL)
		# --------------------------------------------------
            

		# 4. Clamp to valid DTU range
            depth_metric = depth_ref
            depth_metric = depth_metric.clamp(min=0.1, max=50.0).float()  # ? USED FOR PHOTOMETRIC LOSS
 
            #depth_raw = depth_metric.detach()   # ? used for smoothness only

            # ---------------------------
            # GEOMETRY (FP32 ONLY)
            # ---------------------------
            Kref, Tref = split_proj_matrix(batch["proj_mats"][0])
            Kref = Kref.to(device=device, dtype=torch.float32)
            Tref = Tref.to(device=device, dtype=torch.float32)	   

	  

            with torch.cuda.amp.autocast(enabled=False):
                Tref_h = Tref.float()

            l_photo = 0.0
            num_views = len(batch["proj_mats"])
            photo_losses = []

            for v in range(1, num_views):
                Ksrc, Tsrc = split_proj_matrix(batch["proj_mats"][v])
                Ksrc = Ksrc.to(device=device, dtype=torch.float32)
                Tsrc = Tsrc.to(device=device, dtype=torch.float32)

                with torch.cuda.amp.autocast(enabled=False):
                    Tsrc_h = Tsrc.float()
                    Tref_h = Tref.float()
                    Tref_c2w = torch.inverse(Tref_h)
                    Trel = torch.bmm(Tsrc_h, Tref_c2w)
    		    # DTU: Tref_h, Tsrc_h are camera-to-world (c2w)
                    
                    #Trel = relative_pose_w2c(Tref, Tsrc)
                    if epoch == start_epoch and step == 1:
                        center = torch.tensor([0,0,1,1.0], device=Trel.device).view(1,4,1)
                        pt = Trel @ center
                        print("[SANITY] z_src:", pt[:,2].item())

                #ref_img = rgb[:, 0:1].mean(1, keepdim=True)  # [B,1,H,W]
                #src_img = rgb[:, v:v+1].mean(1, keepdim=True)

                ref_view = ref_img[:, 0:1].float()
                src_view = src_img[:, v:v+1].float()


                # ---- photometric loss MUST be FP32
                assert depth_pred.dim() in [3, 4]
                assert Kref.shape[-2:] == (3, 3)
                assert depth_ref.shape[1] == 1
                assert Trel.shape[-2:] == (4, 4)
                assert Ksrc.shape[-2:] == (3, 3)
                

                with torch.cuda.amp.autocast(enabled=False):
                    l_view = photometric_loss(
    			ref_view,
    			src_view,
    			depth_metric,  	
    			Kref,
    			Ksrc,
    			Trel,
			debug=True  # <-- ENABLE DEBUG
			)
                    #l_photo += l_view
                    if torch.isfinite(l_view) and l_view.item() > 0:
                        photo_losses.append(l_view)

                        if len(photo_losses) > 0:
                            l_photo = torch.stack(photo_losses).mean()
                        else:
                            l_photo = torch.zeros((), device=depth_metric.device, requires_grad=True)
        # -----------------------------
        # DEBUG: Check if loss is zero or NaN
        # -----------------------------
                    if not torch.isfinite(l_view) or l_view.item() == 0.0:
                        print(f"[DEBUG][Step {step} View {v}] Photometric loss = {l_view.item():.6f}")
                        print(f"  Depth min/max/mean: {depth_metric.min().item():.3f}/"
                        f"{depth_metric.max().item():.3f}/{depth_metric.mean().item():.3f}")
            # Optional: check center pixel projection
                        H_img, W_img = ref_view.shape[-2:]
                        center_pix = torch.tensor([[W_img//2, H_img//2, 1]], device=depth_metric.device).float()
                        cam_center = torch.inverse(Kref) @ center_pix.T * depth_metric[0,0,H_img//2,W_img//2]
                        cam_center_h = torch.cat([cam_center, torch.ones(1,1,1,device=depth_metric.device)], dim=1)
                        pt_src = Trel @ cam_center_h
                        print(f"  Center pixel projected z_src: {pt_src[0,2,0].item()}")

            #l_photo /= max(1, num_views - 1)

            # ---------------------------
            # OTHER LOSSES (SAFE)
            # ---------------------------
            l_smooth = smoothness(depth_metric, ref_img[:, 0:1])
            l_cons = depth_consistency(depth_pred)

            loss = (
                cfg.photo_weight * l_photo +
                cfg.smooth_weight * l_smooth +
                cfg.consistency_weight * l_cons
            )
            # ---------------------------
            # BACKWARD
            # ---------------------------
            loss = loss / cfg.accum_steps
	    # # Replace NaN loss with zero (SAFE)
            if not torch.isfinite(loss):
                loss = torch.zeros_like(loss)

            scaler.scale(loss).backward()

	    # ---- detect NaN / Inf gradients
            found_inf = False
	    
	    # ---- GRADIENT SANITY CHECK (debug)
            #if step == 1:   # only first step of epoch
             #   scaler.unscale_(optimizer)
              #  grad_norm = compute_grad_norm(model)
               # print(f"[DEBUG] Grad norm @ epoch {epoch}: {grad_norm:.6f}")
            if step % cfg.accum_steps == 0 or step == len(train_loader):
                #scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()

            if step % 20 == 0:
                pbar.set_postfix(
                    	loss=f"{loss.item():.4f}" if torch.is_tensor(loss) else f"{loss:.4f}",
    			photo=f"{l_photo.item():.4f}" if torch.is_tensor(l_photo) else f"{l_photo:.4f}",
    			smooth=f"{l_smooth.item():.4f}" if torch.is_tensor(l_smooth) else f"{l_smooth:.4f}",
    			cons=f"{float(l_cons):.4f}",
    			lr=f"{optimizer.param_groups[0]['lr']:.2e}"
                )

       
	# ===== LR STEP =====
        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()
        avg_loss = total_loss / len(train_loader)

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": {
    		"warmup": warmup_scheduler.state_dict(),
    		"main": main_scheduler.state_dict()
		},
            "scaler_state": scaler.state_dict(),
            "loss": avg_loss,
        }, f"{cfg.save_dir}/{cfg.dataset}_epoch_{epoch:03d}.pth")

        val_epe = validate(model, val_loader, cfg)
        if val_epe < best_val_epe:
    	    best_val_epe = val_epe
    	    torch.save(
            model.state_dict(),
            f"{cfg.save_dir}/geofusionformer_dtu_best.pth"
         )
    	    print(f"[INFO] Best model updated (Val EPE = {val_epe:.4f})")

	# Always save last epoch (resume/reproducibility)
        torch.save(
    	model.state_dict(),
    	f"{cfg.save_dir}/geofusionformer_dtu_last.pth"
	)
        print(f"Epoch {epoch} | Train Loss {avg_loss:.4f} | Val EPE {val_epe:.4f}")
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    train()
