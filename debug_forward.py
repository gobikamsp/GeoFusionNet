# debug_forward.py
import torch, gc
from models import fusion_block as fbmod
from models.hybridfusionformer import HybridFusionFormer, Config

print("fusion file:", fbmod.__file__)
print("MultiScaleFusion debug default:", getattr(fbmod.MultiScaleFusion, "__doc__", "")[:80])

gc.collect(); torch.cuda.empty_cache()
cfg = Config()
m = HybridFusionFormer(cfg).cuda().eval()

rgb = torch.randn(1,3,576,768).cuda()
depth = torch.randn(1,1,576,768).cuda()
proj = [torch.eye(4).unsqueeze(0).cuda()]
depth_hypos = torch.linspace(0.5,10,48).unsqueeze(0).cuda()

with torch.no_grad():
    try:
        out = m(rgb, depth, proj, depth_hypos)
        print("Forward OK. Output shape:", out.shape)
    except Exception as e:
        print("Forward failed:", e)

print("CUDA allocated (MiB):", torch.cuda.memory_allocated()/1024**2)
print("CUDA reserved  (MiB):", torch.cuda.memory_reserved()/1024**2)
