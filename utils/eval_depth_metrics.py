import torch

def compute_abs_rel_error(pred, gt):
    return torch.mean(torch.abs(pred - gt) / gt).item()

def compute_thresh_metrics(pred, gt):
    thresh = torch.max((gt / pred), (pred / gt))
    delta1 = (thresh < 1.25).float().mean().item()
    delta2 = (thresh < 1.25 ** 2).float().mean().item()
    delta3 = (thresh < 1.25 ** 3).float().mean().item()
    return delta1, delta2, delta3
