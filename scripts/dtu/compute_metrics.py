import numpy as np

def compute_metrics(gt, pred, mask):
    """
    gt: ground truth depth map
    pred: predicted depth map
    mask: valid pixel mask (where gt > 0)
    """
    gt = gt[mask]
    pred = pred[mask]

    # AbsRel
    abs_rel = np.mean(np.abs(gt - pred) / gt)

    # RMSE 
    rmse = np.sqrt(np.mean((gt - pred) ** 2))

    # Threshold Accuracy (delta < 1.25)
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()

    return rmse, abs_rel, a1