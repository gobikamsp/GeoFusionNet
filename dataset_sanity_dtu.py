import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from datasets.geofusion_dataset_dtu import GeoFusionDatasetDTU

dataset = GeoFusionDatasetDTU(
    datapath="datasets/dtu_training/mvs_training/dtu",
    listfile="lists/dtu/train.txt",
    nviews=3,
    use_input_depth=True,
    eval=False,
)

print("Dataset size:", len(dataset))

sample = dataset[0]
for k, v in sample.items():
    if hasattr(v, "shape"):
        print(k, v.shape)
