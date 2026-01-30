from datasets.geofusion_dataset import GeoFusionDataset

ds = GeoFusionDataset(
    root_dir="datasets/dtu_training/mvs_training/dtu",
    split="train",
    use_input_depth=True,
    eval=False
)

print("Dataset size:", len(ds))

sample = ds[0]
for k, v in sample.items():
    if hasattr(v, "shape"):
        print(k, v.shape)
