import torch

pth_path = "outputs/dtu/000000.pth"
d = torch.load(pth_path, map_location="cpu")

print("Type:", type(d))

if isinstance(d, dict):
    for k in d.keys():
        print("Key:", k, "shape:", d[k].shape)
    d = list(d.values())[0]

print("Shape:", d.shape)
print("Min:", d.min().item())
print("Max:", d.max().item())
print("Mean:", d.mean().item())
print("Std:", d.std().item())
