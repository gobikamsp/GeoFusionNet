import os
import numpy as np
import cv2
import glob
import tqdm

def simulate_input_depths(scan_path, output_dir):
    input_dir = os.path.join(scan_path, "rendered_depth_maps")
    os.makedirs(output_dir, exist_ok=True)

    depth_files = sorted(glob.glob(os.path.join(input_dir, "*.pfm")))

    for pfm_path in tqdm.tqdm(depth_files, desc=os.path.basename(scan_path)):
        try:
            # Load .pfm depth file
            with open(pfm_path, "rb") as f:
                header = f.readline().rstrip()
                if header.decode("utf-8") == "Pf":
                    channels = 1
                else:
                    raise Exception("Not a valid PFM file")

                dims = f.readline().decode("utf-8")
                width, height = map(int, dims.strip().split())
                scale = float(f.readline().decode("utf-8"))
                data = np.fromfile(f, "<f4" if scale < 0 else ">f4")
                data = np.reshape(data, (height, width))

            # Simulate depth: Add Gaussian noise or clip
            noisy_depth = data + np.random.normal(0, 0.01, size=data.shape)
            noisy_depth = np.clip(noisy_depth, 0.0, None).astype(np.float32)

            # Save as .npy
            filename = os.path.basename(pfm_path).replace(".pfm", ".npy")
            np.save(os.path.join(output_dir, filename), noisy_depth)

        except Exception as e:
            print(f"[!] Failed to process {pfm_path}: {e}")


if __name__ == "__main__":
    root_dir = "datasets/blendedmvs"  # change if needed

    scans = sorted(os.listdir(root_dir))
    for scan in scans:
        scan_path = os.path.join(root_dir, scan)
        if not os.path.isdir(scan_path):
            continue
        output_path = os.path.join(scan_path, "input_depth_maps")
        simulate_input_depths(scan_path, output_path)
