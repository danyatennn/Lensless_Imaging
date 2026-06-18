import argparse
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from torchmetrics import MeanSquaredError
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)


def load_image(path, shape=None):
    """
    Load an image as a (1, 3, H, W) float tensor in [0, 1].

    Args:
        path (Path): path to the image.
        shape (tuple | None): if given, resize to (H, W).
    Returns:
        Tensor: image tensor of shape (1, 3, H, W).
    """
    image = Image.open(path).convert("RGB")
    if shape is not None:
        image = image.resize((shape[1], shape[0]), Image.BICUBIC)
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def resolve_ground_truth_dir(path):
    """
    Resolve the directory that holds the ground-truth images.
    """
    path = Path(path)
    if (path / "lensed").exists():
        return path / "lensed"
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Compute PSNR, SSIM, LPIPS, and MSE between reconstructions "
        "and ground-truth images."
    )
    parser.add_argument(
        "-g",
        "--ground_truth",
        required=True,
        help="Path to ground-truth images (or a data dir with a 'lensed' subdir).",
    )
    parser.add_argument(
        "-r",
        "--reconstructions",
        required=True,
        help="Path to the directory with reconstructed images.",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device
    gt_dir = resolve_ground_truth_dir(args.ground_truth)
    rec_dir = Path(args.reconstructions)

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    mse = MeanSquaredError().to(device)
    perceptual = lpips.LPIPS(net="vgg").to(device)

    scores = {"PSNR": [], "SSIM": [], "MSE": [], "LPIPS": []}

    for rec_path in sorted(rec_dir.glob("*.png")):
        gt_path = gt_dir / rec_path.name
        if not gt_path.exists():
            continue

        rec = load_image(rec_path).to(device)
        gt = load_image(gt_path, shape=rec.shape[-2:]).to(device)

        scores["PSNR"].append(psnr(rec, gt).item())
        scores["SSIM"].append(ssim(rec, gt).item())
        scores["MSE"].append(mse(rec, gt).item())
        scores["LPIPS"].append(perceptual(rec * 2 - 1, gt * 2 - 1).mean().item())

    n = len(scores["PSNR"])
    if n == 0:
        print("No matching reconstruction / ground-truth pairs were found.")
        return

    print(f"Computed metrics over {n} images:")
    for name, values in scores.items():
        print(f"    {name:6s}: {np.mean(values):.4f}")


if __name__ == "__main__":
    main()
