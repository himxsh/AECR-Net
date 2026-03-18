import argparse
import csv
import glob
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from metrics import psnr, ssim
from models.AECRNet import Dehaze


def parse_args():
    parser = argparse.ArgumentParser(description="Run AECR-Net inference on NH-HAZE PNG pairs")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pk/.pt/.pth checkpoint")
    parser.add_argument("--nh_root", type=str, required=True, help="Path to NH-HAZE folder with *_hazy.png and *_GT.png")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save restored images + metrics")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format in {path}")

    cleaned = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        cleaned[nk] = v
    return cleaned


def collect_pairs(nh_root):
    hazy_files = sorted(glob.glob(os.path.join(nh_root, "*_hazy.png")))
    pairs = []
    for hazy in hazy_files:
        gt = hazy.replace("_hazy.png", "_GT.png")
        if os.path.exists(gt):
            pairs.append((hazy, gt))
    return pairs


def main():
    args = parse_args()
    device = resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    pairs = collect_pairs(args.nh_root)
    if not pairs:
        raise FileNotFoundError(f"No NH-HAZE pairs found in {args.nh_root}")

    model = Dehaze(3, 3).to(device)
    state = load_checkpoint(args.checkpoint, device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] Unexpected keys: {len(unexpected)}")
    model.eval()

    to_tensor = transforms.ToTensor()
    rows = []
    ssim_vals, psnr_vals = [], []

    for hazy_path, gt_path in pairs:
        hazy_img = Image.open(hazy_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        hazy = to_tensor(hazy_img).unsqueeze(0).to(device)
        gt = to_tensor(gt_img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(hazy)
        pred = pred.clamp(0, 1)

        stem = os.path.basename(hazy_path).replace("_hazy.png", "")
        out_path = os.path.join(args.output_dir, f"{stem}_restored.png")
        save_image(pred, out_path)

        cur_ssim = float(ssim(pred, gt).item())
        cur_psnr = float(psnr(pred.detach(), gt.detach()))
        ssim_vals.append(cur_ssim)
        psnr_vals.append(cur_psnr)
        rows.append([stem, cur_ssim, cur_psnr, out_path])

    csv_path = os.path.join(args.output_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "ssim", "psnr", "output_path"])
        writer.writerows(rows)

    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"checkpoint: {args.checkpoint}\n")
        f.write(f"device: {device}\n")
        f.write(f"images: {len(rows)}\n")
        f.write(f"mean_ssim: {np.mean(ssim_vals):.6f}\n")
        f.write(f"mean_psnr: {np.mean(psnr_vals):.6f}\n")

    print(f"Done. Images: {len(rows)}")
    print(f"mean_ssim: {np.mean(ssim_vals):.6f}")
    print(f"mean_psnr: {np.mean(psnr_vals):.6f}")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
