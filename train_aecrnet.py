import json
import math
import os
import time
import warnings

import numpy as np
import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

from option import opt, log_dir
from metrics import psnr, ssim
from models.AECRNet import Dehaze
from models.CR import ContrastLoss
from data_utils.NH_png import NH_PNG_Dataset

warnings.filterwarnings("ignore")


def lr_schedule_cosdecay(t, total_steps, init_lr=opt.lr):
    return 0.5 * (1 + math.cos(t * math.pi / total_steps)) * init_lr


def unwrap_output(model_output):
    # Legacy checkpoints/scripts may return tuples. AECR-Net returns a tensor.
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output


def build_loader_registry():
    loaders = {}

    try:
        from data_utils.ITS_h5 import ITS_train_loader, ITS_test_loader
        from data_utils.NH import NH_train_loader, NH_test_loader
        from data_utils.DH import DH_train_loader, DH_test_loader

        loaders.update(
            {
                "ITS_train": ITS_train_loader,
                "ITS_test": ITS_test_loader,
                "NH_train": NH_train_loader,
                "NH_test": NH_test_loader,
                "DH_train": DH_train_loader,
                "DH_test": DH_test_loader,
            }
        )
    except Exception as e:
        if opt.runtime_mode == "legacy":
            raise
        print(f"[compat] H5 loaders unavailable: {e}")

    if opt.runtime_mode == "compat":
        train_size = opt.crop_size if opt.crop else "whole_img"
        nh_png_train = NH_PNG_Dataset(opt.nh_png_root, train=True, size=train_size)
        nh_png_test = NH_PNG_Dataset(opt.nh_png_root, train=False, size="whole_img")
        loaders["NH_PNG_train"] = DataLoader(dataset=nh_png_train, batch_size=opt.bs, shuffle=True)
        loaders["NH_PNG_test"] = DataLoader(dataset=nh_png_test, batch_size=1, shuffle=False)
        loaders["NH_PNG"] = loaders["NH_PNG_test"]

    return loaders


def test(net, loader_test):
    net.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ssims, psnrs = [], []
    for inputs, targets in loader_test:
        inputs = inputs.to(opt.device)
        targets = targets.to(opt.device)
        with torch.no_grad():
            pred = unwrap_output(net(inputs))
        ssims.append(ssim(pred, targets).item())
        psnrs.append(psnr(pred, targets))
    return float(np.mean(ssims)), float(np.mean(psnrs))


def train(net, loader_train, loader_test, optimizer, criterion_l1, criterion_contrast):
    losses = []
    ssims = []
    psnrs = []
    start_step = 0
    max_ssim = 0.0
    max_psnr = 0.0
    total_steps = opt.eval_step * opt.epochs
    start_time = time.time()
    log_file = os.path.join(log_dir, "train.log")

    if opt.resume and os.path.exists(opt.model_dir):
        ckp_path = os.path.join("./trained_models", opt.pre_model) if opt.pre_model != "null" else opt.model_dir
        ckp = torch.load(ckp_path, map_location=opt.device)
        net.load_state_dict(ckp["model"])
        optimizer.load_state_dict(ckp["optimizer"])
        losses = ckp.get("losses", [])
        start_step = ckp.get("step", 0)
        max_ssim = ckp.get("max_ssim", 0.0)
        max_psnr = ckp.get("max_psnr", 0.0)
        ssims = ckp.get("ssims", [])
        psnrs = ckp.get("psnrs", [])
        print(f"resume from {ckp_path}")
    else:
        print("train from scratch")

    train_iter = iter(loader_train)
    for step in range(start_step + 1, total_steps + 1):
        net.train()
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(step, total_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        else:
            lr = opt.lr

        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(loader_train)
            x, y = next(train_iter)

        x = x.to(opt.device)
        y = y.to(opt.device)
        out = unwrap_output(net(x))

        loss_rec = criterion_l1(out, y) if opt.w_loss_l1 > 0 else torch.tensor(0.0, device=opt.device)
        loss_vgg7 = (
            criterion_contrast(out, y, x)
            if (criterion_contrast is not None and opt.w_loss_vgg7 > 0)
            else torch.tensor(0.0, device=opt.device)
        )
        loss = opt.w_loss_l1 * loss_rec + opt.w_loss_vgg7 * loss_vgg7

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

        print(
            f"\rloss:{loss.item():.5f} l1:{(opt.w_loss_l1 * loss_rec):.5f} "
            f"contrast:{(opt.w_loss_vgg7 * loss_vgg7):.5f} "
            f"| step:{step}/{total_steps} | lr:{lr:.7f} "
            f"| time_used:{(time.time() - start_time)/60:.1f}m",
            end="",
            flush=True,
        )

        if step % opt.eval_step == 0:
            epoch = int(step / opt.eval_step)
            ssim_eval, psnr_eval = test(net, loader_test)
            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)

            log_line = f"\nstep:{step} | epoch:{epoch} | ssim:{ssim_eval:.4f} | psnr:{psnr_eval:.4f}"
            print(log_line)
            with open(log_file, "a") as f:
                f.write(log_line + "\n")

            save_model_dir = opt.model_dir
            if psnr_eval > max_psnr:
                max_ssim = max(max_ssim, ssim_eval)
                max_psnr = max(max_psnr, psnr_eval)
                save_model_dir = opt.model_dir + ".best"
                print(f"best update -> psnr:{max_psnr:.4f}, ssim:{max_ssim:.4f}")

            torch.save(
                {
                    "epoch": epoch,
                    "step": step,
                    "max_psnr": max_psnr,
                    "max_ssim": max_ssim,
                    "ssims": ssims,
                    "psnrs": psnrs,
                    "losses": losses,
                    "model": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "runtime_mode": opt.runtime_mode,
                },
                save_model_dir,
            )

    np.save(f"./numpy_files/{opt.model_name}_{total_steps}_losses.npy", losses)
    np.save(f"./numpy_files/{opt.model_name}_{total_steps}_ssims.npy", ssims)
    np.save(f"./numpy_files/{opt.model_name}_{total_steps}_psnrs.npy", psnrs)


def set_seed_torch(seed=2018):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    set_seed_torch(666)
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, "args.json"), "w") as f:
        json.dump(opt.__dict__, f, indent=2)

    loader_registry = build_loader_registry()
    if (
        opt.runtime_mode == "compat"
        and opt.trainset == "NH_train"
        and opt.testset == "NH_test"
        and "NH_train" not in loader_registry
    ):
        print("[compat] Falling back to NH_PNG loaders because H5 NH loaders are unavailable.")
        opt.trainset = "NH_PNG_train"
        opt.testset = "NH_PNG_test"

    if opt.trainset not in loader_registry or opt.testset not in loader_registry:
        raise KeyError(
            f"Unknown loader key(s): trainset={opt.trainset}, testset={opt.testset}. "
            f"Available: {sorted(loader_registry.keys())}"
        )

    loader_train = loader_registry[opt.trainset]
    loader_test = loader_registry[opt.testset]
    net = Dehaze(3, 3).to(opt.device)

    if opt.device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    print("epoch_size:", len(loader_train))
    print("Total_params:", sum(p.numel() for p in net.parameters() if p.requires_grad))
    print(f"runtime_mode: {opt.runtime_mode}")

    criterion_l1 = nn.L1Loss().to(opt.device)
    criterion_contrast = None
    if opt.w_loss_vgg7 > 0:
        criterion_contrast = ContrastLoss(ablation=opt.is_ab, force_legacy_cuda=(opt.runtime_mode == "legacy"))

    optimizer = optim.Adam(
        params=filter(lambda x: x.requires_grad, net.parameters()),
        lr=opt.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
    )
    optimizer.zero_grad()
    train(net, loader_train, loader_test, optimizer, criterion_l1, criterion_contrast)
