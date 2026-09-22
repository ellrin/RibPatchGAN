"""Step 2 — train the LaMa fracture-inpainting GAN on rib patches.

Non-saturating GAN + lazy R1, with L1 (in-mask, fx sources), VGG
perceptual+style (fx sources), context L1 (outside mask, all sources) and a
seam-continuity loss. Every generated sample is conditioned as fx.

Outputs under --output-dir:
    checkpoints/G_latest.pt   <- consumed by step 3
    checkpoints/D_latest.pt
    checkpoints/G_best.pt     (lowest G_adv after the half-way gate)
    vis/epXXX.png             fixed-patch progress panel
    train_log.csv

Usage:
    python -m step2_train_inpainting_gan.train --config configs/default.yaml
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import load_config
from common.lama import build_lama_discriminator, build_lama_generator
from common.lama.blending import seam_loss
from step2_train_inpainting_gan.losses import VGGLoss, r1_penalty
from step2_train_inpainting_gan.data.dataset import (
    BalancedBatchSampler,
    CollatePatchPool,
    InpaintingPatchDataset,
    build_training_data_list,
    make_mask,
)


def nonsaturating_g_loss(d_fake):
    return F.softplus(-d_fake).mean()


def nonsaturating_d_loss(d_real, d_fake):
    return F.softplus(-d_real).mean() + F.softplus(d_fake).mean()


def d_forward(D, img):
    return D(img).mean(dim=[1, 2, 3])


def masked_l1_on_fx(fake, real, mask, fx_mask, weight):
    if fx_mask.sum() == 0:
        return fake.sum() * 0.0
    m = mask[fx_mask]
    return weight * (torch.abs(fake[fx_mask] - real[fx_mask]) * m).mean() / (m.mean() + 1e-8)


def context_l1(fake, real, mask, weight):
    if weight <= 0:
        return fake.sum() * 0.0
    context = 1 - mask
    return weight * (torch.abs(fake - real) * context).mean() / (context.mean() + 1e-8)


def vgg_on_fx(vgg, fake, real, fx_mask, weight):
    if fx_mask.sum() == 0 or weight <= 0:
        return fake.sum() * 0.0
    percep, style = vgg(fake[fx_mask], real[fx_mask])
    return weight * (percep + 100.0 * style)


@torch.no_grad()
def visualise(G, fixed_patches, fixed_labels, fixed_mask, noise_dim, epoch, vis_dir):
    """Three-row panel (original | masked | generated fx) on a fixed batch."""
    G.eval()
    B = fixed_patches.size(0)
    z = torch.randn(B, noise_dim, device=fixed_patches.device)
    generated = G(fixed_patches, fixed_mask, fixed_labels, z)
    G.train()

    fig, axes = plt.subplots(3, B, figsize=(B * 1.6, 5.2))
    rows = [fixed_patches, fixed_patches * (1 - fixed_mask), generated]
    titles = ["Original", "Masked", "Generated FX"]
    for r, (row_t, title) in enumerate(zip(rows, titles)):
        for c_idx, img in enumerate(row_t):
            ax = axes[r, c_idx] if B > 1 else axes[r]
            ax.imshow(img.cpu().permute(1, 2, 0).clamp(0, 1).numpy(), cmap="gray")
            ax.axis("off")
            if c_idx == 0:
                ax.set_title(title, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"ep{epoch:03d}.png"), dpi=80)
    plt.close()


def maybe_resume(ckpt_dir, modules, resume, device):
    if resume == "none":
        return False
    suffix = "best" if resume == "best" else "latest"
    for name, module in modules.items():
        path = os.path.join(ckpt_dir, f"{name}_{suffix}.pt")
        if not os.path.exists(path) and resume == "best":
            path = os.path.join(ckpt_dir, f"{name}_latest.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint for {name}: {path}")
        module.load_state_dict(torch.load(path, map_location=device))
    print(f"Resumed from {suffix} checkpoints (optimizer state reset)")
    return True


def train(cfg, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c = cfg.gan_training
    arch = cfg.lama

    result_dir = str(args.output_dir or cfg.step2_output_dir())
    vis_dir = os.path.join(result_dir, "vis")
    ckpt_dir = os.path.join(result_dir, "checkpoints")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    epochs = args.epochs or c.epochs

    print("=" * 70)
    print("Step 2 — LaMa inpainting GAN (non-saturating + R1)")
    print(f"Device: {device} | epochs: {epochs}")
    print(f"Train data: {cfg.datasets.train.images_root}")
    print(f"Output: {result_dir}")
    print("=" * 70)

    data_list = build_training_data_list(cfg.datasets.train.images_root, c.use_nofx_training)
    n_fx = sum(1 for *_, lbl in data_list if lbl == 1)
    n_norm = sum(1 for *_, lbl in data_list if lbl == 0)
    print(f"Dataset: fx={n_fx} nofx={n_norm} total={len(data_list)}")

    dataset = InpaintingPatchDataset(data_list, c, is_train=True)
    sampler = BalancedBatchSampler(dataset, c.batch_size, c.use_nofx_training)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=CollatePatchPool(c),
                        num_workers=args.num_workers, pin_memory=True)
    n_batches_per_epoch = len(sampler)
    if args.max_batches_per_epoch > 0:
        n_batches_per_epoch = min(n_batches_per_epoch, args.max_batches_per_epoch)
    print(f"Batches/epoch: {n_batches_per_epoch} (image batch_size={c.batch_size})")

    G = build_lama_generator(arch).to(device)
    D = build_lama_discriminator(arch).to(device)
    vgg = VGGLoss(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=c.g_lr, betas=(c.adam_b1, c.adam_b2))
    opt_D = torch.optim.Adam(D.parameters(), lr=c.d_lr, betas=(c.adam_b1, c.adam_b2))
    resumed = maybe_resume(ckpt_dir, {"G": G, "D": D}, args.resume, device)
    first_epoch = args.start_epoch if args.start_epoch is not None else 1

    fixed_patches = fixed_labels = fixed_mask = None
    log_path = os.path.join(result_dir, "train_log.csv")
    if not resumed or not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch",
                "D_real", "D_fake", "D_r1",
                "G_adv", "G_l1", "G_vgg", "G_seam",
            ])

    best_g_adv = float("inf")
    best_start_epoch = epochs // 2 + 1
    d_step_global = (first_epoch - 1) * n_batches_per_epoch

    for epoch in range(first_epoch, epochs + 1):
        d_real_acc = d_fake_acc = d_r1_acc = 0.0
        g_adv_acc = g_l1_acc = g_vgg_acc = g_seam_acc = 0.0
        n_batches = 0

        for batch_idx, (patches, masks, patch_labels) in enumerate(loader):
            if args.max_batches_per_epoch > 0 and batch_idx >= args.max_batches_per_epoch:
                break
            real = patches.to(device)
            mask = masks.to(device)
            source_labels = patch_labels.long().to(device)
            labels = source_labels
            if c.force_generate_fx:
                labels = torch.full_like(labels, arch.fx_label)
            B = real.size(0)
            if B < 2:
                continue
            fx_idx = source_labels == arch.fx_label
            if fx_idx.sum() < 2:
                continue

            if fixed_patches is None:
                fixed_patches = real[:8].detach()
                fixed_labels = labels[:8].detach()
                fixed_mask = make_mask(fixed_patches.size(0), device, c, jitter=False)

            with torch.no_grad():
                z = torch.randn(B, arch.noise_dim, device=device)
                fake = G(real, mask, labels, z)

            opt_D.zero_grad()
            real_fx = real[fx_idx]
            dr_adv = d_forward(D, real_fx)
            df_adv = d_forward(D, fake.detach())
            loss_D_adv = nonsaturating_d_loss(dr_adv, df_adv)
            loss_D_adv.backward()

            d_r1_val = 0.0
            d_step_global += 1
            if d_step_global % c.d_reg_interval == 0:
                real_r1 = real_fx.detach().requires_grad_(True)
                d_r1_adv = d_forward(D, real_r1)
                r1_loss = (c.r1_gamma / 2 * c.d_reg_interval) * r1_penalty(d_r1_adv, real_r1)
                r1_loss.backward()
                d_r1_val = r1_loss.item()
            opt_D.step()

            opt_G.zero_grad()
            z = torch.randn(B, arch.noise_dim, device=device)
            fake2 = G(real, mask, labels, z)
            gd_adv = d_forward(D, fake2)
            loss_adv = nonsaturating_g_loss(gd_adv)
            loss_l1 = masked_l1_on_fx(fake2, real, mask, fx_idx, c.lambda_rec)
            loss_ctx = context_l1(fake2, real, mask, c.lambda_context)
            loss_vgg = vgg_on_fx(vgg, fake2, real, fx_idx, c.lambda_vgg)
            loss_seam = seam_loss(fake2, real, mask, weight=c.lambda_seam,
                                  band_width=c.seam_loss_width)
            loss_G = loss_adv + loss_l1 + loss_ctx + loss_vgg + loss_seam
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt_G.step()

            d_real_acc += dr_adv.mean().item()
            d_fake_acc += df_adv.mean().item()
            d_r1_acc += d_r1_val
            g_adv_acc += loss_adv.item()
            g_l1_acc += loss_l1.item()
            g_vgg_acc += loss_vgg.item()
            g_seam_acc += loss_seam.item()
            n_batches += 1

            nb = n_batches
            print(f"\r[{epoch:03d}/{epochs}] {batch_idx + 1}/{n_batches_per_epoch} | "
                  f"D={d_real_acc / nb:.3f}/{d_fake_acc / nb:.3f} "
                  f"R1={d_r1_acc / nb:.3f} | "
                  f"Gadv={g_adv_acc / nb:.3f} Gl1={g_l1_acc / nb:.3f} "
                  f"Gvgg={g_vgg_acc / nb:.3f} Gseam={g_seam_acc / nb:.3f}",
                  end="", flush=True)

        if n_batches == 0:
            continue

        nb = n_batches
        g_adv_e = g_adv_acc / nb
        print(f"\n[{epoch:03d}/{epochs}] SUMMARY | "
              f"D={d_real_acc / nb:.3f}/{d_fake_acc / nb:.3f} R1={d_r1_acc / nb:.3f} | "
              f"Gadv={g_adv_e:.3f} Gl1={g_l1_acc / nb:.3f} Gvgg={g_vgg_acc / nb:.3f} "
              f"Gseam={g_seam_acc / nb:.3f}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{d_real_acc / nb:.6f}", f"{d_fake_acc / nb:.6f}", f"{d_r1_acc / nb:.6f}",
                f"{g_adv_e:.6f}", f"{g_l1_acc / nb:.6f}",
                f"{g_vgg_acc / nb:.6f}", f"{g_seam_acc / nb:.6f}",
            ])

        if fixed_patches is not None:
            visualise(G, fixed_patches, fixed_labels, fixed_mask, arch.noise_dim, epoch, vis_dir)

        torch.save(G.state_dict(), os.path.join(ckpt_dir, "G_latest.pt"))
        torch.save(D.state_dict(), os.path.join(ckpt_dir, "D_latest.pt"))
        if epoch >= best_start_epoch and g_adv_e < best_g_adv:
            best_g_adv = g_adv_e
            torch.save(G.state_dict(), os.path.join(ckpt_dir, "G_best.pt"))
            print(f"  checkpoint saved (new best G_adv={g_adv_e:.4f})")
        else:
            print("  checkpoint saved (latest)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default=None,
                        help="Default: <results_root>/step2_inpainting_gan")
    parser.add_argument("--max-batches-per-epoch", type=int, default=0,
                        help="Cap batches per epoch (smoke test). 0 = no cap.")
    parser.add_argument("--resume", choices=["none", "latest", "best"], default="none")
    parser.add_argument("--start-epoch", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, args)


if __name__ == "__main__":
    main()
