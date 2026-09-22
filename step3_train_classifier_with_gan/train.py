"""Step 3 — train the patch classifier with GAN augmentation and cooperative
G feedback (the proposed method).

Per batch:
  * real-patch BCE (fg/bg pooled, dynamic weights) stays intact;
  * generated fx patches (mixed_pos paste from the step-2 LaMa G) add a
    low-weight auxiliary BCE;
  * every `g_update_interval` steps (from `g_update_start_epoch`) G takes one
    distribution-stable step against the frozen EMA teacher.

Checkpoint selection: valid-set image-topk AUC of the EMA teacher.
Outputs under --output-dir: metrics.csv, best_auc.pt, latest.pt.
Run test.py afterwards for the single test inference pass.

Usage:
    python -m step3_train_classifier_with_gan.train --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import load_config
from step3_train_classifier_with_gan.data.items import build_items
from step3_train_classifier_with_gan.data.preprocessing import load_site_stats
from step3_train_classifier_with_gan.engine import (
    dynamic_loss_weights,
    evaluate,
    flatten_bags,
    make_loader,
    make_warmup_cosine_scheduler,
    seed_all,
)
from step3_train_classifier_with_gan.gan_augmentation.feedback import (
    build_ema_teacher,
    cooperative_g_update,
    ema_update_teacher,
    per_patch_image_label,
)
from step3_train_classifier_with_gan.gan_augmentation.generator_wrapper import (
    PretrainedInpaintingGan,
)
from step3_train_classifier_with_gan.gan_augmentation.paste import fake_fx_paste_flat
from step3_train_classifier_with_gan.losses import patch_bce
from step3_train_classifier_with_gan.metrics import fmt
from step3_train_classifier_with_gan.models.patch_classifier import build_patch_classifier


def _linear_anneal(epoch: int, start_val: float, end_val: Optional[float],
                   anneal_end_epoch: int) -> float:
    if end_val is None or anneal_end_epoch <= 0:
        return float(start_val)
    if epoch >= anneal_end_epoch:
        return float(end_val)
    t = max(0.0, min(1.0, float(epoch - 1) / max(1, anneal_end_epoch - 1)))
    return float(start_val + t * (end_val - start_val))


def _save_ckpt(model, epoch: int, path: Path):
    torch.save({"model": model.state_dict(), "epoch": epoch}, path)


def train_epoch(student, teacher, gan, loader, opt, scaler, g_opt, g_scaler,
                cls_cfg, device, epoch: int, total_epochs: int,
                w_fg: float, w_bg: float, scheduler=None,
                run_name: str = "v5") -> Dict[str, float]:
    student.train()
    ga = cls_cfg.gan_augmentation
    fake_enabled = (
        gan is not None
        and bool(ga.use_fake)
        and epoch >= int(ga.fake_start_epoch)
    )
    active_g_enabled = (
        fake_enabled
        and bool(ga.active_g)
        and epoch >= int(ga.g_update_start_epoch)
    )
    # fractions anneal linearly over the full run
    cur_fraction = _linear_anneal(
        epoch, float(ga.fake_fraction), ga.fake_fraction_end, total_epochs)
    cur_counterfactual_fraction = _linear_anneal(
        epoch, float(ga.counterfactual_fraction), ga.counterfactual_fraction_end, total_epochs)

    g_update_interval = max(1, int(ga.g_update_interval))
    ema_m = float(ga.ema_m)
    skipped_nan = 0
    skipped_oom = 0
    losses, losses_real, losses_fake = [], [], []
    grad_norms: List[float] = []
    g_losses: Dict[str, List[float]] = {
        "loss_g": [], "loss_g_recon": [], "loss_dsr_mean": [],
        "loss_dsr_var": [], "loss_dsr_floor": [],
    }
    n_fake_used_total = 0.0
    n_fake_counterfactual_used_total = 0.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    max_batches = int(cls_cfg.train.max_train_batches or 0)
    n_batches = min(max_batches, len(loader)) if max_batches > 0 else len(loader)

    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        flat_patches, flat_labels, sizes = flatten_bags(batch["patches"], batch["patch_labels"])
        flat_patches = flat_patches.to(device, non_blocking=True)
        flat_labels = flat_labels.to(device, non_blocking=True)
        img_label_per_patch = per_patch_image_label(batch["image_label"].to(device), sizes)

        fakes = None
        if fake_enabled:
            fakes = fake_fx_paste_flat(
                gan, flat_patches, flat_labels, img_label_per_patch, ga,
                fraction=cur_fraction,
                counterfactual_fraction=cur_counterfactual_fraction,
                requires_grad=False,
            )
            n_fake_used_total += float(fakes["stats"]["fake_used"])
            n_fake_counterfactual_used_total += float(
                fakes["stats"].get("fake_counterfactual_pos_used", 0.0))
            if not active_g_enabled:
                for k in ("_sel_region_unnorm", "_fake_region_unnorm", "_paste_mask", "_sel_idx"):
                    fakes.pop(k, None)

        opt.zero_grad(set_to_none=True)
        try:
            with autocast(enabled=cls_cfg.train.amp and device.type == "cuda"):
                out = student(flat_patches)
                loss_real = patch_bce(out["patch_logits"], flat_labels, w_fg=w_fg, w_bg=w_bg)
                if fakes is not None and fakes["fake_mask"].any():
                    aux_out = student(fakes["aux_patches"])
                    fake_mask = fakes["fake_mask"]
                    loss_fake = patch_bce(
                        aux_out["patch_logits"][fake_mask],
                        fakes["aux_labels"][fake_mask],
                        w_fg=w_fg, w_bg=w_bg,
                    )
                else:
                    loss_fake = out["patch_logits"].new_tensor(0.0)
                loss = loss_real + float(ga.alpha) * loss_fake
        except torch.cuda.OutOfMemoryError:
            skipped_oom += 1
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f"\n[{run_name}] CUDA OOM at ep{epoch} bi{bi + 1} — skip", flush=True)
            continue

        if not torch.isfinite(loss):
            skipped_nan += 1
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue

        scaler.scale(loss).backward()
        if cls_cfg.train.grad_clip and cls_cfg.train.grad_clip > 0:
            scaler.unscale_(opt)
            total_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), cls_cfg.train.grad_clip)
            gn = float(total_norm)
            if np.isfinite(gn):
                grad_norms.append(gn)
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        optimizer_was_run = scaler.get_scale() >= scale_before
        if scheduler is not None and optimizer_was_run:
            scheduler.step()
        if teacher is not None:
            ema_update_teacher(teacher, student, ema_m)

        if active_g_enabled and g_opt is not None and (bi % g_update_interval == 0):
            g_stats = cooperative_g_update(
                teacher=teacher, gan=gan, g_optimizer=g_opt, g_scaler=g_scaler,
                patches=flat_patches, patch_labels=flat_labels,
                image_label_per_patch=img_label_per_patch,
                ga_cfg=ga,
                amp_enabled=cls_cfg.train.amp and device.type == "cuda",
                counterfactual_fraction=cur_counterfactual_fraction,
            )
            for k in g_losses:
                v = g_stats.get(k)
                if v is not None:
                    g_losses[k].append(float(v.detach().item()) if isinstance(v, torch.Tensor) else float(v))

        losses.append(float(loss.detach().item()))
        losses_real.append(float(loss_real.detach().item()))
        losses_fake.append(float(loss_fake.detach().item()))

        live_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"\r[{run_name}] [{epoch:03d}/{total_epochs:03d}] {bi + 1:04d}/{n_batches:04d} | "
              f"loss={live_loss:.3f}", end="", flush=True)
    print("", flush=True)

    out_stats = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "loss_real": float(np.mean(losses_real)) if losses_real else float("nan"),
        "loss_fake": float(np.mean(losses_fake)) if losses_fake else float("nan"),
        "epoch_seconds": time.time() - t0,
        "w_fg": w_fg, "w_bg": w_bg,
        "skipped_nan": float(skipped_nan),
        "skipped_oom": float(skipped_oom),
        "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else float("nan"),
        "fake_used_total": n_fake_used_total,
        "fake_counterfactual_used_total": n_fake_counterfactual_used_total,
        "fake_fraction": cur_fraction,
        "counterfactual_fraction": cur_counterfactual_fraction,
    }
    for k, vs in g_losses.items():
        out_stats[k] = float(np.mean(vs)) if vs else float("nan")
    if device.type == "cuda":
        out_stats["cuda_peak_alloc_gb"] = float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))
    return out_stats


def run(cfg, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls_cfg = cfg.classifier
    seed_all(cls_cfg.train.seed)

    train_items = build_items(cfg.datasets.train)
    valid_items = build_items(cfg.datasets.valid)
    print(f"[v5] train={len(train_items)} valid({cfg.datasets.valid.name})={len(valid_items)}",
          flush=True)

    train_loader = make_loader(train_items, cls_cfg, True)
    valid_loader = make_loader(valid_items, cls_cfg, False)

    student = build_patch_classifier(cls_cfg).to(device)

    # EMA teacher: feature encoder for G feedback AND the eval/checkpoint
    # network (smooths noisy single-batch updates of the fast student).
    teacher = build_ema_teacher(student)
    use_ema_for_eval = bool(cls_cfg.gan_augmentation.ema_student_for_eval)

    gan = None
    g_opt = g_scaler = None
    ga = cls_cfg.gan_augmentation
    gan_ckpt = Path(args.gan_checkpoint) if args.gan_checkpoint else cfg.gan_checkpoint_path()
    if bool(ga.use_fake):
        gan = PretrainedInpaintingGan(
            checkpoint_path=gan_ckpt,
            lama_arch=cfg.lama,
            device=device,
            active_g=bool(ga.active_g),
        )
        print(f"[v5] loaded pretrained G: {gan_ckpt} active_g={bool(ga.active_g)}", flush=True)
        if bool(ga.active_g):
            g_opt = torch.optim.AdamW(gan.G.parameters(), lr=float(ga.g_lr),
                                      betas=(0.0, 0.99), weight_decay=0.0)
            g_scaler = GradScaler(enabled=cls_cfg.train.amp and device.type == "cuda")

    opt = torch.optim.AdamW(student.parameters(), lr=cls_cfg.train.lr,
                            weight_decay=cls_cfg.train.weight_decay)
    scaler = GradScaler(enabled=cls_cfg.train.amp and device.type == "cuda")
    total_steps = cls_cfg.train.epochs * max(1, len(train_loader))
    warmup_steps = max(1, cls_cfg.train.warmup_epochs) * max(1, len(train_loader))
    scheduler = make_warmup_cosine_scheduler(opt, total_steps, warmup_steps)

    out_dir = Path(args.output_dir or cfg.step3_output_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_path = out_dir / "metrics.csv"
    best_auc = -1.0
    best_epoch = 0
    prev_valid = None
    saved_columns: List[str] = []

    eval_net = teacher if use_ema_for_eval else student
    print(f"[v5] eval target = {'EMA teacher' if use_ema_for_eval else 'student'}", flush=True)

    for epoch in range(1, cls_cfg.train.epochs + 1):
        w_fg, w_bg = dynamic_loss_weights(prev_valid, cls_cfg)
        train_m = train_epoch(
            student, teacher, gan, train_loader, opt, scaler,
            g_opt, g_scaler, cls_cfg, device, epoch, cls_cfg.train.epochs,
            w_fg=w_fg, w_bg=w_bg, scheduler=scheduler,
            run_name=f"v5/{out_dir.name}",
        )
        print(f"[v5] eval epoch {epoch:03d} ({cfg.datasets.valid.name} valid)...", flush=True)
        valid_m = evaluate(eval_net, valid_loader, cls_cfg, device)
        valid_auc = float(valid_m.get("image_topk_auc", float("nan")))

        flat_m = {**{f"train_{k}": v for k, v in train_m.items()},
                  **{f"valid_{k}": v for k, v in valid_m.items()}}
        if not saved_columns:
            saved_columns = list(flat_m.keys())
            with metric_path.open("w", newline="") as f:
                csv.writer(f).writerow(["epoch"] + saved_columns)
        with metric_path.open("a", newline="") as f:
            csv.writer(f).writerow([epoch] + [fmt(flat_m.get(c, float("nan"))) for c in saved_columns])

        improved = valid_auc == valid_auc and valid_auc > best_auc
        if improved:
            best_auc = valid_auc
            best_epoch = epoch
            _save_ckpt(eval_net, epoch, out_dir / "best_auc.pt")
        _save_ckpt(student, epoch, out_dir / "latest.pt")

        sen = valid_m.get("image_topk_youden_best_sen", float("nan"))
        spec = valid_m.get("image_topk_youden_best_spec", float("nan"))
        print(f"[v5] ep{epoch:03d} | loss={train_m.get('loss', float('nan')):.3f} | "
              f"valid auc={valid_auc:.3f} sen={sen:.3f} spec={spec:.3f}"
              f"{' (NEW BEST)' if improved else ''}", flush=True)
        prev_valid = valid_m

    print(f"[v5] done. best valid AUC={best_auc:.4f} @ ep{best_epoch} -> {out_dir / 'best_auc.pt'}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--output-dir", default=None,
                    help="Default: <results_root>/step3_classifier")
    ap.add_argument("--gan-checkpoint", default=None,
                    help="Step-2 G_latest.pt path (default: config value).")
    ap.add_argument("--max-train-batches", type=int, default=None,
                    help="Cap training batches per epoch (smoke test).")
    ap.add_argument("--max-eval-batches", type=int, default=None,
                    help="Cap validation batches per epoch (smoke test).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    t = cfg.classifier.train
    if args.epochs is not None:
        t.epochs = args.epochs
    if args.lr is not None:
        t.lr = args.lr
    if args.batch_size is not None:
        t.batch_size = args.batch_size
    if args.num_workers is not None:
        t.num_workers = args.num_workers
    if args.max_train_batches is not None:
        t.max_train_batches = args.max_train_batches
    if args.max_eval_batches is not None:
        t.max_eval_batches = args.max_eval_batches

    load_site_stats(cfg.paths.site_stats)
    ga = cfg.classifier.gan_augmentation
    print(f"[v5] config: encoder={cfg.classifier.model.encoder} "
          f"active_g={ga.active_g} frac={ga.fake_fraction}->{ga.fake_fraction_end} "
          f"cf={ga.counterfactual_fraction}->{ga.counterfactual_fraction_end} "
          f"alpha={ga.alpha} epochs={t.epochs}", flush=True)
    run(cfg, args)


if __name__ == "__main__":
    main()
