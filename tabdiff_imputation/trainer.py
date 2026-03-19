"""
TabDiff training loop.

Features:
- Linear LR warmup + cosine annealing decay
- Gradient norm clipping
- AMP (fp16) mixed precision
- Best-model checkpoint saving
- Early stopping
- Per-epoch validation loss logging
"""
import copy
import os
import math
import time
import logging
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .model import TabDiffDenoiser
from .diffusion import TabDiffusion

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
class CosineWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup + cosine decay LR scheduler."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, lr_min: float = 1e-6):
        self.warmup_steps = max(warmup_steps, 1)
        self.total_steps = total_steps
        self.lr_min = lr_min
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                lr = base_lr * step / self.warmup_steps
            else:
                progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
                lr = self.lr_min + 0.5 * (base_lr - self.lr_min) * (1 + math.cos(math.pi * progress))
            lrs.append(lr)
        return lrs


# ─────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    """Stop training when val loss stops improving."""

    def __init__(self, patience: int = 30, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ─────────────────────────────────────────────────────────────────────────────
class TabDiffTrainer:
    """Full training pipeline for TabDiff."""

    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.save_path = Path(config.save_dir) / "tabdiff_best.pt"

    # ------------------------------------------------------------------
    def build_model(self, num_numeric: int, cat_vocab_sizes: List[int]) -> TabDiffDenoiser:
        model = TabDiffDenoiser(
            num_numeric=num_numeric,
            cat_vocab_sizes=cat_vocab_sizes,
            d_embed_cat=self.config.d_embed_cat,
            d_time=self.config.d_time,
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
        )
        return model.to(self.device)

    # ------------------------------------------------------------------
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_numeric: int,
        cat_vocab_sizes: List[int],
        model: Optional[TabDiffDenoiser] = None,
    ) -> TabDiffDenoiser:

        cfg = self.config

        if model is None:
            model = self.build_model(num_numeric, cat_vocab_sizes)

        # EMA model — used for inference; shadow-copies the training model
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        ema_decay = 0.9999

        diffusion = TabDiffusion(cfg).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        total_steps = cfg.max_epochs * len(train_loader)
        warmup_steps = cfg.warmup_epochs * len(train_loader)
        scheduler = CosineWarmupLR(optimizer, warmup_steps, total_steps, cfg.lr_min)

        scaler = GradScaler(enabled=cfg.use_amp and self.device == "cuda")
        early_stop = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)

        best_val_loss = float("inf")
        train_losses, val_losses = [], []

        logger.info(f"Training TabDiff: {num_numeric} numeric, {len(cat_vocab_sizes)} categorical")
        logger.info(f"  Device: {self.device} | Epochs: {cfg.max_epochs} | Batch: {cfg.batch_size}")

        for epoch in range(1, cfg.max_epochs + 1):
            # ── Train ──────────────────────────────────────────────────────────
            model.train()
            ep_loss = ep_num = ep_cat = 0.0
            n_batches = 0

            for x_num, x_cat in train_loader:
                x_num = x_num.to(self.device, non_blocking=True)
                x_cat = x_cat.to(self.device, non_blocking=True)
                B = x_num.shape[0]

                t = torch.randint(0, cfg.num_timesteps, (B,), device=self.device)

                with autocast(enabled=cfg.use_amp and self.device == "cuda"):
                    # Forward diffusion
                    x_t_num, noise = diffusion.q_sample_num(x_num, t)
                    x_t_cat, keep = diffusion.q_sample_cat(x_cat, t, cat_vocab_sizes)

                    # Denoising network
                    noise_pred, logits_cat = model(x_t_num, x_t_cat, t)

                    # Loss
                    loss, loss_n, loss_c = diffusion.compute_loss(
                        noise_pred, noise,
                        logits_cat, x_cat, keep,
                        lambda_num=cfg.lambda_num,
                        lambda_cat=cfg.lambda_cat,
                    )

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()

                # Gradient clipping (unscale first for AMP)
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                # EMA update
                with torch.no_grad():
                    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                        ema_p.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

                ep_loss += loss.item()
                ep_num += loss_n.item()
                ep_cat += loss_c.item()
                n_batches += 1

            train_loss = ep_loss / n_batches
            train_losses.append(train_loss)

            # ── Validation (using EMA model for realistic eval quality) ────────
            ema_model.eval()
            vl = vn = vc = 0.0
            vn_batches = 0
            with torch.no_grad():
                for x_num, x_cat in val_loader:
                    x_num = x_num.to(self.device, non_blocking=True)
                    x_cat = x_cat.to(self.device, non_blocking=True)
                    B = x_num.shape[0]
                    t = torch.randint(0, cfg.num_timesteps, (B,), device=self.device)
                    with autocast(enabled=cfg.use_amp and self.device == "cuda"):
                        x_t_num, noise = diffusion.q_sample_num(x_num, t)
                        x_t_cat, keep = diffusion.q_sample_cat(x_cat, t, cat_vocab_sizes)
                        noise_pred, logits_cat = ema_model(x_t_num, x_t_cat, t)
                        loss, loss_n, loss_c = diffusion.compute_loss(
                            noise_pred, noise, logits_cat, x_cat, keep,
                            lambda_num=cfg.lambda_num, lambda_cat=cfg.lambda_cat,
                        )
                    vl += loss.item()
                    vn += loss_n.item()
                    vc += loss_c.item()
                    vn_batches += 1

            val_loss = vl / vn_batches
            val_losses.append(val_loss)
            lr_now = scheduler.get_last_lr()[0]

            logger.info(
                f"Epoch {epoch:4d}/{cfg.max_epochs} | "
                f"LR={lr_now:.2e} | "
                f"Train {train_loss:.4f} (num={ep_num/n_batches:.4f}, cat={ep_cat/n_batches:.4f}) | "
                f"Val {val_loss:.4f} (num={vn/vn_batches:.4f}, cat={vc/vn_batches:.4f})"
            )

            # ── Save best ─────────────────────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "ema_model_state": ema_model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "num_numeric": num_numeric,
                        "cat_vocab_sizes": cat_vocab_sizes,
                        "config": cfg,
                    },
                    self.save_path,
                )
                logger.info(f"  ✓ Best model saved  (val_loss={best_val_loss:.4f})")

            # ── Early stopping ────────────────────────────────────────────────
            if early_stop.step(val_loss):
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {cfg.patience} epochs)"
                )
                break

        # Reload best weights (prefer EMA weights for inference)
        ckpt = torch.load(self.save_path, map_location=self.device, weights_only=False)
        ema_model.load_state_dict(ckpt.get("ema_model_state", ckpt["model_state"]))
        ema_model.eval()
        logger.info(f"Best EMA model reloaded (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

        return ema_model, {"train_losses": train_losses, "val_losses": val_losses}

    # ------------------------------------------------------------------
    def load_best(self, num_numeric: int, cat_vocab_sizes: List[int]) -> TabDiffDenoiser:
        ckpt = torch.load(self.save_path, map_location=self.device, weights_only=False)
        # Restore architecture from checkpoint config if available
        saved_cfg = ckpt.get("config", self.config)
        model = TabDiffDenoiser(
            num_numeric=ckpt.get("num_numeric", num_numeric),
            cat_vocab_sizes=ckpt.get("cat_vocab_sizes", cat_vocab_sizes),
            d_embed_cat=saved_cfg.d_embed_cat,
            d_time=saved_cfg.d_time,
            hidden_dims=saved_cfg.hidden_dims,
            dropout=saved_cfg.dropout,
        ).to(self.device)
        # Prefer EMA weights for inference
        model.load_state_dict(ckpt.get("ema_model_state", ckpt["model_state"]))
        model.eval()
        return model
