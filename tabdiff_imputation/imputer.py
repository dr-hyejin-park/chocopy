"""
Imputation with trained TabDiff (Repaint-style DDIM).

For each masking ratio (10%, 20%, 30%):
  1. Randomly mask test set values.
  2. Run DDIM reverse process conditioned on observed values.
  3. Replace masked positions with model outputs.
"""
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch

from .diffusion import TabDiffusion
from .model import TabDiffDenoiser

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
class TabDiffImputer:
    """Wraps a trained TabDiffDenoiser for imputation."""

    def __init__(self, model: TabDiffDenoiser, config, cat_vocab_sizes: List[int]):
        self.model = model
        self.config = config
        self.cat_vocab_sizes = cat_vocab_sizes
        self.device = config.device

        self.diffusion = TabDiffusion(config).to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    def impute(
        self,
        x_num_masked: np.ndarray,    # [N, n_num], NaN at masked
        x_cat_masked: np.ndarray,    # [N, n_cat], vocab_size at masked
        mask_num: np.ndarray,        # bool [N, n_num], True = masked
        mask_cat: np.ndarray,        # bool [N, n_cat], True = masked
        chunk_size: int = 128,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Repaint-DDIM imputation.
        Returns (x_num_imputed, x_cat_imputed) with same shape as inputs.
        """
        N, n_num = x_num_masked.shape
        _, n_cat = x_cat_masked.shape
        cfg = self.config

        # Replace NaN observed values with 0 (will be overwritten at masked positions)
        x_num_obs = np.nan_to_num(x_num_masked, nan=0.0).astype(np.float32)
        x_cat_obs = x_cat_masked.copy().astype(np.int64)
        # Clip MASK tokens to valid range (model MASK = C_i)
        for i, C in enumerate(self.cat_vocab_sizes):
            # Where cat is still MASK (= C), keep as-is; model handles it
            x_cat_obs[:, i] = np.clip(x_cat_obs[:, i], 0, C)

        # Observed masks (True = observed / known)
        obs_mask_num = ~mask_num        # [N, n_num] bool
        obs_mask_cat = ~mask_cat        # [N, n_cat] bool

        x_num_t = torch.tensor(x_num_obs, dtype=torch.float32)
        x_cat_t = torch.tensor(x_cat_obs, dtype=torch.long)
        obs_num_t = torch.tensor(obs_mask_num, dtype=torch.bool)
        obs_cat_t = torch.tensor(obs_mask_cat, dtype=torch.bool)

        logger.info(
            f"TabDiff imputing {N} rows | {cfg.n_ddim_steps} DDIM steps | "
            f"resample={cfg.n_resample} | chunk={chunk_size}"
        )

        num_out, cat_out = self.diffusion.ddim_sample(
            model=self.model,
            n_samples=N,
            num_numeric=n_num,
            cat_vocab_sizes=self.cat_vocab_sizes,
            n_steps=cfg.n_ddim_steps,
            eta=0.0,
            x_obs_num=x_num_t.to(self.device),
            x_obs_cat=x_cat_t.to(self.device),
            obs_mask_num=obs_num_t.to(self.device),
            obs_mask_cat=obs_cat_t.to(self.device),
            n_resample=cfg.n_resample,
            device=self.device,
            chunk_size=chunk_size,
        )

        num_arr = num_out.numpy()
        cat_arr = cat_out.numpy()

        # Restore observed values exactly
        num_arr[obs_mask_num] = x_num_obs[obs_mask_num]
        for i, C in enumerate(self.cat_vocab_sizes):
            observed_rows = obs_mask_cat[:, i]
            # For observed, restore original (non-MASK) value
            cat_arr[observed_rows, i] = x_cat_obs[observed_rows, i]
            # For imputed, clip to valid range
            cat_arr[~observed_rows, i] = np.clip(cat_arr[~observed_rows, i], 0, C - 1)

        return num_arr.astype(np.float32), cat_arr.astype(np.int64)
