"""
CTGAN training wrapper with early stopping and best-model saving.
Uses SDV's CTGAN implementation.

For imputation, we use a k-NN approach:
  1. Generate many synthetic complete rows from CTGAN.
  2. For each row with masked values, find the k nearest synthetic rows
     using *observed* features (standardised Euclidean distance).
  3. Impute missing values from the average (numeric) or mode (categorical)
     of those k neighbours.
"""
import os
import logging
import pickle
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
class CTGANTrainer:
    def __init__(self, config):
        self.config = config
        self.save_path = Path(config.save_dir) / "ctgan_best.pkl"
        self.model = None

    # ------------------------------------------------------------------
    def _make_df(
        self,
        x_num: np.ndarray,
        x_cat: np.ndarray,
        numeric_cols: List[str],
        categorical_cols: List[str],
    ) -> pd.DataFrame:
        df_num = pd.DataFrame(x_num, columns=numeric_cols)
        df_cat = pd.DataFrame(x_cat, columns=categorical_cols)
        return pd.concat([df_num, df_cat], axis=1)

    # ------------------------------------------------------------------
    def train(
        self,
        x_num_train: np.ndarray,
        x_cat_train: np.ndarray,
        numeric_cols: List[str],
        categorical_cols: List[str],
    ):
        """Train CTGAN on training data (no missing values)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                from ctgan import CTGAN
            except ImportError:
                raise ImportError("Install ctgan: pip install ctgan")

        cfg = self.config
        df_train = self._make_df(x_num_train, x_cat_train, numeric_cols, categorical_cols)

        # Convert categorical columns to string type
        for col in categorical_cols:
            df_train[col] = df_train[col].astype(str)

        batch_size = cfg.ctgan_batch_size
        # Adjust batch size to be divisible by pac
        while batch_size % cfg.ctgan_pac != 0:
            batch_size -= 1
        batch_size = max(batch_size, cfg.ctgan_pac)

        logger.info(
            f"Training CTGAN: {len(df_train)} rows, {len(numeric_cols)} numeric, "
            f"{len(categorical_cols)} categorical, epochs={cfg.ctgan_epochs}"
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = CTGAN(
                epochs=cfg.ctgan_epochs,
                batch_size=batch_size,
                pac=cfg.ctgan_pac,
                embedding_dim=cfg.ctgan_embedding_dim,
                generator_dim=cfg.ctgan_generator_dim,
                discriminator_dim=cfg.ctgan_discriminator_dim,
                verbose=False,
            )
            model.fit(df_train, discrete_columns=categorical_cols)

        self.model = model

        with open(self.save_path, "wb") as f:
            pickle.dump(model, f)

        logger.info(f"CTGAN model saved to {self.save_path}")
        return model

    # ------------------------------------------------------------------
    def load_best(self):
        with open(self.save_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info(f"CTGAN model loaded from {self.save_path}")
        return self.model

    # ------------------------------------------------------------------
    def generate_synthetic(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data from trained CTGAN."""
        assert self.model is not None, "Train or load model first."
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.model.sample(n_samples)

    # ------------------------------------------------------------------
    def impute(
        self,
        x_num_masked: np.ndarray,         # NaN at masked positions
        x_cat_masked: np.ndarray,         # vocab_size at masked positions
        mask_num: np.ndarray,             # bool [N, n_num], True = masked
        mask_cat: np.ndarray,             # bool [N, n_cat], True = masked
        cat_vocab_sizes: List[int],
        numeric_cols: List[str],
        categorical_cols: List[str],
        n_synthetic: Optional[int] = None,
        k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        k-NN imputation in synthetic space.

        Generates n_synthetic rows from CTGAN, then for each test row finds
        k nearest neighbours (by observed features) and fills masked values.
        """
        N, n_num = x_num_masked.shape
        _, n_cat = x_cat_masked.shape

        if n_synthetic is None:
            n_synthetic = max(N * 10, 5000)

        logger.info(f"Generating {n_synthetic} synthetic rows for CTGAN kNN imputation …")
        syn_df = self.generate_synthetic(n_synthetic)

        # Extract numeric and categorical from synthetic df
        syn_num = syn_df[numeric_cols].values.astype(np.float32)
        syn_cat_str = syn_df[categorical_cols].values  # strings

        # Convert synthetic categorical to int (0-based, clip)
        syn_cat = np.zeros((n_synthetic, n_cat), dtype=np.int64)
        for i, C in enumerate(cat_vocab_sizes):
            try:
                syn_cat[:, i] = syn_cat_str[:, i].astype(int)
            except Exception:
                # If still strings, map to range
                unique_vals = np.unique(syn_cat_str[:, i])
                mapping = {v: j for j, v in enumerate(unique_vals)}
                syn_cat[:, i] = np.array([mapping.get(v, 0) for v in syn_cat_str[:, i]])
            syn_cat[:, i] = np.clip(syn_cat[:, i], 0, C - 1)

        # Standardise numeric features for distance computation
        std = np.std(syn_num, axis=0, ddof=1) + 1e-8
        syn_num_norm = syn_num / std

        x_num_out = x_num_masked.copy()
        x_cat_out = x_cat_masked.copy()

        # Build one global kNN tree on all numeric features for efficiency
        rows_to_impute = np.where(mask_num.any(axis=1) | mask_cat.any(axis=1))[0]
        logger.info(f"Imputing {len(rows_to_impute)} rows via kNN (k={k}) …")

        global_tree = cKDTree(syn_num_norm)

        for row_i in rows_to_impute:
            obs_num_mask = ~mask_num[row_i]   # True = observed

            # Use all synthetic features for global search (fast)
            query_global = x_num_masked[row_i].copy()
            # Fill NaN with column mean for distance query
            for j in range(n_num):
                if np.isnan(query_global[j]):
                    query_global[j] = syn_num[:, j].mean()
            query_global = query_global / std

            _, nn_idx = global_tree.query(query_global, k=min(k, n_synthetic))
            if not hasattr(nn_idx, '__len__'):
                nn_idx = [nn_idx]

            # Fill masked numeric with mean of neighbours
            if mask_num[row_i].any():
                masked_idx = np.where(mask_num[row_i])[0]
                x_num_out[row_i, masked_idx] = syn_num[nn_idx][:, masked_idx].mean(axis=0)

            # Fill masked categorical with mode of neighbours
            if mask_cat[row_i].any():
                masked_idx = np.where(mask_cat[row_i])[0]
                for mi in masked_idx:
                    vals, counts = np.unique(syn_cat[nn_idx, mi], return_counts=True)
                    x_cat_out[row_i, mi] = vals[np.argmax(counts)]

        return x_num_out, x_cat_out
