"""Configuration for TabDiff imputation pipeline."""
import os
import torch
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TabDiffConfig:
    # ── Data ──────────────────────────────────────────────────────────────────
    data_path: str = "data/real_data.csv"
    # Provide these lists explicitly; None = auto-detect by dtype
    numeric_cols: Optional[List[str]] = None
    categorical_cols: Optional[List[str]] = None
    test_size: float = 0.10
    val_size: float = 0.10
    random_seed: int = 42

    # ── Model ─────────────────────────────────────────────────────────────────
    d_embed_cat: int = 8          # embedding dim per categorical feature
    d_time: int = 128             # sinusoidal time embedding dim
    # Hidden layer sizes for the denoising MLP
    hidden_dims: List[int] = field(default_factory=lambda: [2048, 2048, 1024, 1024])
    dropout: float = 0.0          # usually 0 for diffusion models

    # ── Diffusion ─────────────────────────────────────────────────────────────
    num_timesteps: int = 1000
    num_schedule: str = "cosine"  # schedule type for continuous features
    cat_schedule: str = "cosine"  # schedule type for categorical features
    s_num: float = 0.008          # cosine offset for continuous
    s_cat: float = 0.008          # cosine offset for categorical

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size: int = 512
    max_epochs: int = 500
    lr: float = 3e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5        # linear LR warmup
    lr_min: float = 1e-6          # cosine annealing floor
    patience: int = 30            # early-stopping patience (epochs)
    min_delta: float = 1e-4       # min val-loss improvement to reset counter
    grad_clip: float = 1.0        # gradient norm clip
    use_amp: bool = True          # automatic mixed precision (fp16)
    num_workers: int = 4

    # ── Loss weights ──────────────────────────────────────────────────────────
    lambda_num: float = 1.0       # weight for continuous MSE loss
    lambda_cat: float = 1.0       # weight for categorical CE loss

    # ── CTGAN ─────────────────────────────────────────────────────────────────
    ctgan_epochs: int = 300
    ctgan_batch_size: int = 500
    ctgan_pac: int = 10           # PAC-GAN grouping (must divide batch_size)
    ctgan_embedding_dim: int = 128
    ctgan_generator_dim: tuple = (256, 256)
    ctgan_discriminator_dim: tuple = (256, 256)

    # ── Imputation ────────────────────────────────────────────────────────────
    mask_ratios: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    n_ddim_steps: int = 100       # DDIM sampling steps (≪ T for speed)
    n_resample: int = 3           # Repaint resampling rounds per DDIM step
    ctgan_nn_k: int = 5           # k for CTGAN kNN imputation

    # ── Paths ─────────────────────────────────────────────────────────────────
    save_dir: str = "checkpoints"
    plot_dir: str = "plots"
    results_dir: str = "results"

    # ── Visualization ─────────────────────────────────────────────────────────
    max_plot_features: int = 30   # cap number of per-feature plots
    plot_dpi: int = 150
    fig_width: int = 6
    fig_height: int = 4

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        for d in [self.save_dir, self.plot_dir, self.results_dir]:
            os.makedirs(d, exist_ok=True)
