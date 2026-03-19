"""
TabDiff Imputation Pipeline — Main Runner
==========================================

Usage
-----
  # 1. Generate sample data (if no real data available)
  python generate_sample_data.py

  # 2. Run full pipeline
  python run_pipeline.py

  # 3. Or with a real CSV
  python run_pipeline.py --data data/real_data.csv

Configuration
-------------
  Edit `tabdiff_imputation/config.py` or pass CLI overrides:
    --batch_size 256
    --max_epochs 300
    --ctgan_epochs 200
    --mask_ratios 0.1 0.2 0.3
"""
# ── suppress all third-party warnings ────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# ── logging setup ─────────────────────────────────────────────────────────────
Path("results").mkdir(parents=True, exist_ok=True)
Path("plots").mkdir(parents=True, exist_ok=True)
Path("checkpoints").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/pipeline.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# ── local imports ──────────────────────────────────────────────────────────────
from tabdiff_imputation.config import TabDiffConfig
from tabdiff_imputation.data_utils import (
    TabularPreprocessor, load_and_split, make_dataloaders, apply_masking,
)
from tabdiff_imputation.model import TabDiffDenoiser
from tabdiff_imputation.diffusion import TabDiffusion
from tabdiff_imputation.trainer import TabDiffTrainer
from tabdiff_imputation.ctgan_trainer import CTGANTrainer
from tabdiff_imputation.imputer import TabDiffImputer
from tabdiff_imputation.metrics import evaluate_imputation, compare_results
from tabdiff_imputation.visualize import (
    setup_korean_font,
    plot_numeric_features,
    plot_categorical_features,
    plot_training_curves,
    plot_metric_summary,
)


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="TabDiff Imputation Pipeline")
    p.add_argument("--data", default="data/real_data.csv")
    p.add_argument("--meta", default="data/column_meta.json",
                   help="JSON with numeric_cols/categorical_cols lists")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--ctgan_epochs", type=int, default=None)
    p.add_argument("--mask_ratios", type=float, nargs="+", default=None)
    p.add_argument("--skip_tabdiff_train", action="store_true",
                   help="Load existing TabDiff checkpoint (skip training)")
    p.add_argument("--skip_ctgan_train", action="store_true",
                   help="Load existing CTGAN checkpoint (skip training)")
    p.add_argument("--skip_imputation", action="store_true",
                   help="Skip imputation (only train models)")
    p.add_argument("--n_ddim_steps", type=int, default=None)
    p.add_argument("--chunk_size", type=int, default=128,
                   help="Rows per chunk during DDIM sampling (OOM guard)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def build_config(args) -> TabDiffConfig:
    cfg = TabDiffConfig()
    cfg.data_path = args.data
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.max_epochs:
        cfg.max_epochs = args.max_epochs
    if args.ctgan_epochs:
        cfg.ctgan_epochs = args.ctgan_epochs
    if args.mask_ratios:
        cfg.mask_ratios = args.mask_ratios
    if args.n_ddim_steps:
        cfg.n_ddim_steps = args.n_ddim_steps
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
def load_column_meta(meta_path: str, preprocessor: TabularPreprocessor):
    """Load pre-specified column lists from JSON if available."""
    if Path(meta_path).exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        preprocessor.numeric_cols = meta.get("numeric_cols")
        preprocessor.categorical_cols = meta.get("categorical_cols")
        logger.info(
            f"Column meta loaded: {len(preprocessor.numeric_cols or [])} numeric, "
            f"{len(preprocessor.categorical_cols or [])} categorical"
        )
    else:
        logger.info("No column_meta.json found – auto-detecting column types.")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    cfg = build_config(args)
    chunk_size = args.chunk_size

    Path("results").mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("  TabDiff Imputation Pipeline")
    logger.info(f"  Device : {cfg.device}")
    logger.info(f"  Data   : {cfg.data_path}")
    logger.info("=" * 60)

    # ── Setup Korean font ─────────────────────────────────────────────────────
    font_name = setup_korean_font()
    logger.info(f"Korean font: {font_name}")

    # ── Load & preprocess data ────────────────────────────────────────────────
    preprocessor = TabularPreprocessor()
    load_column_meta(args.meta, preprocessor)

    (x_num_train, x_cat_train), (x_num_val, x_cat_val), (x_num_test, x_cat_test) = \
        load_and_split(cfg.data_path, preprocessor,
                       val_size=cfg.val_size, test_size=cfg.test_size,
                       random_seed=cfg.random_seed)

    num_numeric = x_num_train.shape[1]
    cat_vocab_sizes = preprocessor.cat_vocab_sizes
    numeric_cols = preprocessor.numeric_cols
    categorical_cols = preprocessor.categorical_cols

    # Raw (original-scale) numeric arrays — used for CTGAN and final evaluation.
    # preprocessor.transform() now applies QuantileTransformer, so we invert here.
    x_num_train_raw = preprocessor.inverse_transform_num(x_num_train).values.astype(np.float32)
    x_num_test_raw  = preprocessor.inverse_transform_num(x_num_test).values.astype(np.float32)

    logger.info(
        f"Data split → Train: {len(x_num_train)}, Val: {len(x_num_val)}, Test: {len(x_num_test)}"
    )
    logger.info(f"Numeric: {num_numeric}, Categorical: {len(cat_vocab_sizes)}")

    # ── Build DataLoaders ─────────────────────────────────────────────────────
    train_loader, val_loader = make_dataloaders(
        x_num_train, x_cat_train, x_num_val, x_cat_val,
        batch_size=cfg.batch_size, num_workers=0,  # 0 for portability
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 1.  Train / load TabDiff
    # ══════════════════════════════════════════════════════════════════════════
    tabdiff_trainer = TabDiffTrainer(cfg)

    if args.skip_tabdiff_train and Path(cfg.save_dir, "tabdiff_best.pt").exists():
        logger.info("Loading existing TabDiff checkpoint …")
        model = tabdiff_trainer.load_best(num_numeric, cat_vocab_sizes)
        train_history = None
    else:
        logger.info("Training TabDiff …")
        t0 = time.time()
        model, train_history = tabdiff_trainer.train(
            train_loader, val_loader, num_numeric, cat_vocab_sizes
        )
        logger.info(f"TabDiff training done in {(time.time()-t0)/60:.1f} min")

    if train_history is not None:
        plot_training_curves(
            train_history["train_losses"],
            train_history["val_losses"],
            cfg.plot_dir,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 2.  Train / load CTGAN
    # ══════════════════════════════════════════════════════════════════════════
    ctgan_trainer = CTGANTrainer(cfg)

    if args.skip_ctgan_train and Path(cfg.save_dir, "ctgan_best.pkl").exists():
        logger.info("Loading existing CTGAN checkpoint …")
        ctgan_trainer.load_best()
    else:
        logger.info("Training CTGAN …")
        t0 = time.time()
        # CTGAN receives raw-scale data (it applies its own VGM normalisation)
        ctgan_trainer.train(
            x_num_train_raw, x_cat_train, numeric_cols, categorical_cols
        )
        logger.info(f"CTGAN training done in {(time.time()-t0)/60:.1f} min")

    if args.skip_imputation:
        logger.info("--skip_imputation flag set. Done.")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # 3.  Imputation & evaluation per masking ratio
    # ══════════════════════════════════════════════════════════════════════════
    tabdiff_imputer = TabDiffImputer(model, cfg, cat_vocab_sizes)
    rng = np.random.RandomState(cfg.random_seed)

    results_by_ratio: Dict = {}

    for mask_ratio in cfg.mask_ratios:
        ratio_pct = int(mask_ratio * 100)
        logger.info(f"\n{'─'*50}")
        logger.info(f"  Mask ratio: {ratio_pct}%")
        logger.info(f"{'─'*50}")

        # ── Apply masking ─────────────────────────────────────────────────────
        # Masking is applied to *normalised* test data (x_num_test) for TabDiff,
        # and to raw test data for CTGAN.  Both use the same boolean mask so the
        # evaluation targets the same positions.
        x_num_masked, x_cat_masked, mask_num, mask_cat = apply_masking(
            x_num_test, x_cat_test, mask_ratio, cat_vocab_sizes, rng=rng
        )
        # Raw masked version (same mask, raw scale) for CTGAN
        x_num_masked_raw = x_num_test_raw.copy()
        x_num_masked_raw[mask_num] = np.nan

        # ── TabDiff imputation (normalised space) ─────────────────────────────
        logger.info("Running TabDiff imputation …")
        t0 = time.time()
        x_num_td_norm, x_cat_td = tabdiff_imputer.impute(
            x_num_masked, x_cat_masked, mask_num, mask_cat,
            chunk_size=chunk_size,
        )
        logger.info(f"  TabDiff imputation: {time.time()-t0:.1f}s")

        # Inverse-transform TabDiff output to raw scale for evaluation
        x_num_td = preprocessor.inverse_transform_num(x_num_td_norm).values.astype(np.float32)
        # Restore observed values exactly (raw scale)
        x_num_td[~mask_num] = x_num_test_raw[~mask_num]

        # ── CTGAN imputation (raw scale) ──────────────────────────────────────
        logger.info("Running CTGAN imputation …")
        t0 = time.time()
        x_num_cg, x_cat_cg = ctgan_trainer.impute(
            x_num_masked_raw, x_cat_masked, mask_num, mask_cat,
            cat_vocab_sizes, numeric_cols, categorical_cols,
            n_synthetic=max(len(x_num_test) * 10, 2000),
            k=cfg.ctgan_nn_k,
        )
        logger.info(f"  CTGAN imputation: {time.time()-t0:.1f}s")

        # ── Evaluate both in raw scale ────────────────────────────────────────
        res_td = evaluate_imputation(
            x_num_test_raw, x_cat_test,
            x_num_td, x_cat_td,
            mask_num, mask_cat,
            cat_vocab_sizes, numeric_cols, categorical_cols,
        )
        res_cg = evaluate_imputation(
            x_num_test_raw, x_cat_test,
            x_num_cg, x_cat_cg,
            mask_num, mask_cat,
            cat_vocab_sizes, numeric_cols, categorical_cols,
        )

        results_by_ratio[mask_ratio] = {"tabdiff": res_td, "ctgan": res_cg}

        # Print comparison table
        print(compare_results(res_td, res_cg, mask_ratio))

        # Save results JSON
        res_path = Path(cfg.results_dir) / f"results_{ratio_pct}pct.json"
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump({"tabdiff": res_td, "ctgan": res_cg}, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved: {res_path}")

        # ── Visualise ─────────────────────────────────────────────────────────
        logger.info("Generating per-feature plots …")

        plot_numeric_features(
            x_num_test_raw, x_num_td, x_num_cg,
            mask_num, numeric_cols, mask_ratio,
            cfg.plot_dir,
            max_features=cfg.max_plot_features,
            dpi=cfg.plot_dpi,
        )
        plot_categorical_features(
            x_cat_test, x_cat_td, x_cat_cg,
            mask_cat, cat_vocab_sizes, categorical_cols, mask_ratio,
            cfg.plot_dir,
            max_features=cfg.max_plot_features,
            dpi=cfg.plot_dpi,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 4.  Summary metric bar charts across all ratios
    # ══════════════════════════════════════════════════════════════════════════
    if len(results_by_ratio) > 0:
        plot_metric_summary(results_by_ratio, cfg.plot_dir, dpi=cfg.plot_dpi)

    logger.info("\nPipeline complete. Plots → %s | Results → %s", cfg.plot_dir, cfg.results_dir)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
