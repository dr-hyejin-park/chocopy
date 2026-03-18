"""
Per-feature distribution comparison plots.

Handles Korean feature names by:
  1. Trying apt-installed Nanum fonts.
  2. Downloading NanumGothic if needed.
  3. Falling back to DejaVu Sans (boxes for Korean — but no crash).
"""
import io
import os
import logging
import subprocess
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")           # non-interactive backend — must set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
_KOREAN_FONTS = [
    "NanumGothic", "NanumBarunGothic", "NanumSquare",
    "Malgun Gothic", "AppleGothic", "Batang", "Dotum",
]


def _find_installed_korean_font() -> Optional[str]:
    for name in _KOREAN_FONTS:
        if any(name.lower() in f.name.lower() for f in fm.fontManager.ttflist):
            return name
    return None


def _try_install_nanum() -> Optional[str]:
    """Attempt `apt-get install fonts-nanum` and reload fontmanager."""
    try:
        subprocess.run(
            ["apt-get", "install", "-y", "-q", "fonts-nanum"],
            capture_output=True, timeout=60,
        )
        # Rebuild font cache
        try:
            fm.fontManager.__init__()
        except Exception:
            pass
        # Also try rebuilding with find_font path
        cache_file = fm.get_cachedir()
        for f in Path(cache_file).glob("fontlist*"):
            f.unlink(missing_ok=True)
        # Re-init
        importlib_reload_font_manager()
        return _find_installed_korean_font()
    except Exception as e:
        logger.debug(f"apt-get nanum install failed: {e}")
        return None


def importlib_reload_font_manager():
    try:
        import importlib
        importlib.reload(fm)
    except Exception:
        pass


def _try_download_nanum() -> Optional[str]:
    """Download NanumGothic-Regular.ttf from Google Fonts GitHub."""
    try:
        import urllib.request
        font_dir = Path.home() / ".local" / "share" / "fonts"
        font_dir.mkdir(parents=True, exist_ok=True)
        font_path = font_dir / "NanumGothic-Regular.ttf"
        if not font_path.exists():
            url = (
                "https://raw.githubusercontent.com/google/fonts/main/"
                "ofl/nanumgothic/NanumGothic-Regular.ttf"
            )
            urllib.request.urlretrieve(url, font_path)
        fm.fontManager.addfont(str(font_path))
        prop = fm.FontProperties(fname=str(font_path))
        return prop.get_name()
    except Exception as e:
        logger.debug(f"NanumGothic download failed: {e}")
        return None


def setup_korean_font() -> str:
    """Configure matplotlib to support Korean characters. Returns font name used."""
    # 1. Already installed?
    font = _find_installed_korean_font()
    if font is None:
        font = _try_install_nanum()
    if font is None:
        font = _try_download_nanum()
    if font is None:
        font = "DejaVu Sans"
        logger.warning(
            "Korean font not found; Korean labels may render as boxes. "
            "Install 'fonts-nanum' via apt-get for proper display."
        )

    plt.rcParams.update({
        "font.family": font,
        "axes.unicode_minus": False,
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })
    logger.info(f"Matplotlib font set to: {font}")
    return font


# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "original": "#2196F3",
    "tabdiff":  "#4CAF50",
    "ctgan":    "#FF9800",
}
LABELS = {
    "original": "원본 (Original)",
    "tabdiff":  "TabDiff",
    "ctgan":    "CTGAN",
}


# ─────────────────────────────────────────────────────────────────────────────
def _safe_label(name: str) -> str:
    """Truncate very long feature names for axis labels."""
    return name if len(name) <= 30 else name[:27] + "…"


def _kde_or_hist_num(
    ax: plt.Axes,
    data: np.ndarray,
    label: str,
    color: str,
    alpha: float = 0.55,
    bins: int = 40,
):
    """Draw KDE (if enough data) or histogram."""
    data = data[np.isfinite(data)]
    if len(data) < 5:
        return
    if len(np.unique(data)) >= 5:
        try:
            kde = stats.gaussian_kde(data, bw_method="silverman")
            xr = np.linspace(data.min(), data.max(), 300)
            ax.plot(xr, kde(xr), color=color, label=label, linewidth=1.8)
            ax.fill_between(xr, kde(xr), alpha=0.15, color=color)
            return
        except Exception:
            pass
    ax.hist(data, bins=bins, density=True, color=color, label=label, alpha=alpha)


def _bar_cat(
    ax: plt.Axes,
    true_labels: np.ndarray,
    pred_tabdiff: np.ndarray,
    pred_ctgan: np.ndarray,
    n_classes: int,
    col_name: str,
):
    """Grouped bar chart for categorical distributions."""
    classes = np.arange(n_classes)
    def freq(arr):
        c = np.bincount(np.clip(arr.astype(int), 0, n_classes - 1), minlength=n_classes)
        return c / c.sum() if c.sum() > 0 else c

    p_true = freq(true_labels)
    p_td = freq(pred_tabdiff)
    p_cg = freq(pred_ctgan)

    width = 0.26
    ax.bar(classes - width, p_true, width, label=LABELS["original"], color=COLORS["original"], alpha=0.8)
    ax.bar(classes,          p_td,   width, label=LABELS["tabdiff"],  color=COLORS["tabdiff"],  alpha=0.8)
    ax.bar(classes + width,  p_cg,   width, label=LABELS["ctgan"],    color=COLORS["ctgan"],    alpha=0.8)

    ax.set_xticks(classes)
    if n_classes <= 20:
        ax.set_xticklabels([str(c) for c in classes], rotation=45, ha="right")
    else:
        ax.set_xticklabels([])
    ax.set_ylabel("Frequency")


# ─────────────────────────────────────────────────────────────────────────────
def plot_numeric_features(
    x_num_true: np.ndarray,
    x_num_tabdiff: np.ndarray,
    x_num_ctgan: np.ndarray,
    mask_num: np.ndarray,
    numeric_cols: List[str],
    mask_ratio: float,
    plot_dir: str,
    max_features: int = 30,
    dpi: int = 150,
):
    """Plot per-feature KDE comparison for numeric columns (masked positions only)."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    n_cols = min(len(numeric_cols), max_features)
    n_rows_per_fig = 5
    n_cols_per_fig = 4
    features_per_fig = n_rows_per_fig * n_cols_per_fig

    figs_saved = []
    ratio_tag = f"{int(mask_ratio*100)}pct"

    for fig_start in range(0, n_cols, features_per_fig):
        feat_subset = list(range(fig_start, min(fig_start + features_per_fig, n_cols)))
        n_axes = len(feat_subset)
        ncols = min(n_cols_per_fig, n_axes)
        nrows = (n_axes + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.2))
        axes = np.array(axes).flatten() if n_axes > 1 else [axes]

        for ax_i, feat_i in enumerate(feat_subset):
            ax = axes[ax_i]
            col = numeric_cols[feat_i]
            mask_i = mask_num[:, feat_i]
            if mask_i.sum() == 0:
                ax.set_visible(False)
                continue

            _kde_or_hist_num(ax, x_num_true[mask_i, feat_i],    LABELS["original"], COLORS["original"])
            _kde_or_hist_num(ax, x_num_tabdiff[mask_i, feat_i], LABELS["tabdiff"],  COLORS["tabdiff"])
            _kde_or_hist_num(ax, x_num_ctgan[mask_i, feat_i],   LABELS["ctgan"],    COLORS["ctgan"])

            ax.set_title(_safe_label(col), pad=4)
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            if ax_i == 0:
                ax.legend(fontsize=7, framealpha=0.6)

        for ax in axes[len(feat_subset):]:
            ax.set_visible(False)

        fig.suptitle(
            f"수치형 변수 분포 비교 — 마스킹 비율 {int(mask_ratio*100)}%  "
            f"(Numeric Feature Distributions, mask={int(mask_ratio*100)}%)",
            fontsize=10, y=1.01,
        )
        plt.tight_layout()

        fig_idx = fig_start // features_per_fig + 1
        fname = plot_dir / f"numeric_{ratio_tag}_fig{fig_idx}.png"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        figs_saved.append(str(fname))
        logger.info(f"  Saved: {fname}")

    return figs_saved


def plot_categorical_features(
    x_cat_true: np.ndarray,
    x_cat_tabdiff: np.ndarray,
    x_cat_ctgan: np.ndarray,
    mask_cat: np.ndarray,
    cat_vocab_sizes: List[int],
    categorical_cols: List[str],
    mask_ratio: float,
    plot_dir: str,
    max_features: int = 30,
    dpi: int = 150,
):
    """Plot per-feature bar comparison for categorical columns (masked positions only)."""
    plot_dir = Path(plot_dir)
    n_cols = min(len(categorical_cols), max_features)
    ratio_tag = f"{int(mask_ratio*100)}pct"

    features_per_fig = 12
    figs_saved = []

    for fig_start in range(0, n_cols, features_per_fig):
        feat_subset = list(range(fig_start, min(fig_start + features_per_fig, n_cols)))
        n_axes = len(feat_subset)
        ncols = min(4, n_axes)
        nrows = (n_axes + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.2))
        axes = np.array(axes).flatten() if n_axes > 1 else [axes]

        for ax_i, feat_i in enumerate(feat_subset):
            ax = axes[ax_i]
            col = categorical_cols[feat_i]
            mask_i = mask_cat[:, feat_i]
            if mask_i.sum() == 0:
                ax.set_visible(False)
                continue
            C = cat_vocab_sizes[feat_i]
            _bar_cat(
                ax,
                x_cat_true[mask_i, feat_i],
                x_cat_tabdiff[mask_i, feat_i],
                x_cat_ctgan[mask_i, feat_i],
                C, col,
            )
            ax.set_title(_safe_label(col), pad=4)
            if ax_i == 0:
                ax.legend(fontsize=7, framealpha=0.6)

        for ax in axes[len(feat_subset):]:
            ax.set_visible(False)

        fig.suptitle(
            f"범주형 변수 분포 비교 — 마스킹 비율 {int(mask_ratio*100)}%  "
            f"(Categorical Feature Distributions, mask={int(mask_ratio*100)}%)",
            fontsize=10, y=1.01,
        )
        plt.tight_layout()

        fig_idx = fig_start // features_per_fig + 1
        fname = plot_dir / f"categorical_{ratio_tag}_fig{fig_idx}.png"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        figs_saved.append(str(fname))
        logger.info(f"  Saved: {fname}")

    return figs_saved


# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    plot_dir: str,
    title: str = "TabDiff Training Loss",
    dpi: int = 150,
):
    """Plot train vs val loss over epochs."""
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train", color="#2196F3", linewidth=1.5)
    ax.plot(epochs, val_losses,   label="Validation", color="#F44336", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    fname = Path(plot_dir) / "training_loss.png"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Training curve saved: {fname}")
    return str(fname)


def plot_metric_summary(
    results_by_ratio: Dict,          # {0.1: {tabdiff: {summary:...}, ctgan: {summary:...}}, ...}
    plot_dir: str,
    dpi: int = 150,
):
    """Bar chart comparing TabDiff vs CTGAN across masking ratios."""
    ratios = sorted(results_by_ratio.keys())
    metrics_num = ["mean_rmse_num", "mean_mae_num", "mean_wasserstein_num"]
    metrics_cat = ["mean_accuracy_cat", "mean_js_div_cat"]
    metric_labels = {
        "mean_rmse_num":        "RMSE (Numeric)",
        "mean_mae_num":         "MAE (Numeric)",
        "mean_wasserstein_num": "Wasserstein (Numeric)",
        "mean_accuracy_cat":    "Accuracy (Categorical) ↑",
        "mean_js_div_cat":      "JS Divergence (Categorical) ↓",
    }

    all_metrics = metrics_num + metrics_cat
    n_metrics = len(all_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 3.5, 4))

    for ax, metric in zip(axes, all_metrics):
        td_vals = [
            results_by_ratio[r]["tabdiff"]["summary"].get(metric)
            for r in ratios
        ]
        cg_vals = [
            results_by_ratio[r]["ctgan"]["summary"].get(metric)
            for r in ratios
        ]

        x = np.arange(len(ratios))
        width = 0.35
        ax.bar(x - width / 2, td_vals, width, label="TabDiff", color=COLORS["tabdiff"], alpha=0.85)
        ax.bar(x + width / 2, cg_vals, width, label="CTGAN",   color=COLORS["ctgan"],   alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(r*100)}%" for r in ratios])
        ax.set_title(metric_labels.get(metric, metric), fontsize=9)
        ax.set_xlabel("Mask Ratio")
        if ax is axes[0]:
            ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("TabDiff vs CTGAN 성능 비교 (Imputation Performance)", fontsize=11)
    plt.tight_layout()
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    fname = Path(plot_dir) / "metric_summary.png"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Metric summary saved: {fname}")
    return str(fname)
