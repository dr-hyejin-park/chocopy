"""Evaluation metrics for imputation quality."""
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
def rmse(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def mae(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(true - pred)))


def accuracy(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(true == pred))


def wasserstein1d(a: np.ndarray, b: np.ndarray) -> float:
    """1-D Wasserstein distance between two samples."""
    a_s = np.sort(a)
    b_s = np.sort(b)
    n = min(len(a_s), len(b_s))
    return float(np.mean(np.abs(a_s[:n] - b_s[:n])))


def js_divergence_cat(true_labels: np.ndarray, pred_labels: np.ndarray, n_classes: int) -> float:
    """Jensen-Shannon divergence for categorical predictions."""
    eps = 1e-10
    p = np.bincount(true_labels, minlength=n_classes).astype(float) + eps
    q = np.bincount(pred_labels, minlength=n_classes).astype(float) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


# ─────────────────────────────────────────────────────────────────────────────
def evaluate_imputation(
    x_num_true: np.ndarray,
    x_cat_true: np.ndarray,
    x_num_imputed: np.ndarray,
    x_cat_imputed: np.ndarray,
    mask_num: np.ndarray,
    mask_cat: np.ndarray,
    cat_vocab_sizes: List[int],
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> Dict:
    """
    Compute per-column and aggregate imputation metrics.

    Only evaluates at masked (imputed) positions.
    """
    results: Dict = {
        "numeric": {},
        "categorical": {},
        "summary": {},
    }

    # ── Numeric metrics ───────────────────────────────────────────────────────
    num_rmses, num_maes, num_ws = [], [], []

    for i in range(x_num_true.shape[1]):
        mask_i = mask_num[:, i]
        if mask_i.sum() == 0:
            continue
        true_i = x_num_true[mask_i, i]
        pred_i = x_num_imputed[mask_i, i]
        r = rmse(true_i, pred_i)
        m = mae(true_i, pred_i)
        w = wasserstein1d(true_i, pred_i)
        col_name = numeric_cols[i] if numeric_cols else f"num_{i}"
        results["numeric"][col_name] = {"rmse": r, "mae": m, "wasserstein": w, "n_masked": int(mask_i.sum())}
        num_rmses.append(r)
        num_maes.append(m)
        num_ws.append(w)

    # ── Categorical metrics ───────────────────────────────────────────────────
    cat_accs, cat_js = [], []

    for i, C in enumerate(cat_vocab_sizes):
        mask_i = mask_cat[:, i]
        if mask_i.sum() == 0:
            continue
        true_i = x_cat_true[mask_i, i]
        pred_i = np.clip(x_cat_imputed[mask_i, i], 0, C - 1)
        acc = accuracy(true_i, pred_i)
        js = js_divergence_cat(true_i, pred_i, C)
        col_name = categorical_cols[i] if categorical_cols else f"cat_{i}"
        results["categorical"][col_name] = {"accuracy": acc, "js_div": js, "n_masked": int(mask_i.sum())}
        cat_accs.append(acc)
        cat_js.append(js)

    results["summary"] = {
        "mean_rmse_num": float(np.mean(num_rmses)) if num_rmses else None,
        "mean_mae_num": float(np.mean(num_maes)) if num_maes else None,
        "mean_wasserstein_num": float(np.mean(num_ws)) if num_ws else None,
        "mean_accuracy_cat": float(np.mean(cat_accs)) if cat_accs else None,
        "mean_js_div_cat": float(np.mean(cat_js)) if cat_js else None,
    }

    return results


# ─────────────────────────────────────────────────────────────────────────────
def compare_results(
    results_tabdiff: Dict,
    results_ctgan: Dict,
    mask_ratio: float,
) -> str:
    """Format a human-readable comparison table."""
    lines = [
        f"\n{'='*65}",
        f"  Imputation Comparison  |  Mask ratio = {mask_ratio:.0%}",
        f"{'='*65}",
        f"{'Metric':<28}{'TabDiff':>14}{'CTGAN':>14}{'Winner':>8}",
        f"{'-'*65}",
    ]

    def row(label, td, cg, lower_better=True):
        if td is None or cg is None:
            return
        winner = "TabDiff" if (td < cg) == lower_better else "CTGAN"
        lines.append(f"  {label:<26}{td:>14.5f}{cg:>14.5f}{winner:>8}")

    s_td = results_tabdiff.get("summary", {})
    s_cg = results_ctgan.get("summary", {})

    row("RMSE (numeric)", s_td.get("mean_rmse_num"), s_cg.get("mean_rmse_num"))
    row("MAE  (numeric)", s_td.get("mean_mae_num"), s_cg.get("mean_mae_num"))
    row("Wasserstein (numeric)", s_td.get("mean_wasserstein_num"), s_cg.get("mean_wasserstein_num"))
    row("Accuracy (categorical)", s_td.get("mean_accuracy_cat"), s_cg.get("mean_accuracy_cat"), lower_better=False)
    row("JS Divergence (categorical)", s_td.get("mean_js_div_cat"), s_cg.get("mean_js_div_cat"))

    lines.append(f"{'='*65}\n")
    return "\n".join(lines)
