"""
Generate synthetic Hyundai Card-style data for testing.

Mimics real data structure:
  - 1800 numeric features (quantile-transformed → ~N(0,1))
  - 60 categorical features (2–15 classes each, Korean-style names)
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

N_SAMPLES = 3000
N_NUMERIC  = 50   # reduce to 50 for quick demo; change to 1800 for full-scale
N_CAT      = 10   # reduce to 10; change to 60 for full-scale

# ── Korean-style feature names ────────────────────────────────────────────────
NUMERIC_PREFIXES = [
    "총구매금액", "월평균이용금액", "최근이용금액", "카드이용횟수", "포인트잔액",
    "할부이용금액", "해외이용금액", "온라인이용금액", "오프라인이용금액", "현금서비스금액",
]
CAT_NAMES = [
    "카드등급", "직업코드", "거주지역", "성별", "혼인여부",
    "주거유형", "결제방법", "이용업종", "우대혜택유형", "채널구분",
]


def make_korean_num_cols(n: int) -> list:
    cols = []
    for i in range(n):
        prefix = NUMERIC_PREFIXES[i % len(NUMERIC_PREFIXES)]
        cols.append(f"{prefix}_{i:04d}")
    return cols


def make_korean_cat_cols(n: int) -> list:
    return [CAT_NAMES[i % len(CAT_NAMES)] + f"_{i:02d}" for i in range(n)]


# ── Generate correlated numeric features ──────────────────────────────────────
def generate_numeric(n_samples: int, n_features: int) -> np.ndarray:
    # Create block-correlated structure (realistic for card data)
    block_size = max(1, n_features // 5)
    data = []
    for b in range(0, n_features, block_size):
        sz = min(block_size, n_features - b)
        # Shared latent factor
        factor = np.random.randn(n_samples)
        block = np.outer(factor, np.random.randn(sz) * 0.5) + np.random.randn(n_samples, sz)
        # Quantile-transform approximation: already ~N(0,1) after randn
        data.append(block)
    return np.hstack(data)[:, :n_features].astype(np.float32)


# ── Generate categorical features ─────────────────────────────────────────────
def generate_categorical(n_samples: int, n_features: int) -> tuple:
    """Returns (data [N, n_features], vocab_sizes [n_features])."""
    rng = np.random.RandomState(42)
    cat_vocab = [rng.randint(2, 15) for _ in range(n_features)]  # 2-15 classes each
    data = np.zeros((n_samples, n_features), dtype=np.int64)
    for i, C in enumerate(cat_vocab):
        probs = rng.dirichlet(np.ones(C) * 0.7)  # unequal class distribution
        data[:, i] = rng.choice(C, size=n_samples, p=probs)
    return data, cat_vocab


# ─────────────────────────────────────────────────────────────────────────────
def main():
    out_path = Path("data/real_data.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic data: {N_SAMPLES} rows × "
          f"({N_NUMERIC} numeric + {N_CAT} categorical) ...")

    x_num = generate_numeric(N_SAMPLES, N_NUMERIC)
    x_cat, vocab_sizes = generate_categorical(N_SAMPLES, N_CAT)

    num_cols = make_korean_num_cols(N_NUMERIC)
    cat_cols = make_korean_cat_cols(N_CAT)

    df_num = pd.DataFrame(x_num, columns=num_cols)
    df_cat = pd.DataFrame(x_cat, columns=cat_cols)
    df = pd.concat([df_num, df_cat], axis=1)

    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({df.shape[0]} rows, {df.shape[1]} columns)")

    # Save column lists for the pipeline
    import json
    meta = {
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "cat_vocab_sizes": vocab_sizes,
    }
    meta_path = Path("data/column_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Column metadata: {meta_path}")


if __name__ == "__main__":
    main()
