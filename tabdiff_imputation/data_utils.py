"""Data loading, preprocessing, and masking utilities."""
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
class TabularDataset(Dataset):
    """PyTorch dataset wrapping numeric + categorical tensors."""

    def __init__(self, x_num: np.ndarray, x_cat: np.ndarray):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, idx):
        return self.x_num[idx], self.x_cat[idx]


# ─────────────────────────────────────────────────────────────────────────────
class TabularPreprocessor:
    """
    Preprocesses a DataFrame into:
      - x_num : float32 array  [N, n_num]  (already quantile-transformed input)
      - x_cat : int32 array    [N, n_cat]  (label-encoded categoricals)

    Categorical MASK token index = vocab_size (stored in self.cat_vocab_sizes).
    """

    def __init__(
        self,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.cat_vocab_sizes: List[int] = []
        self.num_scaler: Optional[QuantileTransformer] = None
        self.fitted = False

    # ------------------------------------------------------------------
    def auto_detect_cols(self, df: pd.DataFrame):
        """Infer numeric / categorical columns from DataFrame dtypes."""
        num_cols, cat_cols = [], []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                num_cols.append(col)
            else:
                cat_cols.append(col)
        return num_cols, cat_cols

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame):
        if self.numeric_cols is None or self.categorical_cols is None:
            n, c = self.auto_detect_cols(df)
            if self.numeric_cols is None:
                self.numeric_cols = n
            if self.categorical_cols is None:
                self.categorical_cols = c

        self.cat_vocab_sizes = []
        self.label_encoders = {}

        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.label_encoders[col] = le
            self.cat_vocab_sizes.append(len(le.classes_))

        # Fit QuantileTransformer on numeric features → output ~N(0,1)
        # Diffusion models assume normalised input; this is the single most
        # impactful fix when raw features span very different scales.
        x_num_raw = df[self.numeric_cols].values.astype(np.float32)
        n_quantiles = min(1000, max(10, len(df)))
        self.num_scaler = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=n_quantiles,
            random_state=42,
            subsample=max(n_quantiles, 10_000),
        )
        self.num_scaler.fit(x_num_raw)

        self.fitted = True
        return self

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        assert self.fitted, "Call fit() first."
        x_num = df[self.numeric_cols].values.astype(np.float32)
        # Apply quantile normalization so all features are ~N(0,1)
        if self.num_scaler is not None:
            x_num = self.num_scaler.transform(x_num).astype(np.float32)
        x_cat = np.zeros((len(df), len(self.categorical_cols)), dtype=np.int64)
        for i, col in enumerate(self.categorical_cols):
            le = self.label_encoders[col]
            vals = df[col].astype(str)
            # Handle unseen categories → last known class
            vals = vals.map(lambda v: v if v in le.classes_ else le.classes_[0])
            x_cat[:, i] = le.transform(vals)
        return x_num, x_cat

    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    def inverse_transform_num(self, x_num: np.ndarray) -> pd.DataFrame:
        arr = np.asarray(x_num, dtype=np.float32)
        if self.num_scaler is not None:
            arr = self.num_scaler.inverse_transform(arr).astype(np.float32)
        return pd.DataFrame(arr, columns=self.numeric_cols)

    def inverse_transform_cat(self, x_cat: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame()
        for i, col in enumerate(self.categorical_cols):
            le = self.label_encoders[col]
            # Clip to valid range (in case model generates out-of-range)
            idx = np.clip(x_cat[:, i], 0, len(le.classes_) - 1)
            df[col] = le.inverse_transform(idx)
        return df


# ─────────────────────────────────────────────────────────────────────────────
def load_and_split(
    data_path: str,
    preprocessor: TabularPreprocessor,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_seed: int = 42,
):
    """Load CSV → train/val/test split → preprocess."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_csv(data_path, low_memory=False)

    df = df.dropna(how="all")
    n = len(df)
    test_n = int(n * test_size)
    val_n = int(n * val_size)

    rng = np.random.RandomState(random_seed)
    idx = rng.permutation(n)
    test_idx = idx[:test_n]
    val_idx = idx[test_n: test_n + val_n]
    train_idx = idx[test_n + val_n:]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    preprocessor.fit(df_train)
    x_num_train, x_cat_train = preprocessor.transform(df_train)
    x_num_val, x_cat_val = preprocessor.transform(df_val)
    x_num_test, x_cat_test = preprocessor.transform(df_test)

    return (
        (x_num_train, x_cat_train),
        (x_num_val, x_cat_val),
        (x_num_test, x_cat_test),
    )


# ─────────────────────────────────────────────────────────────────────────────
def make_dataloaders(
    x_num_train, x_cat_train,
    x_num_val, x_cat_val,
    batch_size: int = 512,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = TabularDataset(x_num_train, x_cat_train)
    val_ds = TabularDataset(x_num_val, x_cat_val)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
def apply_masking(
    x_num: np.ndarray,
    x_cat: np.ndarray,
    mask_ratio: float,
    cat_vocab_sizes: List[int],
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly mask `mask_ratio` of entries in both numeric and categorical.

    Returns
    -------
    x_num_masked : numeric with NaN at masked positions
    x_cat_masked : categorical with MASK token (= vocab_size) at masked positions
    mask_num      : bool array [N, n_num], True = masked
    mask_cat      : bool array [N, n_cat], True = masked
    """
    if rng is None:
        rng = np.random.RandomState(42)

    N, n_num = x_num.shape
    _, n_cat = x_cat.shape

    mask_num = rng.random((N, n_num)) < mask_ratio
    mask_cat = rng.random((N, n_cat)) < mask_ratio

    x_num_masked = x_num.copy().astype(np.float32)
    x_num_masked[mask_num] = np.nan

    x_cat_masked = x_cat.copy().astype(np.int64)
    for i, C in enumerate(cat_vocab_sizes):
        x_cat_masked[mask_cat[:, i], i] = C  # MASK token = vocab_size

    return x_num_masked, x_cat_masked, mask_num, mask_cat
