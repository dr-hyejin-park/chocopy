"""
prepare_data.py — Parquet → Pipeline 데이터 준비 스크립트
=========================================================

실제 현대카드 Parquet 파일을 읽어 파이프라인이 기대하는 형식으로 변환합니다.

출력물
------
  data/real_data.csv      — 파이프라인 학습/평가용 CSV
  data/column_meta.json   — 컬럼 목록 + 범주형 vocab 크기

자동 컬럼 분류 기준
-------------------
  수치형 : 숫자 dtype이고 고유값 수 > cat_threshold (기본 20)
  범주형 : 문자열 dtype이거나, 숫자지만 고유값 수 ≤ cat_threshold

사용 예시
---------
  # 기본 실행 (자동 분류)
  python prepare_data.py --input /path/to/data.parquet

  # 임계값 조정 (고유값 50 이하면 범주형으로 분류)
  python prepare_data.py --input data.parquet --cat_threshold 50

  # 특정 컬럼을 수치형/범주형으로 강제 지정
  python prepare_data.py --input data.parquet \\
      --force_numeric age income \\
      --force_categorical grade channel

  # 특정 컬럼 제외
  python prepare_data.py --input data.parquet --drop_cols id created_at

  # 다른 출력 경로
  python prepare_data.py --input data.parquet \\
      --out_csv data/real_data.csv \\
      --out_meta data/column_meta.json
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Parquet 데이터를 TabDiff 파이프라인 형식으로 변환"
    )
    p.add_argument("--input", required=True,
                   help="입력 Parquet 파일 경로 (또는 CSV, 확장자로 자동 판별)")
    p.add_argument("--out_csv",  default="data/real_data.csv",
                   help="출력 CSV 경로 (기본: data/real_data.csv)")
    p.add_argument("--out_meta", default="data/column_meta.json",
                   help="출력 메타 JSON 경로 (기본: data/column_meta.json)")
    p.add_argument("--cat_threshold", type=int, default=20,
                   help="이 값 이하의 고유값을 갖는 수치 컬럼은 범주형으로 분류 (기본: 20)")
    p.add_argument("--force_numeric", nargs="*", default=[],
                   help="강제로 수치형으로 처리할 컬럼명 목록")
    p.add_argument("--force_categorical", nargs="*", default=[],
                   help="강제로 범주형으로 처리할 컬럼명 목록")
    p.add_argument("--drop_cols", nargs="*", default=[],
                   help="제거할 컬럼명 목록 (id, 날짜 등)")
    p.add_argument("--drop_na_thresh", type=float, default=0.5,
                   help="결측률이 이 비율 초과인 컬럼 자동 제거 (기본: 0.5 = 50%%)")
    p.add_argument("--sample_n", type=int, default=None,
                   help="지정 시 해당 행 수만큼 무작위 샘플링 (전체 사용 시 생략)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def load_file(path: str) -> pd.DataFrame:
    """확장자를 보고 Parquet 또는 CSV를 읽는다."""
    p = Path(path)
    if not p.exists():
        sys.exit(f"[오류] 파일을 찾을 수 없습니다: {path}")

    suffix = p.suffix.lower()
    if suffix in (".parquet", ".pq"):
        print(f"Parquet 읽기: {path}")
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            sys.exit(f"[오류] Parquet 읽기 실패: {e}\n  pyarrow 또는 fastparquet 설치 필요")
    elif suffix == ".csv":
        print(f"CSV 읽기: {path}")
        df = pd.read_csv(path, low_memory=False)
    else:
        # 확장자 없이도 Parquet 시도
        print(f"확장자 미확인 — Parquet로 시도: {path}")
        try:
            df = pd.read_parquet(path)
        except Exception:
            df = pd.read_csv(path, low_memory=False)

    print(f"  로드 완료: {df.shape[0]:,} 행 × {df.shape[1]:,} 열")
    return df


# ─────────────────────────────────────────────────────────────────────────────
def classify_columns(
    df: pd.DataFrame,
    cat_threshold: int,
    force_numeric: list,
    force_categorical: list,
) -> tuple:
    """
    컬럼을 수치형 / 범주형으로 분류한다.

    반환
    ----
    numeric_cols    : List[str]
    categorical_cols: List[str]
    """
    force_num_set = set(force_numeric)
    force_cat_set = set(force_categorical)

    numeric_cols, categorical_cols = [], []

    for col in df.columns:
        # 강제 지정 우선
        if col in force_num_set:
            numeric_cols.append(col)
            continue
        if col in force_cat_set:
            categorical_cols.append(col)
            continue

        dtype = df[col].dtype
        n_unique = df[col].nunique(dropna=True)

        if pd.api.types.is_numeric_dtype(dtype):
            if n_unique <= cat_threshold:
                # 고유값이 적으면 범주형으로 취급
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            # 문자열·object·카테고리 dtype → 범주형
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


# ─────────────────────────────────────────────────────────────────────────────
def compute_vocab_sizes(df: pd.DataFrame, categorical_cols: list) -> list:
    """범주형 컬럼별 고유값 수를 반환 (NaN 제외)."""
    return [int(df[col].nunique(dropna=True)) for col in categorical_cols]


# ─────────────────────────────────────────────────────────────────────────────
def print_summary(df, numeric_cols, categorical_cols, vocab_sizes, drop_na_cols):
    """분류 결과 요약 출력."""
    print("\n" + "=" * 60)
    print("  컬럼 분류 결과")
    print("=" * 60)
    print(f"  수치형   : {len(numeric_cols):4d} 개")
    print(f"  범주형   : {len(categorical_cols):4d} 개")
    if drop_na_cols:
        print(f"  결측 제거 : {len(drop_na_cols):4d} 개  ({', '.join(drop_na_cols[:5])}"
              f"{'…' if len(drop_na_cols) > 5 else ''})")
    print()

    if numeric_cols:
        print("  [수치형 컬럼 — 상위 10개]")
        for col in numeric_cols[:10]:
            series = df[col].dropna()
            print(f"    {col:<40s}  "
                  f"min={series.min():.3g}  max={series.max():.3g}  "
                  f"null={df[col].isna().mean():.1%}")
        if len(numeric_cols) > 10:
            print(f"    … 외 {len(numeric_cols)-10}개")

    print()
    if categorical_cols:
        print("  [범주형 컬럼 — 전체]")
        for col, vs in zip(categorical_cols, vocab_sizes):
            top_vals = df[col].value_counts(dropna=True).index[:3].tolist()
            print(f"    {col:<40s}  vocab={vs:3d}  "
                  f"null={df[col].isna().mean():.1%}  "
                  f"예시={top_vals}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    # ── 1. 데이터 로드 ────────────────────────────────────────────────────────
    df = load_file(args.input)

    # ── 2. 지정 컬럼 제거 ────────────────────────────────────────────────────
    drop_explicit = [c for c in args.drop_cols if c in df.columns]
    if drop_explicit:
        df = df.drop(columns=drop_explicit)
        print(f"제거된 컬럼 (명시적): {drop_explicit}")

    # ── 3. 결측률 높은 컬럼 제거 ─────────────────────────────────────────────
    na_rate = df.isna().mean()
    drop_na_cols = na_rate[na_rate > args.drop_na_thresh].index.tolist()
    if drop_na_cols:
        df = df.drop(columns=drop_na_cols)
        print(f"제거된 컬럼 (결측률>{args.drop_na_thresh:.0%}): "
              f"{drop_na_cols[:5]}{'…' if len(drop_na_cols) > 5 else ''}")

    # 완전 빈 행 제거
    before = len(df)
    df = df.dropna(how="all")
    if len(df) < before:
        print(f"완전 빈 행 {before - len(df):,}개 제거")

    # ── 4. 샘플링 ────────────────────────────────────────────────────────────
    if args.sample_n is not None and args.sample_n < len(df):
        df = df.sample(n=args.sample_n, random_state=args.seed).reset_index(drop=True)
        print(f"샘플링: {len(df):,} 행 선택")

    # ── 5. 컬럼 분류 ─────────────────────────────────────────────────────────
    numeric_cols, categorical_cols = classify_columns(
        df,
        cat_threshold=args.cat_threshold,
        force_numeric=args.force_numeric,
        force_categorical=args.force_categorical,
    )

    if not numeric_cols:
        sys.exit("[오류] 수치형 컬럼이 0개입니다. --cat_threshold를 높이거나 "
                 "--force_numeric으로 직접 지정하세요.")
    if not categorical_cols:
        print("[경고] 범주형 컬럼이 0개입니다. 수치형 전용 모드로 진행합니다.")

    # vocab 크기 계산
    vocab_sizes = compute_vocab_sizes(df, categorical_cols)

    # ── 6. 요약 출력 ─────────────────────────────────────────────────────────
    print_summary(df, numeric_cols, categorical_cols, vocab_sizes, drop_na_cols)

    # ── 7. 범주형 컬럼 문자열 통일 (파이프라인이 str로 LabelEncode 함) ───────
    for col in categorical_cols:
        df[col] = df[col].astype(str)

    # ── 8. 최종 DataFrame: 수치형 + 범주형 순으로 열 정렬 ────────────────────
    df_out = df[numeric_cols + categorical_cols]

    # ── 9. CSV 저장 ──────────────────────────────────────────────────────────
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nCSV 저장 완료: {out_csv}")
    print(f"  {df_out.shape[0]:,} 행 × {df_out.shape[1]:,} 열")

    # ── 10. column_meta.json 저장 ─────────────────────────────────────────────
    meta = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "cat_vocab_sizes": vocab_sizes,
        # 참고용 통계 (파이프라인에서는 사용 안 함)
        "_info": {
            "source": str(Path(args.input).resolve()),
            "n_rows": int(df_out.shape[0]),
            "n_numeric": len(numeric_cols),
            "n_categorical": len(categorical_cols),
            "cat_threshold_used": args.cat_threshold,
        },
    }
    out_meta = Path(args.out_meta)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"메타 저장 완료: {out_meta}")

    # ── 11. 파이프라인 실행 안내 ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("다음 명령으로 파이프라인을 실행하세요:")
    print(f"  python run_pipeline.py --data {out_csv} --meta {out_meta}")
    print("─" * 60)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
