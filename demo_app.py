"""
TabDiff Imputation Demo — Streamlit App
=======================================

시각화 예시 화면: TabDiff vs CTGAN imputation 비교 대시보드

실행:
    streamlit run demo_app.py
"""
import warnings
warnings.filterwarnings("ignore")
import os, json, time
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── 페이지 설정 ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TabDiff Imputation Demo",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 한글 폰트 설정 ─────────────────────────────────────────────────────────────
@st.cache_resource
def init_font():
    from tabdiff_imputation.visualize import setup_korean_font
    return setup_korean_font()

font_name = init_font()

# ── CSS 스타일 ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 700; color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem; color: #555; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 14px 18px; border-left: 4px solid #2196F3;
        margin-bottom: 8px;
    }
    .metric-card.green  { border-left-color: #4CAF50; }
    .metric-card.orange { border-left-color: #FF9800; }
    .metric-card.red    { border-left-color: #F44336; }
    .winner-badge {
        display: inline-block; padding: 2px 10px;
        border-radius: 12px; font-size: 0.8rem; font-weight: 600;
        background: #E8F5E9; color: #2E7D32;
    }
    .section-header {
        font-size: 1.15rem; font-weight: 600;
        color: #1a1a2e; margin-top: 1rem; margin-bottom: 0.5rem;
        border-bottom: 2px solid #E3E8EF; padding-bottom: 4px;
    }
    .info-box {
        background: #EEF2FF; border-radius: 8px; padding: 12px 16px;
        border-left: 4px solid #6366F1; color: #3730A3;
        font-size: 0.9rem; margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 / 모델 로드 헬퍼
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    from tabdiff_imputation.config import TabDiffConfig
    from tabdiff_imputation.data_utils import TabularPreprocessor, load_and_split, apply_masking

    cfg = TabDiffConfig()
    preprocessor = TabularPreprocessor()
    with open("data/column_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    preprocessor.numeric_cols  = meta["numeric_cols"]
    preprocessor.categorical_cols = meta["categorical_cols"]

    _, _, (x_num_test, x_cat_test) = load_and_split(
        cfg.data_path, preprocessor, random_seed=42
    )
    return x_num_test, x_cat_test, preprocessor


@st.cache_resource
def load_models():
    from tabdiff_imputation.config import TabDiffConfig
    from tabdiff_imputation.trainer import TabDiffTrainer
    from tabdiff_imputation.ctgan_trainer import CTGANTrainer

    cfg = TabDiffConfig()
    preprocessor_stub, _, _ = _get_preprocessor_data()
    trainer  = TabDiffTrainer(cfg)
    model    = trainer.load_best(len(preprocessor_stub.numeric_cols), preprocessor_stub.cat_vocab_sizes)
    ctgan_tr = CTGANTrainer(cfg)
    ctgan_tr.load_best()
    return model, ctgan_tr, cfg


@st.cache_data
def _get_preprocessor_data():
    from tabdiff_imputation.data_utils import TabularPreprocessor, load_and_split
    from tabdiff_imputation.config import TabDiffConfig
    cfg = TabDiffConfig()
    preprocessor = TabularPreprocessor()
    with open("data/column_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    preprocessor.numeric_cols     = meta["numeric_cols"]
    preprocessor.categorical_cols = meta["categorical_cols"]
    _, _, _ = load_and_split(cfg.data_path, preprocessor, random_seed=42)
    return preprocessor, None, None


@st.cache_data
def load_results(mask_pct: int):
    path = Path(f"results/results_{mask_pct}pct.json")
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 실시간 imputation 헬퍼
# ─────────────────────────────────────────────────────────────────────────────
def run_imputation_live(mask_ratio: float, n_rows: int = 50):
    """소규모(n_rows)로 실시간 imputation 후 플롯 반환."""
    from tabdiff_imputation.config import TabDiffConfig
    from tabdiff_imputation.data_utils import apply_masking
    from tabdiff_imputation.imputer import TabDiffImputer
    from tabdiff_imputation.metrics import evaluate_imputation

    x_num_test, x_cat_test, preprocessor = load_data()
    model, ctgan_tr, cfg = load_models()

    cfg.n_ddim_steps = 20
    cfg.n_resample   = 1

    x_num_s = x_num_test[:n_rows]
    x_cat_s = x_cat_test[:n_rows]
    cat_vocab = preprocessor.cat_vocab_sizes
    num_cols  = preprocessor.numeric_cols
    cat_cols  = preprocessor.categorical_cols

    rng = np.random.RandomState(99)
    x_num_m, x_cat_m, mask_n, mask_c = apply_masking(
        x_num_s, x_cat_s, mask_ratio, cat_vocab, rng=rng
    )

    # TabDiff
    imputer = TabDiffImputer(model, cfg, cat_vocab)
    x_num_td, x_cat_td = imputer.impute(x_num_m, x_cat_m, mask_n, mask_c, chunk_size=25)

    # CTGAN
    x_num_cg, x_cat_cg = ctgan_tr.impute(
        x_num_m, x_cat_m, mask_n, mask_c,
        cat_vocab, num_cols, cat_cols,
        n_synthetic=2000, k=5,
    )

    res_td = evaluate_imputation(x_num_s, x_cat_s, x_num_td, x_cat_td,
                                  mask_n, mask_c, cat_vocab, num_cols, cat_cols)
    res_cg = evaluate_imputation(x_num_s, x_cat_s, x_num_cg, x_cat_cg,
                                  mask_n, mask_c, cat_vocab, num_cols, cat_cols)

    return (x_num_s, x_num_td, x_num_cg, x_cat_s, x_cat_td, x_cat_cg,
            mask_n, mask_c, num_cols, cat_cols, cat_vocab, res_td, res_cg)


# ─────────────────────────────────────────────────────────────────────────────
# 인라인 플롯 생성 함수
# ─────────────────────────────────────────────────────────────────────────────
def make_num_plot_inline(x_true, x_td, x_cg, mask, cols, n_feat=12, title=""):
    from scipy import stats as scipy_stats

    n_show = min(n_feat, len(cols))
    nc = 4
    nr = (n_show + nc - 1) // nc

    fig, axes = plt.subplots(nr, nc, figsize=(nc * 4, nr * 3))
    axes = np.array(axes).flatten()

    COLORS = {"orig": "#2196F3", "td": "#4CAF50", "cg": "#FF9800"}
    shown = 0
    for i in range(len(cols)):
        if shown >= n_show:
            break
        mi = mask[:, i]
        if mi.sum() < 3:
            continue
        ax = axes[shown]
        for arr, lbl, col in [
            (x_true[mi, i], "원본",     COLORS["orig"]),
            (x_td[mi, i],   "TabDiff",  COLORS["td"]),
            (x_cg[mi, i],   "CTGAN",    COLORS["cg"]),
        ]:
            d = arr[np.isfinite(arr)]
            if len(np.unique(d)) >= 4:
                try:
                    kde = scipy_stats.gaussian_kde(d, bw_method="silverman")
                    xr  = np.linspace(d.min(), d.max(), 200)
                    ax.plot(xr, kde(xr), color=col, label=lbl, linewidth=1.8)
                    ax.fill_between(xr, kde(xr), alpha=0.12, color=col)
                except Exception:
                    ax.hist(d, bins=25, density=True, color=col, label=lbl, alpha=0.5)
            else:
                ax.hist(d, bins=20, density=True, color=col, label=lbl, alpha=0.5)

        col_name = cols[i]
        ax.set_title(col_name if len(col_name) <= 22 else col_name[:19] + "…", fontsize=8.5)
        ax.set_xlabel("Value", fontsize=7.5)
        ax.tick_params(labelsize=7)
        if shown == 0:
            ax.legend(fontsize=7, framealpha=0.7)
        shown += 1

    for ax in axes[shown:]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=10, y=1.02)
    plt.tight_layout()
    return fig


def make_cat_plot_inline(x_true, x_td, x_cg, mask, cols, vocab_sizes, n_feat=8):
    n_show = min(n_feat, len(cols))
    nc = 4
    nr = (n_show + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 4, nr * 3))
    axes = np.array(axes).flatten()

    COLORS = {"orig": "#2196F3", "td": "#4CAF50", "cg": "#FF9800"}
    shown = 0
    for i in range(len(cols)):
        if shown >= n_show:
            break
        mi = mask[:, i]
        if mi.sum() < 2:
            continue
        C = vocab_sizes[i]
        ax = axes[shown]
        classes = np.arange(C)
        w = 0.26
        for j, (arr, lbl, col) in enumerate([
            (x_true[mi, i], "원본",    COLORS["orig"]),
            (x_td[mi, i],   "TabDiff", COLORS["td"]),
            (x_cg[mi, i],   "CTGAN",   COLORS["cg"]),
        ]):
            cnt = np.bincount(np.clip(arr.astype(int), 0, C - 1), minlength=C).astype(float)
            cnt = cnt / cnt.sum() if cnt.sum() > 0 else cnt
            ax.bar(classes + (j - 1) * w, cnt, w, label=lbl, color=col, alpha=0.82)

        ax.set_xticks(classes if C <= 15 else [])
        ax.set_title(cols[i] if len(cols[i]) <= 22 else cols[i][:19] + "…", fontsize=8.5)
        ax.set_ylabel("Freq.", fontsize=7.5)
        ax.tick_params(labelsize=7)
        if shown == 0:
            ax.legend(fontsize=7, framealpha=0.7)
        shown += 1

    for ax in axes[shown:]:
        ax.set_visible(False)
    plt.tight_layout()
    return fig


def make_metric_bar(res_td, res_cg, ratios_data=None):
    """TabDiff vs CTGAN 지표 막대그래프."""
    metrics = {
        "RMSE\n(수치형)":           ("mean_rmse_num",        True),
        "MAE\n(수치형)":            ("mean_mae_num",         True),
        "Wasserstein\n(수치형)":    ("mean_wasserstein_num", True),
        "Accuracy\n(범주형) ↑":     ("mean_accuracy_cat",    False),
        "JS Divergence\n(범주형)":  ("mean_js_div_cat",      True),
    }
    labels = list(metrics.keys())
    td_vals = [res_td["summary"].get(v, 0) or 0 for _, (v, _) in metrics.items()]
    cg_vals = [res_cg["summary"].get(v, 0) or 0 for _, (v, _) in metrics.items()]
    lower_better = [lb for _, (_, lb) in metrics.items()]

    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(labels))
    w = 0.35
    bars_td = ax.bar(x - w / 2, td_vals, w, label="TabDiff", color="#4CAF50", alpha=0.88)
    bars_cg = ax.bar(x + w / 2, cg_vals, w, label="CTGAN",   color="#FF9800", alpha=0.88)

    # 승자 표시
    for i, (td, cg, lb) in enumerate(zip(td_vals, cg_vals, lower_better)):
        if td is None or cg is None:
            continue
        winner_td = (td < cg) if lb else (td > cg)
        winner_bar = bars_td[i] if winner_td else bars_cg[i]
        ax.annotate("★", xy=(winner_bar.get_x() + winner_bar.get_width() / 2,
                              winner_bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=10, color="#C62828")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Score")
    ax.set_title("TabDiff vs CTGAN 성능 비교  (★ = Winner)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 메인 UI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── 헤더 ──────────────────────────────────────────────────────────────────
    st.markdown('<p class="main-title">🔵 TabDiff Imputation Demo</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">'
        'Mixed-type Diffusion Model (TabDiff) vs CTGAN — 결측값 보완 성능 비교 대시보드'
        '</p>',
        unsafe_allow_html=True,
    )

    # ── 사이드바 ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://img.shields.io/badge/model-TabDiff-blue?style=for-the-badge", use_container_width=True)
        st.markdown("### ⚙️ 설정")

        mask_ratio = st.selectbox(
            "마스킹 비율 (Mask Ratio)",
            [0.10, 0.20, 0.30],
            index=1,
            format_func=lambda x: f"{int(x*100)}%",
        )
        mode = st.radio(
            "실행 모드",
            ["사전 생성된 결과 보기", "실시간 Imputation 실행"],
            index=0,
        )
        st.markdown("---")
        st.markdown("### 📊 모델 정보")
        st.markdown("""
| 항목 | 값 |
|---|---|
| 확산 단계 T | 1000 |
| DDIM steps | 100 |
| 히든 레이어 | [2048,2048,1024,1024] |
| LR | 3e-4 (cosine) |
| 배치 크기 | 512 |
| Early Stop | patience=30 |
        """)
        st.markdown("---")
        st.markdown(
            "**폰트**: " + font_name + "  \n"
            "**데이터**: 합성 현대카드 데이터  \n"
            "**수치형**: 50개 (QT 변환 완료)  \n"
            "**범주형**: 10개 (2–15 classes)"
        )

    # ── 탭 구성 ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 학습 곡선",
        "🔢 수치형 분포 비교",
        "🏷️ 범주형 분포 비교",
        "📊 성능 지표 비교",
    ])

    mask_pct = int(mask_ratio * 100)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: 학습 곡선
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<p class="section-header">TabDiff 학습 손실 곡선</p>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            loss_img = Path("plots/training_loss.png")
            if loss_img.exists():
                st.image(str(loss_img), use_container_width=True)
            else:
                st.warning("학습 곡선 이미지가 없습니다. `run_pipeline.py`를 먼저 실행하세요.")

        with col2:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("""
**학습 설정**

- **LR 스케줄**: Linear Warmup + Cosine Annealing
- **Early Stopping**: Val Loss 기준, patience=30
- **Best Model**: Val Loss 최소 시 자동 저장
- **Gradient Clip**: norm=1.0
- **AMP**: fp16 혼합 정밀도

**손실 구성**
- `L_total = λ_num · L_num + λ_cat · L_cat`
- `L_num`: MSE(예측 노이즈, 실제 노이즈)
- `L_cat`: CrossEntropy(마스킹된 위치)
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<p class="section-header">확산 과정 개요</p>', unsafe_allow_html=True)
        arch_col1, arch_col2, arch_col3 = st.columns(3)
        with arch_col1:
            st.info("**수치형 (연속)**\nGaussian DDPM\nCosine Noise Schedule\n→ 노이즈 예측 (MSE)")
        with arch_col2:
            st.success("**범주형 (이산)**\nAbsorbing Diffusion\nMASK token 기반\n→ 원본 클래스 예측 (CE)")
        with arch_col3:
            st.warning("**Imputation**\nRepaint-DDIM\n관측값 조건부 생성\n→ 마스킹 위치만 복원")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: 수치형 분포 비교
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown(
            f'<p class="section-header">수치형 변수 분포 비교 — 마스킹 {mask_pct}%</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"마스킹된 {mask_pct}% 위치에서 **원본 vs TabDiff vs CTGAN** 분포를 KDE로 비교합니다.",
        )

        if mode == "사전 생성된 결과 보기":
            # 사전 생성 이미지 표시
            fig_files = sorted(Path("plots").glob(f"numeric_{mask_pct}pct_fig*.png"))
            if fig_files:
                for f in fig_files:
                    st.image(str(f), use_container_width=True)
            else:
                st.warning(f"plots/numeric_{mask_pct}pct_fig*.png 파일이 없습니다.")
        else:
            with st.spinner(f"실시간 imputation 실행 중 (mask={mask_pct}%, 50 rows) …"):
                try:
                    res = run_imputation_live(mask_ratio, n_rows=50)
                    x_num_s, x_num_td, x_num_cg, x_cat_s, x_cat_td, x_cat_cg, \
                        mask_n, mask_c, num_cols, cat_cols, cat_vocab, res_td, res_cg = res

                    st.success("Imputation 완료!")
                    fig = make_num_plot_inline(
                        x_num_s, x_num_td, x_num_cg, mask_n, num_cols, n_feat=12,
                        title=f"수치형 변수 KDE 비교 — mask {mask_pct}%"
                    )
                    st.pyplot(fig)
                    plt.close(fig)
                    st.session_state["live_res"] = (res_td, res_cg)
                    st.session_state["live_cat"] = (x_cat_s, x_cat_td, x_cat_cg, mask_c, cat_cols, cat_vocab)
                except Exception as e:
                    st.error(f"오류 발생: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: 범주형 분포 비교
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown(
            f'<p class="section-header">범주형 변수 분포 비교 — 마스킹 {mask_pct}%</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "마스킹된 위치의 **클래스 빈도 분포**를 TabDiff · CTGAN과 비교합니다. "
            "(막대: 원본=파랑, TabDiff=초록, CTGAN=주황)"
        )

        if mode == "사전 생성된 결과 보기":
            fig_files = sorted(Path("plots").glob(f"categorical_{mask_pct}pct_fig*.png"))
            if fig_files:
                for f in fig_files:
                    st.image(str(f), use_container_width=True)
            else:
                st.warning(f"plots/categorical_{mask_pct}pct_fig*.png 파일이 없습니다.")
        else:
            # 실시간 모드: tab2에서 계산된 결과 재사용
            if "live_cat" in st.session_state:
                x_cat_s, x_cat_td, x_cat_cg, mask_c, cat_cols, cat_vocab = \
                    st.session_state["live_cat"]
                fig = make_cat_plot_inline(
                    x_cat_s, x_cat_td, x_cat_cg, mask_c, cat_cols, cat_vocab, n_feat=8
                )
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("먼저 '수치형 분포 비교' 탭에서 실시간 실행을 완료하세요.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4: 성능 지표 비교
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown(
            '<p class="section-header">TabDiff vs CTGAN 성능 지표</p>',
            unsafe_allow_html=True,
        )

        if mode == "사전 생성된 결과 보기":
            results_data = load_results(mask_pct)
        elif "live_res" in st.session_state:
            res_td, res_cg = st.session_state["live_res"]
            results_data = {"tabdiff": res_td, "ctgan": res_cg}
        else:
            results_data = load_results(mask_pct)

        if results_data:
            res_td = results_data["tabdiff"]
            res_cg = results_data["ctgan"]
            s_td   = res_td["summary"]
            s_cg   = res_cg["summary"]

            # ── 핵심 지표 카드 ────────────────────────────────────────────────
            c1, c2, c3, c4, c5 = st.columns(5)
            metric_cards = [
                (c1, "RMSE (수치형)", "mean_rmse_num",        True,  "blue"),
                (c2, "MAE (수치형)",  "mean_mae_num",         True,  "blue"),
                (c3, "Wasserstein",   "mean_wasserstein_num", True,  "blue"),
                (c4, "Accuracy (범주형) ↑", "mean_accuracy_cat",  False, "green"),
                (c5, "JSD (범주형)",  "mean_js_div_cat",      True,  "orange"),
            ]
            for col, label, key, lower_better, color in metric_cards:
                td_v = s_td.get(key)
                cg_v = s_cg.get(key)
                if td_v is None or cg_v is None:
                    col.metric(label, "N/A")
                    continue
                winner_td = (td_v < cg_v) if lower_better else (td_v > cg_v)
                delta = td_v - cg_v
                col.metric(
                    label=f"{'🟢' if winner_td else '🟠'} {label}",
                    value=f"{td_v:.4f}",
                    delta=f"TabDiff {'↓' if lower_better else '↑'} {abs(delta):.4f} vs CTGAN",
                    delta_color="normal" if winner_td else "inverse",
                )

            st.markdown("---")

            # ── 통합 막대 차트 ────────────────────────────────────────────────
            col_l, col_r = st.columns([2, 1])
            with col_l:
                fig_bar = make_metric_bar(res_td, res_cg)
                st.pyplot(fig_bar)
                plt.close(fig_bar)

            with col_r:
                # 마스킹 비율별 비교 (저장된 JSON 있는 경우)
                all_ratios = {}
                for pct in [10, 20, 30]:
                    r = load_results(pct)
                    if r:
                        all_ratios[pct] = r

                if len(all_ratios) >= 2:
                    st.markdown("**마스킹 비율별 RMSE**")
                    ratio_df = pd.DataFrame({
                        "Mask %": [f"{p}%" for p in sorted(all_ratios)],
                        "TabDiff": [all_ratios[p]["tabdiff"]["summary"].get("mean_rmse_num", 0)
                                    for p in sorted(all_ratios)],
                        "CTGAN":   [all_ratios[p]["ctgan"]["summary"].get("mean_rmse_num", 0)
                                    for p in sorted(all_ratios)],
                    }).set_index("Mask %")
                    st.bar_chart(ratio_df)

            st.markdown("---")

            # ── 요약 표 ───────────────────────────────────────────────────────
            st.markdown('<p class="section-header">전체 지표 요약 테이블</p>', unsafe_allow_html=True)

            def fmt(v):
                return f"{v:.5f}" if v is not None else "N/A"

            summary_rows = []
            metric_map = [
                ("RMSE (수치형)",          "mean_rmse_num",        True),
                ("MAE (수치형)",           "mean_mae_num",         True),
                ("Wasserstein (수치형)",   "mean_wasserstein_num", True),
                ("Accuracy (범주형) ↑",    "mean_accuracy_cat",    False),
                ("JS Divergence (범주형)", "mean_js_div_cat",      True),
            ]
            for label, key, lower_better in metric_map:
                td_v = s_td.get(key)
                cg_v = s_cg.get(key)
                if td_v is None or cg_v is None:
                    continue
                winner_td = (td_v < cg_v) if lower_better else (td_v > cg_v)
                summary_rows.append({
                    "지표": label,
                    "TabDiff": fmt(td_v),
                    "CTGAN":   fmt(cg_v),
                    "승자": "🟢 TabDiff" if winner_td else "🟠 CTGAN",
                    "개선율": f"{abs(td_v - cg_v) / max(abs(cg_v), 1e-9) * 100:.1f}%",
                })

            df_summary = pd.DataFrame(summary_rows)
            st.dataframe(df_summary, use_container_width=True, hide_index=True)

            # ── 전체 마스킹 비율 비교 플롯 ────────────────────────────────────
            if len(all_ratios) >= 2:
                st.markdown('<p class="section-header">마스킹 비율별 성능 비교</p>', unsafe_allow_html=True)
                summary_img = Path("plots/metric_summary.png")
                if summary_img.exists():
                    st.image(str(summary_img), use_container_width=True)

        else:
            st.warning(f"results/results_{mask_pct}pct.json 파일이 없습니다. 파이프라인을 먼저 실행하세요.")

    # ── 푸터 ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "TabDiff: A Mixed-Type Diffusion Model for Tabular Data Generation  |  "
        "현대카드 고객 데이터 Imputation 시연용 Demo  |  "
        "Font: " + font_name
    )


if __name__ == "__main__":
    main()
