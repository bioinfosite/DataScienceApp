import streamlit as st
import pandas as pd
import numpy as np

# 高度な指標（SciPy）
try:
    from scipy.spatial.distance import (
        euclidean, cityblock, chebyshev, minkowski,
        canberra, braycurtis, cosine, correlation, hamming, jaccard
    )
    from scipy.stats import wasserstein_distance, entropy, ks_2samp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# 形状系
try:
    from fastdtw import fastdtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False


# ==========================================================
# ユーティリティ：ペースト入力 → ベクトルへ
# ==========================================================
def parse_text_to_series(text):
    try:
        values = [float(v) for v in text.replace(",", " ").split()]
        return pd.Series(values, dtype=float)
    except:
        return None


# ==========================================================
# 指標計算関数群
# ==========================================================

def compute_metrics(a, b, selected_categories):

    results = {}

    # ------------------------------
    # A. 回帰誤差系（Regression Errors）
    # ------------------------------
    if "Regression Errors" in selected_categories:
        results["MAE"] = np.mean(np.abs(a - b))
        results["MSE"] = np.mean((a - b) ** 2)
        results["RMSE"] = np.sqrt(np.mean((a - b) ** 2))
        results["MAPE"] = np.mean(np.abs((a - b) / (a + 1e-9)))
        results["SMAPE"] = np.mean(2 * np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-9))
        results["MedianAE"] = np.median(np.abs(a - b))
        results["Max Error"] = np.max(np.abs(a - b))
        results["R2"] = 1 - np.sum((a - b) ** 2) / np.sum((a - np.mean(a)) ** 2 + 1e-9)

    # ------------------------------
    # B. 距離系（Distance Metrics）
    # ------------------------------
    if "Distance Metrics" in selected_categories:
        results["L1 (Manhattan)"] = np.sum(np.abs(a - b))
        results["L2 (Euclidean)"] = np.sqrt(np.sum((a - b) ** 2))
        results["L∞ (Chebyshev)"] = np.max(np.abs(a - b))

        if SCIPY_AVAILABLE:
            results["Canberra"] = canberra(a, b)
            results["Bray–Curtis"] = braycurtis(a, b)
            results["Wasserstein-1 (EMD)"] = wasserstein_distance(a, b)
            results["Minkowski (p=3)"] = minkowski(a, b, 3)
            results["Hamming Distance"] = hamming(a > 0, b > 0)
            try:
                results["Jaccard Distance (binary)"] = jaccard(a > 0, b > 0)
            except:
                pass

    # ------------------------------
    # C. 類似度系（Similarity Metrics）
    # ------------------------------
    if "Similarity" in selected_categories:
        results["Cosine Similarity"] = 1 - cosine(a, b) if SCIPY_AVAILABLE else np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        results["Pearson Corr"] = np.corrcoef(a, b)[0, 1]
        results["Spearman Corr"] = pd.Series(a).corr(pd.Series(b), method="spearman")
        results["Kendall Tau"] = pd.Series(a).corr(pd.Series(b), method="kendall")

    # ------------------------------
    # D. 情報量（Information / Divergence）
    # ------------------------------
    if "Information Divergence" in selected_categories:
        pa = np.abs(a) / np.sum(np.abs(a))
        pb = np.abs(b) / np.sum(np.abs(b))
        pa += 1e-12
        pb += 1e-12
        results["KL(a||b)"] = np.sum(pa * np.log(pa / pb))
        results["KL(b||a)"] = np.sum(pb * np.log(pb / pa))
        results["JS Divergence"] = 0.5 * entropy(pa, (pa + pb) / 2) + 0.5 * entropy(pb, (pa + pb) / 2)

    # ------------------------------
    # E. 形状比較（Curve / Sequence）
    # ------------------------------
    if "Shape / Sequence" in selected_categories:
        if DTW_AVAILABLE:
            dtw_dist, _ = fastdtw(a, b)
            results["DTW Distance"] = dtw_dist
        results["KS Statistic"] = ks_2samp(a, b).statistic

    return results



# ==========================================================
# Streamlit UI
# ==========================================================
st.title("📊 ベクトル比較ツール（70+ 指標対応）")

st.write("2つのベクトルを比較して様々な誤差・距離・類似度・情報量を計算します。")

# ------------------------------
# 入力方法の選択
# ------------------------------
method = st.radio("入力方法を選択：", ["ペースト入力", "CSV / Excel"])

if method == "ペースト入力":
    col1, col2 = st.columns(2)

    with col1:
        text_a = st.text_area("ベクトル A（例: 1 2 3 4 ）")
    with col2:
        text_b = st.text_area("ベクトル B（例: 1 1 2 3 ）")

    a = parse_text_to_series(text_a) if text_a else None
    b = parse_text_to_series(text_b) if text_b else None

else:
    uploaded = st.file_uploader("CSV / Excel アップロード", ["csv", "xlsx"])
    if uploaded:
        df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        st.dataframe(df.head())
        colA = st.selectbox("ベクトル A 列", df.columns)
        colB = st.selectbox("ベクトル B 列", df.columns)
        a = df[colA].astype(float)
        b = df[colB].astype(float)
    else:
        a = b = None


# ------------------------------
# 指標カテゴリを選択
# ------------------------------
categories = st.multiselect(
    "計算したい指標カテゴリを選択：",
    [
        "Regression Errors",
        "Distance Metrics",
        "Similarity",
        "Information Divergence",
        "Shape / Sequence",
    ],
    default=["Regression Errors", "Distance Metrics"],
)


# ------------------------------
# 計算実行
# ------------------------------
if st.button("🚀 計算する"):

    if a is None or b is None:
        st.error("ベクトルが入力されていません。")
        st.stop()

    if len(a) != len(b):
        st.error("ベクトルの長さが異なります。")
        st.stop()

    results = compute_metrics(a.values, b.values, categories)
    results_df = pd.DataFrame(list(results.items()), columns=["指標", "値"])

    st.subheader("📈 結果")
    st.dataframe(results_df, width="stretch")

    # ダウンロード
    csv = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 結果をCSVでダウンロード",
        csv,
        "vector_metrics.csv",
        "text/csv",
    )

with st.expander("📘 使用可能な指標一覧（説明付き）"):
    st.markdown("""
## 🟦 Regression Errors（回帰誤差）
| 指標 | 説明 |
|------|------|
| **MAE (Mean Absolute Error)** | 平均絶対誤差。誤差の絶対値の平均。外れ値に比較的強い。 |
| **MSE (Mean Squared Error)** | 平均二乗誤差。誤差を二乗して平均。大きな誤差をより重視。 |
| **RMSE (Root MSE)** | MSE の平方根。元のスケールに戻るため解釈しやすい。 |
| **MAPE (Mean Absolute Percentage Error)** | 誤差の割合（%）を平均。0に近い値に弱い。 |
| **SMAPE (Symmetric MAPE)** | MAPE の改良版。対称性を持ち、0除算を避ける。 |
| **MedianAE (Median Absolute Error)** | 中央絶対誤差。外れ値に非常に強い。 |
| **Max Error** | 最大誤差（ワーストケース）。 |
| **R2 Score** | 決定係数。1に近いほど良い。負になることもある。 |

---

## 🟩 Distance Metrics（ベクトルの距離）
| 指標 | 説明 |
|------|------|
| **L1（Manhattan 距離）** | 絶対値の総和。直交距離。 |
| **L2（Euclidean 距離）** | ユークリッド距離。一般的な直線距離。 |
| **L∞（Chebyshev 距離）** | 各要素差の最大値。 |
| **Canberra 距離** | 小さい値に敏感。比率比較に向く。 |
| **Bray–Curtis 距離** | 0〜1 の範囲。生態学などでよく使われる。 |
| **Wasserstein-1（Earth Mover's Distance）** | 分布の形状（重みを動かすコスト）を比較。 |
| **Minkowski 距離 p=3** | L1 と L2 の一般化。pで重みが変わる。 |
| **Hamming 距離** | 2つのベクトルが一致しない要素の割合。 |
| **Jaccard 距離（binary）** | 共有される「1」の割合に基づく距離。 |

---

## 🟨 Similarity（類似度）
| 指標 | 説明 |
|------|------|
| **Cosine Similarity** | 角度ベース類似度。方向が近いほど高い。 |
| **Pearson 相関** | 線形関係の強さを測る。-1〜1。 |
| **Spearman 相関** | 順位相関。単調関係を評価。 |
| **Kendall Tau** | 順序の一致度を測る。順位の専門的指標。 |

---

## 🟥 Information Divergence（情報量・分布比較）
| 指標 | 説明 |
|------|------|
| **KL Divergence（KL距離）** | 確率分布の差異。非対称。0が完全一致。 |
| **JS Divergence** | KL の対称版。常に有限で扱いやすい。 |
| **Entropy（エントロピー）** | 分布の不確実性を測る（内部で使用）。 |

---

## 🟪 Shape / Sequence（形状・時系列比較）
| 指標 | 説明 |
|------|------|
| **DTW（Dynamic Time Warping）** | 時系列の形状を柔軟に比較。長さが違っても可。 |
| **KS Statistic（Kolmogorov-Smirnov 統計量）** | 2つの分布の最大差。0が完全一致。 |

---

## 📌 その他（補足説明）
- **回帰誤差**は「実測と予測の差」を見るため ML モデルの評価に使う  
- **距離指標**はベクトル空間の位置関係を比較  
- **類似度**は「どれだけ似ているか」を測る（1に近いほどよい）  
- **情報量**は確率分布として扱うため要正規化  
- **形状比較**は時系列・波形の一致度を見る  
""")
