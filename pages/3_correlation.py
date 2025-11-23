import streamlit as st
import pandas as pd
import plotly.express as px
from io import StringIO
from app import memo_sidebar

# サイドバー
memo_sidebar()

st.title("📈 相関分析")

# -------------------------------------------------------
# ① ファイルアップロード
# -------------------------------------------------------
uploaded = st.file_uploader(
    "相関分析用の CSV / Excel をアップロード",
    type=["csv", "xlsx"],
    key="corr_uploader",
)

if not uploaded:
    st.info("ファイルをアップロードしてください。")
    st.stop()


# -------------------------------------------------------
# ② キャッシュ付きファイル読み込み
# -------------------------------------------------------
@st.cache_data
def load_data(file_bytes, file_name, sheet_name=None):
    """ファイルとシート名をキーにキャッシュ"""
    if file_name.endswith(".csv"):
        return pd.read_csv(StringIO(file_bytes.decode("utf-8", errors="ignore")))

    if file_name.endswith(".xlsx"):
        # Excel の場合
        if sheet_name is None:
            xls = pd.ExcelFile(file_bytes)
            sheet_name = xls.sheet_names[0]
        return pd.read_excel(file_bytes, sheet_name=sheet_name)

    # txt/tsv
    text = file_bytes.decode("utf-8", errors="ignore")
    return pd.read_csv(StringIO(text), sep="\t" if "\t" in text else ",")


# -------------------------------------------------------
# ③ Excel の場合はシート選択
# -------------------------------------------------------
file_bytes = uploaded.getvalue()
file_name = uploaded.name.lower()

sheet_name = None
if file_name.endswith(".xlsx"):
    xls = pd.ExcelFile(uploaded)
    sheet_name = st.selectbox("📄 読み込むシートを選択", xls.sheet_names)

df = load_data(file_bytes, file_name, sheet_name)

# -------------------------------------------------------
# ④ Data Preview
# -------------------------------------------------------
st.subheader("📄 データ Preview")
st.dataframe(df.head())


# -------------------------------------------------------
# ⑤ 数値列選択
# -------------------------------------------------------
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

if not numeric_cols:
    st.error("数値列がありません（相関分析できません）。")
    st.stop()

selected_cols = st.multiselect(
    "分析する数値列を選択（複数選択）",
    numeric_cols,
    default=numeric_cols,
)

if len(selected_cols) < 2:
    st.error("2列以上選択してください。")
    st.stop()

numeric_df = df[selected_cols]


# -------------------------------------------------------
# ⑥ 欠損値処理（キャッシュに影響）
# -------------------------------------------------------
missing_strategy = st.radio(
    "欠損値の扱い",
    ["drop（行削除）", "mean（平均補完）", "zero（0埋め）"],
)


@st.cache_data
def preprocess_missing(df, strategy):
    if strategy.startswith("drop"):
        return df.dropna()
    elif strategy.startswith("mean"):
        return df.fillna(df.mean())
    else:
        return df.fillna(0)


numeric_df = preprocess_missing(numeric_df, missing_strategy)


# -------------------------------------------------------
# ⑦ 相関計算（キャッシュ）
# -------------------------------------------------------
method = st.selectbox("相関係数の種類", ["pearson", "spearman", "kendall"])


@st.cache_data
def compute_corr(df, method):
    return df.corr(method=method)


# 相関行列（2桁）
corr = compute_corr(numeric_df, method).round(2)

st.subheader(f"🔢 {method.upper()} 相関係数行列")
styled_corr = corr.style.format("{:.2f}").background_gradient(  # ← 小数点2桁に丸める
    cmap="RdBu_r"
)
st.dataframe(styled_corr)

# Plotly heatmap（2桁）
fig = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    aspect="auto",
)
st.plotly_chart(fig, use_container_width=True)

# ランキング（2桁）
corr_pairs = (
    corr.abs()
    .where(lambda x: x != 1.0)
    .stack()
    .sort_values(ascending=False)
    .round(2)  # ← ここで丸め
)
top_n = st.slider("表示件数", 5, 50, 10)

st.dataframe(
    corr_pairs.head(top_n)
    .reset_index()
    .rename(columns={"level_0": "変数1", "level_1": "変数2", 0: "相関係数（絶対値）"})
)
