import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from umap import UMAP
from io import StringIO
from app import memo_sidebar

# --------------------------------------------
# 📝 メモ
# --------------------------------------------
memo_sidebar()

st.title("🌀 T-SNE / UMAP 次元削減 + KMeans クラスタリング")

# -------------------------------------------------------
# ① ファイルアップロード
# -------------------------------------------------------
uploaded = st.file_uploader(
    "次元削減用の CSV / Excel をアップロード",
    type=["csv", "xlsx"],
    key="tsne_umap_uploader",
)

if not uploaded:
    st.info("ファイルをアップロードしてください。")
    st.stop()

# -------------------------------------------------------
# ② キャッシュ付き読み込み
# -------------------------------------------------------
@st.cache_data
def load_data(file_bytes, file_name, sheet=None):
    if file_name.endswith(".csv"):
        return pd.read_csv(StringIO(file_bytes.decode("utf-8", errors="ignore")))
    if file_name.endswith(".xlsx"):
        if sheet:
            return pd.read_excel(file_bytes, sheet_name=sheet)
        xls = pd.ExcelFile(file_bytes)
        return pd.read_excel(file_bytes, sheet_name=xls.sheet_names[0])
    text = file_bytes.decode("utf-8", errors="ignore")
    return pd.read_csv(StringIO(text), sep="\t" if "\t" in text else ",")

file_bytes = uploaded.getvalue()
file_name = uploaded.name.lower()

# -------------------------------------------------------
# ③ Excel シート対応
# -------------------------------------------------------
sheet_name = None
if file_name.endswith(".xlsx"):
    xls = pd.ExcelFile(uploaded)
    sheet_name = st.selectbox("📄 読み込むシートを選択", xls.sheet_names)

df = load_data(file_bytes, file_name, sheet_name)

# -------------------------------------------------------
# ④ Preview
# -------------------------------------------------------
st.subheader("📄 データPreview")
st.dataframe(df.head())

# -------------------------------------------------------
# ⑤ 数値列選択
# -------------------------------------------------------
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
if not numeric_cols:
    st.error("数値列が必要です。")
    st.stop()

selected_columns = st.multiselect(
    "次元削減に使用する数値列を選択してください",
    numeric_cols,
    default=numeric_cols,
)

if len(selected_columns) < 2:
    st.warning("2 列以上必要です。")
    st.stop()

numeric_df = df[selected_columns].dropna()
orig_index = numeric_df.index


# -------------------------------------------------------
# ⑥ 手法選択（T-SNE or UMAP）
# -------------------------------------------------------
method = st.selectbox(
    "次元削減手法",
    ["T-SNE", "UMAP"],
    index=0
)

# -------------------------------------------------------
# ⑦ クラスタ数（KMeans）
# -------------------------------------------------------
cluster_n = st.number_input(
    "クラスタ数 (KMeans)",
    min_value=2,
    max_value=10,
    value=3
)

# -------------------------------------------------------
# ⑧ パラメータ調整（詳細設定）
# -------------------------------------------------------
with st.expander("⚙️ 詳細設定（T-SNE / UMAP パラメータ）", expanded=False):

    n_samples = numeric_df.shape[0]

    if method == "T-SNE":
        max_perplexity = max(5, min(50, n_samples - 1))   # ← 自動上限
        perplexity = st.slider(
            "perplexity",
            min_value=5,
            max_value=max_perplexity,
            value=min(30, max_perplexity),
        )

        learning_rate = st.slider("learning_rate", 10, 1000, 200)
        tsne_dim = st.radio("次元数", [2, 3], index=0)

    else:  # UMAP
        n_neighbors = st.slider(
            "n_neighbors",
            min_value=5,
            max_value=min(100, n_samples - 1),   # ← 自動制限
            value=min(15, n_samples - 1),
        )
        min_dist = st.slider("min_dist", 0.0, 1.0, 0.1)
        umap_dim = st.radio("次元数", [2, 3], index=0)


# -------------------------------------------------------
# ⑨ 次元削減（キャッシュ）
# -------------------------------------------------------
@st.cache_data
def compute_tsne(df, dim, perplexity, learning_rate):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    tsne = TSNE(
        n_components=dim,
        perplexity=perplexity,
        learning_rate=learning_rate,
        random_state=42
    )
    return tsne.fit_transform(scaled), scaled


@st.cache_data
def compute_umap(df, dim, n_neighbors, min_dist):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    umap_model = UMAP(
        n_components=dim,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42
    )
    return umap_model.fit_transform(scaled), scaled


# -------------------------------------------------------
# ⑩ 実行ボタン
# -------------------------------------------------------
if st.button("🚀 次元削減 + クラスタリング実行"):

    # ------------------------------
    # T-SNE
    # ------------------------------
    if method == "T-SNE":
        embed, scaled = compute_tsne(numeric_df, tsne_dim, perplexity, learning_rate)
        dim = tsne_dim

    # ------------------------------
    # UMAP
    # ------------------------------
    else:
        embed, scaled = compute_umap(numeric_df, umap_dim, n_neighbors, min_dist)
        dim = umap_dim

    # 結果を DataFrame 化
    cols = [f"Dim{i+1}" for i in range(dim)]
    emb_df = pd.DataFrame(embed, columns=cols)

    # hover用に元データも追加
    emb_df["index"] = orig_index
    for col in selected_columns:
        emb_df[col] = numeric_df[col].values

    # ------------------------------
    # KMeans クラスタリング
    # ------------------------------
    km = KMeans(n_clusters=cluster_n, random_state=42, n_init="auto")
    clusters = km.fit_predict(embed)
    emb_df["cluster"] = clusters

    hover_cols = ["index"] + selected_columns

    # -------------------------------------------------------
    # 2D プロット（完全正方形）
    # -------------------------------------------------------
    if dim == 2:
        fig = px.scatter(
            emb_df,
            x="Dim1",
            y="Dim2",
            color="cluster",
            hover_data=hover_cols,
            title=f"{method} 2D + KMeans (K={cluster_n})",
        )
        fig.update_layout(
            width=600,
            height=600,
            xaxis=dict(scaleanchor="y", showgrid=True, zeroline=True),
            yaxis=dict(showgrid=True, zeroline=True),
        )
        st.plotly_chart(fig, width="content")

    # -------------------------------------------------------
    # 3D プロット
    # -------------------------------------------------------
    if dim == 3:
        fig3d = px.scatter_3d(
            emb_df,
            x="Dim1",
            y="Dim2",
            z="Dim3",
            color="cluster",
            hover_data=hover_cols,
            title=f"{method} 3D + KMeans (K={cluster_n})",
        )
        st.plotly_chart(fig3d, width="stretch")
