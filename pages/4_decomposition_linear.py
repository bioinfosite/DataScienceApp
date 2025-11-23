import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from io import StringIO
from app import memo_sidebar

# --------------------------------------------
# 📝 メモ
# --------------------------------------------
memo_sidebar()

st.title("🧩 PCA")


# -------------------------------------------------------
# ① ファイルアップロード
# -------------------------------------------------------
uploaded = st.file_uploader(
    "PCA用の CSV / Excel をアップロード",
    type=["csv", "xlsx"],
    key="pca_uploader",
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
    st.error("数値列がありません。PCAを実行できません。")
    st.stop()

selected_columns = st.multiselect(
    "PCAに使用する数値列を選択してください",
    numeric_cols,
    default=numeric_cols,
)

if len(selected_columns) < 2:
    st.warning("2 列以上選択してください。")
    st.stop()

numeric_df = df[selected_columns].dropna()
orig_index = numeric_df.index


# -------------------------------------------------------
# ⑥ 主成分数 UI（number_input）
# -------------------------------------------------------
min_comp = 2
max_comp = min(10, numeric_df.shape[1])

n_components = st.number_input(
    "主成分数（PCA components）",
    min_value=min_comp,
    max_value=max_comp,
    value=min_comp,
    step=1
)


# -------------------------------------------------------
# ⑦ クラスタ数 K 指定
# -------------------------------------------------------
cluster_n = st.number_input(
    "クラスタ数 (KMeans)",
    min_value=2,
    max_value=10,
    value=3
)


# -------------------------------------------------------
# ⭐ ⑧ エルボープロット表示オプション
# -------------------------------------------------------
show_elbow = st.checkbox("📉 エルボープロット（主成分数と累積寄与率）を表示する")


# -------------------------------------------------------
# ⑨ PCA + KMeans 計算（キャッシュ）
# -------------------------------------------------------
@st.cache_data
def compute_pca_kmeans(df, n_components, cluster_n):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled)

    pca_df = pd.DataFrame(
        pca_result,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    km = KMeans(n_clusters=cluster_n, random_state=42, n_init="auto")
    clusters = km.fit_predict(pca_df)

    pca_df["cluster"] = clusters
    return pca, pca_df, scaled


# -------------------------------------------------------
# ⑩ エルボープロット（累積寄与率）キャッシュ
# -------------------------------------------------------
@st.cache_data
def compute_elbow_cumulative(df, max_pcs):
    cum_vars = []
    pcs = list(range(1, max_pcs + 1))

    for pc in pcs:
        pca_tmp = PCA(n_components=pc)
        pca_tmp.fit(df)
        cum_vars.append(pca_tmp.explained_variance_ratio_.sum())

    return pcs, cum_vars


# -------------------------------------------------------
# ⑪ 実行ボタン
# -------------------------------------------------------
if st.button("⚙️ PCA + クラスタリング実行"):

    pca, pca_df, scaled_data = compute_pca_kmeans(numeric_df, n_components, cluster_n)

    # hover用に元データを統合
    pca_df["index"] = orig_index
    for col in selected_columns:
        pca_df[col] = numeric_df[col].values

    hover_cols = ["index"] + selected_columns


    # -------------------------------------------------------
    # ⭐ 累積寄与率のエルボープロット
    # -------------------------------------------------------
    if show_elbow:
        st.subheader("📉 エルボープロット（主成分数 vs 累積寄与率）")
        max_pcs = min(10, numeric_df.shape[1])
        pcs, cum_vars = compute_elbow_cumulative(numeric_df, max_pcs)

        fig_elbow = px.line(
            x=pcs,
            y=cum_vars,
            markers=True,
            title="Elbow Plot（主成分数 vs 累積寄与率）",
            labels={"x": "主成分数", "y": "累積寄与率"},
        )
        fig_elbow.update_yaxes(range=[0, 1.05])
        st.plotly_chart(fig_elbow, width="stretch")


    # -------------------------------------------------------
    # ⭐ PCA 2D（正方形 600×600 + グリッド線）
    # -------------------------------------------------------
    if n_components >= 2:
        fig = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="cluster",
            hover_data=hover_cols,
            title=f"PCA (PC1 vs PC2) + KMeans (K={cluster_n})",
        )

        # ★ 完全正方形 & グリッド線
        fig.update_layout(
            width=600,
            height=600,
            xaxis=dict(
                scaleanchor="y",
                scaleratio=1,
                showgrid=True,
                zeroline=True
            ),
            yaxis=dict(
                scaleratio=1,
                showgrid=True,
                zeroline=True
            ),
        )

        st.plotly_chart(fig, width="content")  # ★ stretchを禁止


    # -------------------------------------------------------
    # PCA 3D（通常表示）
    # -------------------------------------------------------
    if n_components >= 3:
        fig3d = px.scatter_3d(
            pca_df,
            x="PC1",
            y="PC2",
            z="PC3",
            color="cluster",
            hover_data=hover_cols,
            title=f"PCA 3D + KMeans (K={cluster_n})",
        )
        st.plotly_chart(fig3d, width="stretch")
