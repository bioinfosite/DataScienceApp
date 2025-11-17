import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import plotly.express as px

def run():
    st.title("🌈 UMAP 次元削減")

    uploaded = st.file_uploader(
        "UMAP用の CSV または Excel をアップロード",
        type=["csv", "xlsx"],
        accept_multiple_files=False,
        key="umap_uploader"
    )

    if not uploaded:
        st.info("ファイルをアップロードしてください")
        return

    # データ読み込み
    df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)

    st.subheader("📄 データPreview")
    st.dataframe(df.head())

    numeric_df = df.select_dtypes(include=["int", "float"]).dropna()
    if numeric_df.empty:
        st.error("数値列が必要です。")
        return

    # 標準化
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)

    # パラメータ
    n_neighbors = st.slider("n_neighbors", 5, 100, value=15)
    min_dist = st.slider("min_dist", 0.0, 1.0, value=0.1)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=42
    )

    embedding = reducer.fit_transform(scaled)

    # 2D Plot
    st.subheader("🌈 UMAP 2D プロット")

    df_plot = pd.DataFrame({
        "UMAP1": embedding[:, 0],
        "UMAP2": embedding[:, 1],
    })

    fig = px.scatter(df_plot, x="UMAP1", y="UMAP2")
    st.plotly_chart(fig, use_container_width=True)
