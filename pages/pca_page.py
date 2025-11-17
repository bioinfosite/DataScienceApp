import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

def run():
    st.title("🧩 PCA 次元削減")

    uploaded = st.file_uploader(
        "PCA用の CSV または Excel をアップロード",
        type=["csv", "xlsx"],
        accept_multiple_files=False,
        key="pca_uploader"
    )

    if not uploaded:
        st.info("ファイルをアップロードしてください")
        return

    # データ読み込み
    df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)

    st.subheader("📄 データPreview")
    st.dataframe(df.head())

    # 数値列のみ抽出
    numeric_df = df.select_dtypes(include=["int", "float"]).dropna()

    if numeric_df.empty:
        st.error("数値列が必要です。")
        return

    # 標準化
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)

    # PCA 次元数スライダー
    n_components = st.slider("主成分数", min_value=2, max_value=min(10, numeric_df.shape[1]), value=2)

    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(scaled)

    # 2D Plot
    st.subheader("📉 PCA 2次元プロット")

    df_plot = pd.DataFrame({
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1]
    })

    fig2 = px.scatter(df_plot, x="PC1", y="PC2")
    st.plotly_chart(fig2, use_container_width=True)

    # 3D Plot
    if n_components >= 3:
        st.subheader("🌐 PCA 3次元プロット")
        df_3d = pd.DataFrame({
            "PC1": pcs[:, 0],
            "PC2": pcs[:, 1],
            "PC3": pcs[:, 2],
        })
        fig3 = px.scatter_3d(df_3d, x="PC1", y="PC2", z="PC3")
        st.plotly_chart(fig3, use_container_width=True)

    # 寄与率
    st.subheader("📊 寄与率（Explained Variance Ratio）")
    st.write(pca.explained_variance_ratio_)
