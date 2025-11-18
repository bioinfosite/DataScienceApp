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
        key="pca_uploader",
    )

    if not uploaded:
        st.info("ファイルをアップロードしてください")
        return

    # データ読み込み
    df = (
        pd.read_csv(uploaded)
        if uploaded.name.endswith(".csv")
        else pd.read_excel(uploaded)
    )

    st.subheader("📄 データPreview")
    st.dataframe(df.head())

    # カラムを選択するUIを追加
    selected_columns = st.multiselect(
        "PCAに使用する数値列を選択してください",
        options=df.select_dtypes(include=["int", "float"]).columns.tolist(),
        default=df.select_dtypes(include=["int", "float"]).columns.tolist(),
    )

    if not selected_columns:
        st.warning("PCA対象のカラムを1つ以上選択してください。")
        return

    numeric_df = df[selected_columns].dropna()

    if numeric_df.empty:
        st.error("数値列が必要です。")
        return

    # 標準化・主成分数の+-ボタン部分はそのまま
    if "n_components" not in st.session_state:
        st.session_state["n_components"] = 2
    min_comp = 2
    max_comp = min(10, numeric_df.shape[1])
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if (
            st.button("-", key="pca_minus")
            and st.session_state["n_components"] > min_comp
        ):
            st.session_state["n_components"] -= 1
    with col2:
        if (
            st.button("+", key="pca_plus")
            and st.session_state["n_components"] < max_comp
        ):
            st.session_state["n_components"] += 1
    with col3:
        st.write(f"主成分数: {st.session_state['n_components']}")
    n_components = st.session_state["n_components"]

    # PCA実行ボタン
    if st.button("PCAを実行"):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(numeric_df)
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled)
        st.subheader("PCA 結果（主成分得点）")
        pca_df = pd.DataFrame(
            pca_result, columns=[f"PC{i+1}" for i in range(n_components)]
        )
        st.dataframe(pca_df)
        st.subheader("分散説明率")
        st.write(pca.explained_variance_ratio_)
        if n_components >= 2:
            fig = px.scatter(
                pca_df,
                x="PC1",
                y="PC2",
                title="PCA: PC1 vs PC2",
                labels={"PC1": "主成分1", "PC2": "主成分2"},
            )
            st.plotly_chart(fig)
        if n_components >= 3:
            fig3d = px.scatter_3d(
                pca_df,
                x="PC1",
                y="PC2",
                z="PC3",
                title="PCA: PC1 vs PC2 vs PC3",
                labels={"PC1": "主成分1", "PC2": "主成分2", "PC3": "主成分3"},
            )
            st.plotly_chart(fig3d)
        # クラスタ数選択
        st.subheader("クラスタリング（KMeans）")
        from sklearn.cluster import KMeans

        cluster_n = st.number_input(
            "クラスタ数 (K)", min_value=2, max_value=10, value=3
        )
        kmeans = KMeans(n_clusters=cluster_n, random_state=42)
        clusters = kmeans.fit_predict(pca_df)
        pca_df["cluster"] = clusters
        st.write("クラスタリング結果: クラスタごとに色分け")
        if n_components >= 2:
            fig = px.scatter(
                pca_df,
                x="PC1",
                y="PC2",
                color="cluster",
                title="PCA: PC1 vs PC2 (クラスタ色分け)",
                labels={"PC1": "主成分1", "PC2": "主成分2", "cluster": "クラスタ"},
            )
            st.plotly_chart(fig)
        if n_components >= 3:
            fig3d = px.scatter_3d(
                pca_df,
                x="PC1",
                y="PC2",
                z="PC3",
                color="cluster",
                title="PCA: PC1 vs PC2 vs PC3 (クラスタ色分け)",
                labels={
                    "PC1": "主成分1",
                    "PC2": "主成分2",
                    "PC3": "主成分3",
                    "cluster": "クラスタ",
                },
            )
            st.plotly_chart(fig3d)
