import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def run():
    st.title("📈 相関分析（Correlation）")

    uploaded = st.file_uploader(
        "相関分析用の CSV/Excel をアップロード",
        type=["csv", "xlsx"],
        accept_multiple_files=False,
        key="corr_uploader"
    )

    if not uploaded:
        st.info("ファイルをアップロードしてください")
        return

    df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)

    st.subheader("📄 データPreview")
    st.dataframe(df.head())

    numeric_df = df.select_dtypes(include=["number"]).dropna()
    if numeric_df.empty:
        st.error("相関分析には数値列が必要です。")
        return

    method = st.selectbox("相関係数の種類", ["pearson", "spearman", "kendall"])

    corr = numeric_df.corr(method=method)

    st.subheader(f"🔢 {method.upper()} 相関係数行列")
    st.dataframe(corr)

    # Seaborn heatmap
    st.subheader("🔥 相関ヒートマップ（Seaborn）")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    # Plotly heatmap
    st.subheader("📊 相関ヒートmap（Plotly）")
    fig2 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
    st.plotly_chart(fig2, use_container_width=True)
