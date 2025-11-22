import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

st.title("🔗 相関分析（Correlation）")

uploaded = st.file_uploader(
    "相関分析用の CSV/Excel をアップロード",
    type=["csv", "xlsx"],
    accept_multiple_files=False,
    key="corr_uploader",
)

if uploaded is None:
    st.info("ファイルをアップロードしてください")
    st.stop()

df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)

st.subheader("🤮 データ Preview")
st.dataframe(df.head())

numeric_df = df.select_dtypes(include=["number"]).dropna()
if numeric_df.empty:
    st.error("相関分析には数値列が必要です。")
else:
    method = st.selectbox("相関係数の種類", ["pearson", "spearman", "kendall"])
    corr = numeric_df.corr(method=method)

    st.subheader(f"{method.upper()} 相関係数行列")
    st.dataframe(corr)

    st.subheader("🌡 相関ヒートmap (Plotly)")
    fig2 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
    st.plotly_chart(fig2, use_container_width=True)
