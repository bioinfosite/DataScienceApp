import streamlit as st

from pages import (
    mito_page,
    sweetviz_page,
    pca_page,
    umap_page,
    corr_page,
    metrics_page,
    feature_importance_page,
)

st.set_page_config(layout="wide", page_title="データサイエンス EDA App")

PAGES = {
    "Mito分析": mito_page,
    "データプロファイリング": sweetviz_page,
    "PCA 次元削減": pca_page,
    "UMAP 次元削減": umap_page,
    "相関分析": corr_page,
    "誤差指標（回帰）": metrics_page,
    "Feature Importance（特徴量重要度）": feature_importance_page,
}

page = st.sidebar.selectbox("ページを選択してください", list(PAGES.keys()))
PAGES[page].run()
