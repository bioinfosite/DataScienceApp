import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import plotly.express as px

def run():
    st.title("🌟 Feature Importance（特徴量重要度）")

    uploaded = st.file_uploader(
        "CSV/Excel をアップロードしてください（ターゲット列を含む）",
        type=["csv", "xlsx"],
        key="fi_uploader"
    )

    if not uploaded:
        st.info("ファイルをアップロードしてください")
        return

    # データ読み込み
    df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)

    st.subheader("📄 データPreview")
    st.dataframe(df.head())

    # ターゲット列
    target_col = st.selectbox("ターゲット列（目的変数）を選択", df.columns)

    # 数値特徴量だけ使用
    numeric_df = df.select_dtypes(include=["int", "float"]).drop(columns=[target_col], errors="ignore")

    if numeric_df.empty:
        st.error("数値列が必要です。")
        return

    X = numeric_df
    y = df[target_col]

    st.write("📌 X shape:", X.shape)
    st.write("📌 y shape:", y.shape)

    # モデルタイプ
    mode = st.selectbox("分析タイプを選択", ["回帰 (Regression)", "分類 (Classification)"])

    if mode == "回帰 (Regression)":
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=42)

    model.fit(X, y)

    # Importance
    importance = model.feature_importances_
    imp_df = pd.DataFrame({"Feature": X.columns, "Importance": importance})
    imp_df = imp_df.sort_values(by="Importance", ascending=False)

    st.subheader("🔍 Feature Importance（重要度）")
    st.dataframe(imp_df)

    # Plot
    st.subheader("📊 可視化（Feature Importance）")
    fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h")
    st.plotly_chart(fig, use_container_width=True)
