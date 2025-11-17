import streamlit as st
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)

def run():
    st.title("📉 誤差指標（回帰）")

    uploaded = st.file_uploader(
        "実績値（y）と予測値（y_pred）を含む CSV/Excel をアップロード",
        type=["csv", "xlsx"],
        key="metrics_uploader"
    )

    if not uploaded:
        st.info("ファイルをアップロードしてください")
        return

    df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)

    st.subheader("📄 データPreview")
    st.dataframe(df.head())

    y_col = st.selectbox("実績値 (y)", df.columns)
    y_pred_col = st.selectbox("予測値 (y_pred)", df.columns)

    y = df[y_col]
    y_pred = df[y_pred_col]

    st.subheader("📊 計算結果")

    mae = mean_absolute_error(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y, y_pred)

    st.write(f"**MAE:** {mae:.4f}")
    st.write(f"**MAPE:** {mape:.4f}")
    st.write(f"**MSE:** {mse:.4f}")
    st.write(f"**RMSE:** {rmse:.4f}")
    st.write(f"**R²:** {r2:.4f}")
