import streamlit as st
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def run():
    st.title("📉 誤差指標（回帰）")

    uploaded = st.file_uploader(
        "実績値（y）と予測値（y_pred）を含む CSV/Excel をアップロード",
        type=["csv", "xlsx"],
        key="metrics_uploader",
    )

    if not uploaded:
        st.info("ファイルをアップロードしてください")
        return

    df = (
        pd.read_csv(uploaded)
        if uploaded.name.endswith(".csv")
        else pd.read_excel(uploaded)
    )

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
    rmse = mse**0.5
    r2 = r2_score(y, y_pred)

    # 追加指標
    from sklearn.metrics import (
        median_absolute_error,
        explained_variance_score,
        max_error,
    )

    medae = median_absolute_error(y, y_pred)
    explained_var = explained_variance_score(y, y_pred)
    maxerr = max_error(y, y_pred)
    # Adjusted R2（自由度調整済み決定係数）
    n = len(y)
    p = 1  # 単回帰の場合。多変量の場合は特徴量数に変更
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else None

    # ピアソン・スピアマン相関係数
    pearson_corr = y.corr(y_pred, method="pearson")
    spearman_corr = y.corr(y_pred, method="spearman")

    # SMAPE（対称平均絶対パーセント誤差）
    smape = (100 * (abs(y - y_pred) / ((abs(y) + abs(y_pred)) / 2))).mean()

    # RMSLE（二乗平均平方対数誤差）
    import numpy as np

    rmsle = np.sqrt(mean_squared_error(np.log1p(y), np.log1p(y_pred)))

    # 指標を辞書でまとめる
    metrics_dict = {
        "MAE (平均絶対誤差)": mae,
        "MAPE (平均絶対パーセント誤差)": mape,
        "SMAPE (対称平均絶対パーセント誤差)": smape,
        "MSE (平均二乗誤差)": mse,
        "RMSE (二乗平均平方根誤差)": rmse,
        "RMSLE (二乗平均平方対数誤差)": rmsle,
        "R2 (決定係数)": r2,
        "Adjusted R2 (自由度調整済み決定係数)": adj_r2,
        "Median Absolute Error (中央値絶対誤差)": medae,
        "Explained Variance (説明分散スコア)": explained_var,
        "Max Error (最大絶対誤差)": maxerr,
        "Pearson Correlation (ピアソン相関係数)": pearson_corr,
        "Spearman Correlation (スピアマン相関係数)": spearman_corr,
    }

    # 表形式で表示
    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["指標", "値"])
    st.subheader("📊 指標一覧")
    st.dataframe(metrics_df, width="stretch")

    # ダウンロードボタン
    csv_data = metrics_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="指標一覧をCSVでダウンロード",
        data=csv_data,
        file_name="metrics_result.csv",
        mime="text/csv",
    )
