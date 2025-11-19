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

    # NaN除外処理（両方のNaNをまとめて除外）
    valid_df = pd.DataFrame({"y": y, "y_pred": y_pred}).dropna().reset_index(drop=True)
    if len(valid_df) < len(df):
        st.warning(
            f"y, y_predにNaNが含まれているため {len(valid_df)} 件のみ計算に使用します。NaN行は除外されます。"
        )
    y_valid = valid_df["y"].astype(float)
    y_pred_valid = valid_df["y_pred"].astype(float)
    if y_valid.isna().sum() > 0 or y_pred_valid.isna().sum() > 0:
        st.error("NaNが残っています。データを確認してください。")
        return

    st.subheader("📊 計算結果")

    mae = mean_absolute_error(y_valid, y_pred_valid)
    mape = mean_absolute_percentage_error(y_valid, y_pred_valid)
    mse = mean_squared_error(y_valid, y_pred_valid)
    rmse = mse**0.5
    r2 = r2_score(y_valid, y_pred_valid)

    # 追加指標
    from sklearn.metrics import (
        median_absolute_error,
        explained_variance_score,
        max_error,
    )

    medae = median_absolute_error(y_valid, y_pred_valid)
    explained_var = explained_variance_score(y_valid, y_pred_valid)
    maxerr = max_error(y_valid, y_pred_valid)
    # Adjusted R2（自由度調整済み決定係数）
    n = len(y_valid)
    p = 1  # 単回帰の場合。多変量の場合は特徴量数に変更
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else None

    # ピアソン・スピアマン相関係数
    pearson_corr = y_valid.corr(y_pred_valid, method="pearson")
    spearman_corr = y_valid.corr(y_pred_valid, method="spearman")

    # SMAPE（対称平均絶対パーセント誤差）
    smape = (
        100 * (abs(y_valid - y_pred_valid) / ((abs(y_valid) + abs(y_pred_valid)) / 2))
    ).mean()

    # RMSLE（二乗平均平方対数誤差）
    import numpy as np

    rmsle = np.sqrt(mean_squared_error(np.log1p(y_valid), np.log1p(y_pred_valid)))

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
