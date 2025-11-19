import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics import mutual_info_score
import plotly.express as px
import plotly.graph_objects as go


# -------------------------------------------------------
# 目的変数 vs 特徴量 の Mutual Information
# -------------------------------------------------------
def compute_mi_with_target(df, target_col, discrete_cols=None, n_neighbors=3):
    df2 = df.copy()

    y = df2[target_col]
    X = df2.drop(columns=[target_col])

    # カテゴリ推定
    if discrete_cols is None:
        discrete_cols = [c for c in X.columns
                         if X[c].dtype == 'object' or str(X[c].dtype).startswith('category')]

    # カテゴリ → LabelEncode
    for c in discrete_cols:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c].astype(str))

    # MI 計算（分類 or 回帰）
    discrete_mask = [c in discrete_cols for c in X.columns]

    if y.dtype == 'object' or str(y.dtype).startswith('category'):
        y_enc = LabelEncoder().fit_transform(y.astype(str))
        mi_vals = mutual_info_classif(
            X.values, y_enc,
            discrete_features=discrete_mask,
            n_neighbors=n_neighbors,
            random_state=0
        )
    else:
        mi_vals = mutual_info_regression(
            X.values, y.values,
            discrete_features=discrete_mask,
            n_neighbors=n_neighbors,
            random_state=0
        )

    return pd.Series(mi_vals, index=X.columns).sort_values(ascending=False)


# -------------------------------------------------------
# 特徴量同士の MI 行列
# -------------------------------------------------------
from sklearn.impute import SimpleImputer

def compute_mi_matrix(df, n_bins=10, strategy='quantile', max_cols=None):
    df2 = df.copy()

    # 列数が多い場合は制限
    if max_cols is not None and df2.shape[1] > max_cols:
        df2 = df2.iloc[:, :max_cols]

    X_disc = pd.DataFrame(index=df2.index)

    for c in df2.columns:
        col = df2[c]

        # --- 数値 ---
        if np.issubdtype(col.dtype, np.number):

            # NaN -> median で補完
            imputer = SimpleImputer(strategy="median")
            filled = imputer.fit_transform(col.values.reshape(-1, 1))

            # 離散化
            est = KBinsDiscretizer(
                n_bins=n_bins,
                encode='ordinal',
                strategy=strategy
            )
            X_disc[c] = est.fit_transform(filled).ravel()

        # --- カテゴリ ---
        else:
            # NaN を "Missing" に置換
            col_filled = col.astype(str).fillna("Missing")

            le = LabelEncoder()
            X_disc[c] = le.fit_transform(col_filled)

    cols = X_disc.columns
    n = len(cols)
    mi_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            mi = mutual_info_score(X_disc.iloc[:, i], X_disc.iloc[:, j])
            mi_mat[i, j] = mi
            mi_mat[j, i] = mi

    return pd.DataFrame(mi_mat, index=cols, columns=cols)



# -------------------------------------------------------
# Streamlit ページ本体
# -------------------------------------------------------
def run():
    st.title("🔍 Mutual Information (MI) 可視化ツール（Plotly版）")

    uploaded = st.file_uploader("CSVファイルをアップロード", type=["csv"])
    if uploaded is None:
        st.info("まず CSV ファイルをアップロードしてください。")
        return

    df = pd.read_csv(uploaded)
    st.write("### 📄 アップロードしたデータ")
    st.dataframe(df.head())

    # 目的変数
    target_col = st.selectbox("🎯 目的変数を選択", [None] + list(df.columns))

    # カテゴリとして扱う列
    discrete_cols = st.multiselect(
        "カテゴリとして扱いたい列を指定（任意）",
        df.columns.tolist()
    )

    n_neighbors = st.slider("近傍数 (MI with target)", 3, 20, 5)

    # -------- MI with Target --------
    if target_col:
        st.markdown("## 📌 目的変数との Mutual Information")

        mi_series = compute_mi_with_target(
            df,
            target_col,
            discrete_cols=discrete_cols,
            n_neighbors=n_neighbors
        )

        # Plotly Bar Chart
        fig = px.bar(
            mi_series,
            orientation='h',
            title=f"Mutual Information with target: {target_col}",
            labels={'value': 'MI', 'index': 'Feature'}
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(mi_series.to_frame("MI").style.format("{:.4f}"))

    # -------- MI Matrix --------
    st.markdown("## 🔥 特徴量同士の Mutual Information (MI) 行列")

    n_bins = st.slider("連続値のビン数", 2, 50, 10)
    strategy = st.selectbox("離散化方法", ["quantile", "uniform", "kmeans"])
    max_cols = st.number_input(
        "最大列数（ヒートマップ計算用）", 
        min_value=2,
        max_value=len(df.columns),
        value=min(20, len(df.columns)),
        step=1
    )

    features_df = df.drop(columns=[target_col]) if target_col else df
    mi_df = compute_mi_matrix(
        features_df,
        n_bins=n_bins,
        strategy=strategy,
        max_cols=int(max_cols)
    )

    # Plotly Heatmap
    heatmap = go.Figure(
        data=go.Heatmap(
            z=mi_df.values,
            x=mi_df.columns,
            y=mi_df.columns,
            colorscale="Viridis",
            colorbar=dict(title="MI")
        )
    )
    heatmap.update_layout(
        title="MI Heatmap (Feature × Feature)",
        xaxis_nticks=mi_df.shape[1]
    )

    st.plotly_chart(heatmap, use_container_width=True)

    st.dataframe(mi_df.style.format("{:.4f}"))

    # CSV ダウンロード
    st.download_button(
        "📥 MI 行列をダウンロード (CSV)",
        mi_df.to_csv(),
        file_name="mi_matrix.csv",
        mime="text/csv"
    )


