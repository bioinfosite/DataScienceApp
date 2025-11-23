import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from lightgbm import LGBMRegressor
from io import StringIO
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from app import memo_sidebar
import pandas.api.types as ptypes


# ============================================
# 📝 メモ Sidebar
# ============================================
memo_sidebar()


# ============================================
# 🔧 状態管理
# ============================================
if "model_trained" not in st.session_state:
    st.session_state["model_trained"] = False
if "shap_run" not in st.session_state:
    st.session_state["shap_run"] = False
if "cv_mode" not in st.session_state:
    st.session_state["cv_mode"] = False


# ============================================
# タイトル
# ============================================
st.title("🌟 LightGBM 回帰 + SHAP（単一 & CV）+ Interaction SHAP")


# ============================================
# 🔧 データ読み込み（キャッシュ）
# ============================================
@st.cache_data
def load_data(file_bytes, name, sheet=None):
    if name.endswith(".csv"):
        return pd.read_csv(StringIO(file_bytes.decode("utf-8", errors="ignore")))
    if name.endswith(".xlsx"):
        if sheet:
            return pd.read_excel(file_bytes, sheet_name=sheet)
        xls = pd.ExcelFile(file_bytes)
        return pd.read_excel(file_bytes, sheet_name=xls.sheet_names[0])
    return pd.read_csv(StringIO(file_bytes.decode("utf-8", errors="ignore")))


# 特徴量側のカテゴリ・文字列を数値にエンコード
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for col in df2.columns:
        if df2[col].dtype == "object" or ptypes.is_categorical_dtype(df2[col]):
            df2[col] = df2[col].astype("category").cat.codes
    return df2


# ============================================
# 🔼 ファイルアップロード
# ============================================
uploaded = st.file_uploader(
    "回帰分析用 CSV / Excel をアップロード",
    type=["csv", "xlsx"],
    key="lgbm_shap_uploader",
)

if uploaded is None:
    st.info("ファイルをアップロードしてください。")
    st.stop()

file_bytes = uploaded.getvalue()
file_name = uploaded.name.lower()

sheet_name = None
if file_name.endswith(".xlsx"):
    xls = pd.ExcelFile(uploaded)
    sheet_name = st.selectbox("📄 シート選択", xls.sheet_names)

df = load_data(file_bytes, file_name, sheet_name)

st.subheader("📄 データ Preview")
st.dataframe(df.head())


# ============================================
# 🎯 目的変数の選択
# ============================================
target_col = st.selectbox("🎯 目的変数（ターゲット）", df.columns)

# 元の特徴量（表示用・Waterfall / summary 用）
X_orig = df.drop(columns=[target_col])
y_raw = df[target_col]

# ID っぽい列は特徴量から自動除外（CustomerID など）
id_cols = [c for c in X_orig.columns if c.lower() == "id" or c.lower().endswith("id")]
if id_cols:
    X_orig = X_orig.drop(columns=id_cols)
    st.warning(f"ID 系列 {id_cols} を特徴量から除外しました。")

# 目的変数は必ず数値にキャスト
try:
    y = pd.to_numeric(y_raw, errors="raise")
except Exception:
    st.error("ターゲット列は数値型である必要があります（回帰）。別の列を選択してください。")
    st.stop()

# 特徴量を数値に変換（モデリング & SHAP 用）
X_num = encode_categoricals(X_orig)

# X, y をまとめて NaN を削除（ここで行数を最終確定）
combined = pd.concat([X_num, y], axis=1)
n_before = len(combined)
combined = combined.dropna()
n_after = len(combined)

if n_after < n_before:
    st.warning(f"NaN を含む {n_before - n_after} 行を削除しました（学習対象 {n_after} 行）。")

X_num = combined.drop(columns=[target_col])
y = combined[target_col]
# 表示用の元データも index を揃える
X_disp = X_orig.loc[X_num.index]

st.write("📌 X shape (numeric for model/SHAP):", X_num.shape)
st.write("📌 y shape:", y.shape)


# ============================================
# 🔧 LightGBM 学習
# ============================================
def train_lgbm_model(X_train, y_train):
    model = LGBMRegressor(
        n_estimators=300,
        random_state=42,
        boosting_type="gbdt",
    )
    model.fit(X_train, y_train)
    return model


# ============================================
# 📊 5-fold CV で性能評価（RMSE / MAE / R²）
# ============================================
@st.cache_data
def compute_cv_metrics(X: pd.DataFrame, y: pd.Series, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    rmse_list = []
    mae_list = []
    r2_list = []

    for tr_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = LGBMRegressor(
            n_estimators=300,
            random_state=42,
            boosting_type="gbdt",
        )
        model.fit(X_train, y_train)

        pred = model.predict(X_val)

        rmse = root_mean_squared_error(y_val, pred)
        mae = mean_absolute_error(y_val, pred)
        r2 = r2_score(y_val, pred)

        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)

    metrics_df = pd.DataFrame({
        "Metric": ["RMSE", "MAE", "R2"],
        "Fold1": [rmse_list[0], mae_list[0], r2_list[0]],
        "Fold2": [rmse_list[1], mae_list[1], r2_list[1]],
        "Fold3": [rmse_list[2], mae_list[2], r2_list[2]],
        "Fold4": [rmse_list[3], mae_list[3], r2_list[3]],
        "Fold5": [rmse_list[4], mae_list[4], r2_list[4]],
        "Mean":  [np.mean(rmse_list), np.mean(mae_list), np.mean(r2_list)],
        "Std":   [np.std(rmse_list),  np.std(mae_list),  np.std(r2_list)],
    })

    return metrics_df


# ============================================
# 🚀 LightGBM モデル学習
# ============================================
if st.button("🚀 LightGBM モデル学習"):
    st.session_state["model"] = train_lgbm_model(X_num, y)
    st.session_state["model_trained"] = True
    st.success("LightGBM の学習が完了しました！")

    metrics_df = compute_cv_metrics(X_num, y, n_splits=5)

    st.subheader("📊 モデル性能（5-fold CV）")
    st.dataframe(metrics_df)

    csv = metrics_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 CVモデル性能を CSV でダウンロード",
        data=csv,
        file_name="cv_model_performance.csv",
        mime="text/csv",
    )


# ============================================
# 🔎 SHAP（単一モデル + Interaction）
# ============================================
@st.cache_data
def compute_shap_and_interactions(X: pd.DataFrame, _model: LGBMRegressor):
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X)              # (N, F)
    expected_value = explainer.expected_value
    interaction_values = explainer.shap_interaction_values(X)  # (N, F, F)
    return expected_value, shap_values, interaction_values


# ============================================
# 🔎 SHAP（5-fold Cross Validation）
# ============================================
@st.cache_data
def compute_cv_shap_values(X: pd.DataFrame, y: pd.Series, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    shap_folds = []

    for tr_idx, val_idx in kf.split(X):
        model = LGBMRegressor(
            n_estimators=300,
            random_state=42,
            boosting_type="gbdt",
        )
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)  # 全サンプルで計算
        shap_folds.append(shap_values)

    shap_array = np.stack(shap_folds, axis=0)  # (K, N, F)
    shap_mean = shap_array.mean(axis=0)        # (N, F)
    shap_std = shap_array.std(axis=0)          # (N, F)

    return shap_mean, shap_std


# ============================================
# SHAP ボタン
# ============================================
colA, colB = st.columns(2)

with colA:
    if st.button("📊 単一モデル SHAP"):
        if not st.session_state["model_trained"]:
            st.warning("先に LightGBM を学習してください。")
        else:
            st.session_state["shap_run"] = True
            st.session_state["cv_mode"] = False

with colB:
    if st.button("📉 5-fold CV SHAP（Mean ± Std）"):
        if not st.session_state["model_trained"]:
            st.warning("先に LightGBM を学習してください。")
        else:
            st.session_state["shap_run"] = True
            st.session_state["cv_mode"] = True


# ============================================
# SHAP 表示ブロック
# ============================================
if st.session_state["shap_run"]:

    # -------------------------------
    # 5-fold CV SHAP（Mean ± Std）
    # -------------------------------
    if st.session_state["cv_mode"]:

        st.header("📉 SHAP（5-fold CV）: Mean ± Std")

        shap_mean, shap_std = compute_cv_shap_values(X_num, y, n_splits=5)

        shap_df = pd.DataFrame({
            "Feature": X_num.columns,
            "Mean|SHAP|": np.abs(shap_mean).mean(axis=0),
            "Std|SHAP|":  np.abs(shap_std).mean(axis=0),
        }).sort_values("Mean|SHAP|", ascending=False)

        fig_cv = px.bar(
            shap_df,
            x="Mean|SHAP|",
            y="Feature",
            error_x="Std|SHAP|",
            orientation="h",
            title="SHAP Importance（Mean ± Std, 5-fold CV）",
        )
        fig_cv.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig_cv, width="stretch")

        st.stop()


    # -------------------------------
    # 単一モデル SHAP
    # -------------------------------
    model: LGBMRegressor = st.session_state["model"]

    expected_value, shap_values, interaction_values = compute_shap_and_interactions(
        X_num, model
    )

    st.header("📊 SHAP 解析（単一モデル）")

    # --- Summary Plot ---
    st.subheader("📌 SHAP Summary Plot")
    fig, ax = plt.subplots(figsize=(10, 5))
    # 表示には元の X（カテゴリ付き）を使うが、行数・列数は X_num と一致している
    shap.summary_plot(shap_values, X_disp, show=False)
    st.pyplot(fig)
    plt.close(fig)

    # --- LightGBM Importance ---
    st.subheader("🔥 LightGBM Feature Importance")
    imp_df = (
        pd.DataFrame({"Feature": X_num.columns, "Importance": model.feature_importances_})
        .sort_values(by="Importance", ascending=False)
    )
    fig_imp = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="LightGBM Feature Importance",
    )
    fig_imp.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig_imp, width="stretch")

    # --- Mean SHAP ---
    st.subheader("📈 Mean |SHAP| Feature Importance")
    shap_mean_single = np.abs(shap_values).mean(axis=0)
    shap_imp_df = pd.DataFrame({
        "Feature": X_num.columns,
        "Mean|SHAP|": shap_mean_single
    }).sort_values("Mean|SHAP|", ascending=False)

    fig_shap = px.bar(
        shap_imp_df,
        x="Mean|SHAP|",
        y="Feature",
        orientation="h",
        title="SHAP Feature Importance（平均絶対値）",
    )
    fig_shap.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig_shap, width="stretch")

    # ============================================
    # 📉 SHAP Dependence Plot（通常版）
    # ============================================

    st.subheader("📉 SHAP Dependence Plot（通常版）")

    dep_feat = st.selectbox("特徴量を選択（X軸）", X_num.columns)

    if dep_feat:
        fig_dep, ax_dep = plt.subplots(figsize=(7, 5))

        # 通常版の dependence plot（SHAP が自動で interaction 相手を選ぶ）
        shap.dependence_plot(
            ind=dep_feat,
            shap_values=shap_values,
            features=X_disp,   # 表示用（カテゴリ値など）
            ax=ax_dep,
            show=False
        )

        st.pyplot(fig_dep)
        plt.close(fig_dep)

    # ============================================
    # Waterfall Plot
    # ============================================
    st.subheader("📜 SHAP Waterfall Plot（個別）")
    idx = st.number_input("行番号", 0, len(X_num)-1, 0)

    st.dataframe(X_disp.iloc[idx:idx+1])

    shap_ex = shap.Explanation(
        values=shap_values[idx],
        base_values=expected_value,
        data=X_disp.iloc[idx],
        feature_names=X_disp.columns,
    )

    fig_w, ax_w = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(shap_ex, show=False)
    st.pyplot(fig_w)
    plt.close(fig_w)

    # ============================================
    # 🔀 SHAP Interaction Values（相互作用）
    # ============================================
    st.header("🔀 SHAP Interaction Values")

    interaction_mean = np.abs(interaction_values).mean(axis=0)
    np.fill_diagonal(interaction_mean, 0)

    interaction_df = pd.DataFrame(
        interaction_mean,
        index=X_num.columns,
        columns=X_num.columns
    )

    st.subheader("📈 Interaction SHAP 行列")
    st.dataframe(interaction_df.style.background_gradient(cmap="RdBu_r"))

    st.subheader("🔥 Interaction Heatmap")
    fig_hm = px.imshow(
        interaction_df,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="SHAP Interaction Heatmap"
    )
    st.plotly_chart(fig_hm, width="stretch")

    st.subheader("🏆 相互作用ランキング Top 20")
    pairs = []
    cols = X_num.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            pairs.append((cols[i], cols[j], interaction_mean[i, j]))

    top_df = (
        pd.DataFrame(pairs, columns=["Feature A", "Feature B", "Mean|Interaction|"])
        .sort_values("Mean|Interaction|", ascending=False)
        .head(20)
    )
    st.dataframe(top_df)

    # st.subheader("📉 Interaction Dependence Plot")
    # feat_x = st.selectbox("X軸の特徴量", X_num.columns)
    # feat_y = st.selectbox("相互作用させる特徴量", X_num.columns)

    # if feat_x and feat_y:
    #     fig2, ax2 = plt.subplots(figsize=(7, 5))
    #     shap.dependence_plot(
    #         feat_x,
    #         interaction_values,
    #         X_disp,  # 行数・列数は X_num と完全一致している
    #         interaction_index=feat_y,
    #         ax=ax2,
    #         show=False
    #     )
    #     st.pyplot(fig2)
    #     plt.close(fig2)
