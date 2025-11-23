import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from lightgbm import LGBMClassifier
from io import StringIO
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from app import memo_sidebar
import pandas.api.types as ptypes

# ============================================
# 📝 メモ Sidebar
# ============================================
memo_sidebar()

# ============================================
# 🔧 状態管理
# ============================================
for key in ["model_trained", "shap_run", "cv_mode"]:
    if key not in st.session_state:
        st.session_state[key] = False

# ============================================
# タイトル
# ============================================
st.title("🔥 LightGBM 分類 + SHAP（単一 & CV）+ Interaction SHAP")


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

# 特徴量のカテゴリ自動エンコード
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
    "分類問題用 CSV / Excel をアップロード",
    type=["csv", "xlsx"],
    key="lgbm_clf_uploader",
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
# 🎯 ターゲット選択（分類問題）
# ============================================
target_col = st.selectbox("🎯 目的変数（クラス）", df.columns)

X_orig = df.drop(columns=[target_col])
y_raw = df[target_col]

# ID っぽい列は自動除外
id_cols = [c for c in X_orig.columns if c.lower() == "id" or c.lower().endswith("id")]
if id_cols:
    X_orig = X_orig.drop(columns=id_cols)
    st.warning(f"ID 系列 {id_cols} を特徴量から除外しました。")

# ターゲットをカテゴリ or 整数化
y = pd.Categorical(y_raw).codes


# 特徴量を LightGBM 用に数値化
X_num = encode_categoricals(X_orig)

# NaN は事前削除
combined = pd.concat([X_num, pd.Series(y, name=target_col)], axis=1)
n_before = len(combined)
combined = combined.dropna()
n_after = len(combined)

if n_after < n_before:
    st.warning(f"NaN を含む {n_before - n_after} 行を削除しました。")

X_num = combined[X_num.columns]
y = combined[target_col]
X_disp = X_orig.loc[X_num.index]  # 表示用（カテゴリ値）


st.write("📌 X shape:", X_num.shape)
st.write("📌 y shape:", y.shape)


# ============================================
# 🔧 LightGBM 分類モデル学習
# ============================================
def train_lgbm_classifier(X_train, y_train):
    model = LGBMClassifier(
        n_estimators=300,
        random_state=42,
        boosting_type="gbdt",
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model


# ============================================
# 📊 5-fold CV 性能
# ============================================
@st.cache_data
def compute_cv_metrics(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    acc_list = []
    pre_list = []
    rec_list = []
    f1_list = []
    auc_list = []

    for tr_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = LGBMClassifier(
            n_estimators=300,
            random_state=42,
            boosting_type="gbdt",
            class_weight="balanced"
        )
        model.fit(X_train, y_train)

        pred = model.predict(X_val)
        prob = model.predict_proba(X_val)[:, 1]

        acc_list.append(accuracy_score(y_val, pred))
        pre_list.append(precision_score(y_val, pred, zero_division=0))
        rec_list.append(recall_score(y_val, pred))
        f1_list.append(f1_score(y_val, pred))
        auc_list.append(roc_auc_score(y_val, prob))

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        "Fold1": [acc_list[0], pre_list[0], rec_list[0], f1_list[0], auc_list[0]],
        "Fold2": [acc_list[1], pre_list[1], rec_list[1], f1_list[1], auc_list[1]],
        "Fold3": [acc_list[2], pre_list[2], rec_list[2], f1_list[2], auc_list[2]],
        "Fold4": [acc_list[3], pre_list[3], rec_list[3], f1_list[3], auc_list[3]],
        "Fold5": [acc_list[4], pre_list[4], rec_list[4], f1_list[4], auc_list[4]],
        "Mean":  [np.mean(acc_list), np.mean(pre_list), np.mean(rec_list), np.mean(f1_list), np.mean(auc_list)],
        "Std":   [np.std(acc_list),  np.std(pre_list),  np.std(rec_list),  np.std(f1_list),  np.std(auc_list)],
    })

    return metrics_df


# ============================================
# 🚀 LightGBM モデル学習
# ============================================
if st.button("🚀 LightGBM モデル学習"):
    st.session_state["model"] = train_lgbm_classifier(X_num, y)
    st.session_state["model_trained"] = True
    st.success("LightGBM の学習が完了しました！")

    metrics_df = compute_cv_metrics(X_num, y, n_splits=5)

    st.subheader("📊 モデル性能（5-fold CV）")
    st.dataframe(metrics_df)


# ============================================
# 🔎 SHAP（分類）
# ============================================
@st.cache_data
def compute_shap_and_interactions(X, _model):
    explainer = shap.TreeExplainer(_model)

    # LightGBM 分類は shap_values がクラスごとに返る（list の形）
    shap_values = explainer.shap_values(X)  # List[F] or ndarray
    expected_value = explainer.expected_value

    # Interaction SHAP（クラス0のみを採用）
    interaction_values = explainer.shap_interaction_values(X)
    if isinstance(interaction_values, list):
        interaction_values = interaction_values[0]  # クラス0を使用

    return expected_value, shap_values, interaction_values


# ============================================
# SHAP ボタン
# ============================================
if st.button("📊 SHAP 計算（単一モデル）"):
    if not st.session_state["model_trained"]:
        st.warning("先に LightGBM を学習してください。")
    else:
        st.session_state["shap_run"] = True


# ============================================
# SHAP 表示
# ============================================
if st.session_state["shap_run"]:

    model = st.session_state["model"]

    expected_value, shap_values, interaction_values = compute_shap_and_interactions(
        X_num, model
    )

    st.header("📊 SHAP 解析（分類）")

    # SHAP Summary（クラス0の shap_values を採用）
    st.subheader("📌 SHAP Summary Plot（Class 0）")
    fig, ax = plt.subplots(figsize=(10, 5))
    # --- shap_values の正しい取り出し ---
    if isinstance(shap_values, list):
        # Binary classification → クラス1（positive）の SHAP を使う
        shap_matrix = shap_values[1]
    else:
        shap_matrix = shap_values

    # --- Summary Plot ---
    shap.summary_plot(shap_matrix, X_disp, show=False)
    st.pyplot(fig)
    plt.close(fig)

    # LightGBM Importance
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

    # Mean|SHAP|
    st.subheader("📈 Mean |SHAP| Feature Importance（Class 0）")
    shap_mean = np.abs(shap_matrix).mean(axis=0)
    shap_imp_df = pd.DataFrame({
        "Feature": X_num.columns,
        "Mean|SHAP|": shap_mean
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
    # Dependence Plot（分類）
    # ============================================
    st.subheader("📉 SHAP Dependence Plot（Class 0）")
    dep_feat = st.selectbox("特徴量（X軸）", X_num.columns)

    if dep_feat:
        fig_dep, ax_dep = plt.subplots(figsize=(7, 5))
        shap.dependence_plot(
            ind=dep_feat,
            shap_values=shap_matrix,
            features=X_disp,
            ax=ax_dep,
            show=False
        )
        st.pyplot(fig_dep)
        plt.close(fig_dep)

    # ============================================
    # Waterfall（分類）
    # ============================================
    st.subheader("📜 SHAP Waterfall Plot（個別, Class 0）")
    idx = st.number_input("行番号", 0, len(X_num)-1, 0)

    shap_ex = shap.Explanation(
        values=shap_matrix[idx],
        base_values=expected_value[1] if isinstance(expected_value, list) else expected_value,
        data=X_disp.iloc[idx],
        feature_names=X_disp.columns,
    )

    fig_w, ax_w = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(shap_ex, show=False)
    st.pyplot(fig_w)
    plt.close(fig_w)

    # ============================================
    # Interaction SHAP（分類）
    # ============================================
    st.header("🔀 Interaction SHAP（Class 0）")

    interaction_mean = np.abs(interaction_values).mean(axis=0)
    np.fill_diagonal(interaction_mean, 0)

    interaction_df = pd.DataFrame(
        interaction_mean,
        index=X_num.columns,
        columns=X_num.columns
    )
    st.subheader("📈 Interaction SHAP 行列")
    st.dataframe(interaction_df.style.background_gradient(cmap="RdBu_r"))

    fig_hm = px.imshow(
        interaction_df,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="SHAP Interaction Heatmap（Class 0）"
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
