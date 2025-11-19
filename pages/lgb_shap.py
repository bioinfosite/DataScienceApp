import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor, LGBMClassifier


def run():
    st.title("🌟 LightGBM Feature Importance & SHAP 解析")

    uploaded = st.file_uploader(
        "CSV/Excel をアップロードしてください（ターゲット列を含む）",
        type=["csv", "xlsx"],
        key="lgbm_shap_uploader",
    )

    if not uploaded:
        st.info("ファイルをアップロードしてください")
        return

    # データ読み込み
    df = (
        pd.read_csv(uploaded)
        if uploaded.name.endswith(".csv")
        else pd.read_excel(uploaded)
    )

    st.subheader("📄 データPreview")
    st.dataframe(df.head())

    # 目的変数
    target_col = st.selectbox("ターゲット列を選択", df.columns)

    # 説明変数
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # LightGBM が扱えるようにカテゴリを category 型に変換
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype("category")

    st.write("📌 X shape:", X.shape)
    st.write("📌 y shape:", y.shape)

    # 分析タイプ
    mode = st.selectbox(
        "分析タイプを選択", ["回帰 (Regression)", "分類 (Classification)"]
    )

    if mode == "回帰 (Regression)":
        model = LGBMRegressor(n_estimators=300, random_state=42, boosting_type="gbdt")
    else:
        model = LGBMClassifier(n_estimators=300, random_state=42, boosting_type="gbdt")

    # モデル学習ボタン
    if st.button("モデル学習"):
        model.fit(X, y)
        st.success("モデル学習が完了しました")
        st.session_state["lgbm_model"] = model
        st.session_state["X_lgbm"] = X
        st.session_state["mode_lgbm"] = mode

    # SHAP解析ボタン
    if "lgbm_model" in st.session_state and st.button("SHAP解析"):
        model = st.session_state["lgbm_model"]
        X = st.session_state["X_lgbm"]
        mode = st.session_state["mode_lgbm"]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        st.subheader("SHAP Summary Plot")
        plt.figure(figsize=(10, 5))
        if mode == "分類 (Classification)" and isinstance(shap_values, list):
            shap.summary_plot(shap_values[0], X, show=False)
        else:
            shap.summary_plot(shap_values, X, show=False)
        st.pyplot(plt)

        # ----------------------------------------
        # 🔥 Feature Importance（LightGBM）
        # ----------------------------------------
        importance = model.feature_importances_
        imp_df = pd.DataFrame({"Feature": X.columns, "Importance": importance})
        imp_df = imp_df.sort_values(by="Importance", ascending=False)

        # Plot
        fig = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="LightGBM Feature Importance",
        )
        st.plotly_chart(fig, width="stretch")

        # ----------------------------------------
        # 🔥 SHAP 解析
        # ----------------------------------------
        st.subheader("✨ SHAP（Shapley Additive Explanations）解析")

        with st.spinner("SHAP 値を計算しています…（数秒かかる場合があります）"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

        st.success("SHAP 計算完了！")

        # ----------------------------------------
        # 📊 SHAP Summary Plot（全体）
        # ----------------------------------------

        st.markdown("### 🔷 SHAP Summary Plot（全体の寄与）")

        # summary plot を図として保存 → Streamlit で表示
        shap_fig = shap.summary_plot(shap_values, X, show=False, plot_type="dot")
        st.pyplot(bbox_inches="tight")

        # ----------------------------------------
        # 📈 SHAP Bar Plot
        # ----------------------------------------
        st.markdown("### 🔷 SHAP Bar Plot（平均絶対 SHAP）")

        if mode == "分類 (Classification)":
            shap_importance = np.abs(shap_values).mean(axis=0)
        else:
            shap_importance = np.abs(shap_values).mean(axis=0)

        shap_imp_df = pd.DataFrame(
            {"Feature": X.columns, "Mean|SHAP|": shap_importance}
        ).sort_values("Mean|SHAP|", ascending=False)

        fig2 = px.bar(
            shap_imp_df,
            x="Mean|SHAP|",
            y="Feature",
            orientation="h",
            title="SHAP Feature Importance",
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ----------------------------------------
        # 🔍 SHAP Dependence Plot（今回追加）
        # ----------------------------------------
        st.subheader("📈 SHAP Dependence Plot")

        col1, col2 = st.columns(2)
        with col1:
            dep_feat = st.selectbox("x軸に使う特徴量（必須）", X.columns)
        with col2:
            interaction_feat = st.selectbox(
                "色付け・相互作用に使う特徴量（任意）", ["(自動選択)"] + list(X.columns)
            )

        if dep_feat:
            plt.figure(figsize=(7, 5))
            if interaction_feat == "(自動選択)":
                shap.dependence_plot(
                    dep_feat, shap_values, X, interaction_index="auto", show=False
                )
            else:
                shap.dependence_plot(
                    dep_feat, shap_values, X, interaction_index=interaction_feat, show=False
                )

            st.pyplot(plt.gcf(), clear_figure=True)

        # ----------------------------------------
        # 🔍 個別 SHAP（ウォーターフォール）
        # ----------------------------------------
        st.subheader("📌 SHAP 個別予測のウォーターフォール Plot")

        idx = st.number_input(
            "表示する行番号 (0〜)", min_value=0, max_value=len(X) - 1, value=0
        )

        st.write(f"選択した行のデータ:")
        st.write(X.iloc[idx : idx + 1])

        # Force plot の代わりに waterfall plot を使用
        st.markdown("### 🔷 SHAP Waterfall Plot")
        shap_fig = shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[idx],
                base_values=explainer.expected_value,
                data=X.iloc[idx],
                feature_names=X.columns,
            ),
            show=False,
        )
        st.pyplot(bbox_inches="tight")


# Streamlit 互換
if __name__ == "__main__":
    run()
