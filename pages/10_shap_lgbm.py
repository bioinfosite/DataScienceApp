import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from lightgbm import LGBMRegressor, LGBMClassifier

# --- Page Title ---
st.title("\u272f LightGBM Feature Importance & SHAP Analysis")

# --- File Upload ---
uploaded = st.file_uploader(
    "Upload a CSV/Excel file (must include target column)",
    type=["csv", "xlsx"],
    key="lgbm_shap_uploader",
)

if uploaded is None:
    st.info("Please upload a file to begin.")
    st.stop()

# --- Load Data ---
df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)

st.subheader("\ud83d\udc4a Data Preview")
st.dataframe(df.head())

# --- Target Column ---
target_col = st.selectbox("Select Target Column", df.columns)

# --- Feature Columns ---
X = df.drop(columns=[target_col])
y = df[target_col]

# Convert object columns for LightGBM
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].astype("category")

st.write("\ud83d\udccc X shape:", X.shape)
st.write("\ud83d\udccc y shape:", y.shape)

# --- Mode ---
mode = st.selectbox("Select Task Type", ["Regression", "Classification"])

if mode == "Regression":
    model = LGBMRegressor(n_estimators=300, random_state=42, boosting_type="gbdt")
else:
    model = LGBMClassifier(n_estimators=300, random_state=42, boosting_type="gbdt")

# --- Train Model ---
if st.button("Train Model"):
    model.fit(X, y)
    st.success("Model training completed!")
    st.session_state["lgbm_model"] = model
    st.session_state["X_lgbm"] = X
    st.session_state["mode_lgbm"] = mode

# --- SHAP Analysis ---
if "lgbm_model" in st.session_state and st.button("Run SHAP Analysis"):
    model = st.session_state["lgbm_model"]
    X = st.session_state["X_lgbm"]
    mode = st.session_state["mode_lgbm"]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # --- SHAP Summary Plot ---
    st.subheader("SHAP Summary Plot")

    plt.figure(figsize=(10, 5))
    if mode == "Classification" and isinstance(shap_values, list):
        shap.summary_plot(shap_values[0], X, show=False)
    else:
        shap.summary_plot(shap_values, X, show=False)
    st.pyplot(plt)

    # --- Feature Importance (LightGBM) ---
    st.subheader("\ud83d\udd25 LightGBM Feature Importance")

    imp_df = (
        pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
        .sort_values(by="Importance", ascending=False)
    )

    fig = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="LightGBM Feature Importance",
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- SHAP Bar Plot ---
    st.subheader("\ud83d\udcca SHAP Feature Importance (Mean |SHAP|)")

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

    # --- SHAP Dependence Plot ---
    st.subheader("\ud83d\udcc8 SHAP Dependence Plot")

    col1, col2 = st.columns(2)
    dep_feat = col1.selectbox("Feature for X-axis", X.columns)
    interaction_feat = col2.selectbox(
        "Feature for color / interaction (optional)",
        ["(auto)"] + list(X.columns),
    )

    if dep_feat:
        plt.figure(figsize=(7, 5))
        if interaction_feat == "(auto)":
            shap.dependence_plot(dep_feat, shap_values, X, show=False)
        else:
            shap.dependence_plot(
                dep_feat, shap_values, X, interaction_index=interaction_feat, show=False
            )
        st.pyplot(plt.gcf(), clear_figure=True)

    # --- SHAP Waterfall Plot ---
    st.subheader("\ud83d\udcdc SHAP Waterfall Plot (Individual Sample)")

    idx = st.number_input(
        "Row index", min_value=0, max_value=len(X) - 1, value=0
    )

    st.write("Selected row:")
    st.write(X.iloc[idx : idx + 1])

    shap_ex = shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X.iloc[idx],
        feature_names=X.columns,
    )

    shap.plots.waterfall(shap_ex, show=False)
    st.pyplot(bbox_inches="tight")
