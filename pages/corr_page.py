import streamlit as st
import pandas as pd
import plotly.express as px

st.title("\U0001F4CA Correlation Analysis")

# File uploader
uploaded = st.file_uploader(
    "Upload a CSV/Excel file for correlation analysis",
    type=["csv", "xlsx"],
    key="corr_uploader",
    accept_multiple_files=False,
)

if uploaded is None:
    st.info("Please upload a file to begin.")
    st.stop()

# Load data
df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)

# Data preview
st.subheader("\U0001F4C4 Data Preview")
st.dataframe(df.head())

# Select numeric columns
numeric_df = df.select_dtypes(include=["number"]).dropna()

if numeric_df.empty:
    st.error("The uploaded file has no numeric columns.")
else:
    # Select correlation method
    method = st.selectbox("Select Correlation Method", ["pearson", "spearman", "kendall"])
    
    # Compute correlation matrix
    corr = numeric_df.corr(method=method)
    
    # Display correlation matrix
    st.subheader(f"{method.upper()} Correlation Matrix")
    st.dataframe(corr)
    
    # Plot correlation heatmap
    st.subheader("Correlation Heatmap (Plotly)")
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)
