import streamlit as st
import pandas as pd
import sweetviz as sv
import streamlit.components.v1 as components

def run():
    st.title("📊 Sweetviz Profiling")

    uploaded = st.file_uploader("CSV/Excel をアップロード", type=["csv", "xlsx"])
    if not uploaded:
        st.info("ファイルをアップロードしてください")
        return

    df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)

    report = sv.analyze(df)
    report.show_html("sweetviz_report.html")

    # with open("sweetviz_report.html") as f:
    #     components.html(f.read(), height=900, scrolling=True)
