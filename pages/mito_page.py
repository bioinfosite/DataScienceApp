import streamlit as st
import pandas as pd
from mitosheet.streamlit.v1 import spreadsheet

def run():
    st.title("🧪 Mito Data Editor")

    uploaded_files = st.file_uploader(
        "CSVまたはExcelをアップロード",
        type=["csv", "xlsx"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("ファイルをアップロードしてください")
        return

    dfs = {}
    df_names = []

    for uploaded in uploaded_files:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        key = uploaded.name
        dfs[key] = df
        df_names.append(key)

    new_dfs, code = spreadsheet(*dfs.values(), df_names=df_names)

    st.subheader("生成されたコード")
    st.code(code)
