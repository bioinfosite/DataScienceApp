import streamlit as st
import pandas as pd
from mitosheet.streamlit.v1 import spreadsheet

def run():
    st.title("🧪 Mito Data Editor")

    uploaded_files = st.file_uploader(
        "CSVまたはExcelファイルをアップロードしてください",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("まず CSV または Excel ファイルをアップロードしてください。")
        st.stop()

    dfs = {}
    df_names = []

    try:
        for uploaded in uploaded_files:
            fname = uploaded.name
            if fname.endswith(".csv"):
                dfs[fname] = pd.read_csv(uploaded)
                df_names.append(fname.replace(".csv", ""))
            else:
                xls = pd.ExcelFile(uploaded)
                sheets = st.multiselect(
                    f"Select sheets from {fname}",
                    xls.sheet_names,
                    default=xls.sheet_names,
                )
                for sheet in sheets:
                    key = f"{sheet}"
                    dfs[key] = pd.read_excel(uploaded, sheet_name=sheet)
                    df_names.append(key)
    except Exception as e:
        st.error(f"ファイルの読み込みに失敗しました: {e}")
        st.stop()

    new_dfs, code = spreadsheet(*dfs.values(), df_names=df_names)

    st.subheader("生成されたコード")
    st.code(code)
