import streamlit as st
import pandas as pd
import sweetviz as sv
import streamlit.components.v1 as components
from io import StringIO
from app import memo_sidebar

memo_sidebar()
st.title("📊 Sweetviz Profiling")

uploaded = st.file_uploader("CSV / Excel をアップロード", type=["csv", "xlsx"])

if not uploaded:
    st.info("ファイルをアップロードしてください。")
    st.stop()

# -------------------------------------------------------
# ① Excel の複数シートを選択できるようにする処理
# -------------------------------------------------------
def load_excel_with_sheet_selection(uploaded_file):
    xls = pd.ExcelFile(uploaded_file)
    sheet = st.selectbox(
        "読み込むシートを選択してください",
        xls.sheet_names,
        index=0
    )
    df = pd.read_excel(uploaded_file, sheet_name=sheet)
    return df


# -------------------------------------------------------
# ② 汎用データ読み込み（キャッシュ付き）
# -------------------------------------------------------
@st.cache_data
def load_data(uploaded_file, sheet_name=None):
    name = uploaded_file.name.lower()

    # CSV
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    # Excel（特定シート指定）
    if name.endswith(".xlsx") and sheet_name:
        return pd.read_excel(uploaded_file, sheet_name=sheet_name)

    # Excel（シート記載なし → 最初のシート）
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file, sheet_name=0)

    # その他（txt/tsv）
    text = uploaded_file.read().decode("utf-8", errors="ignore")
    return pd.read_csv(StringIO(text), sep="\t" if "\t" in text else ",")


# -------------------------------------------------------
# ③ 実際のシート選択 → データ読み込み
# -------------------------------------------------------
if uploaded.name.endswith(".xlsx"):
    # Excel の場合は一度シート名だけ抽出するために読み込み直す
    xls = pd.ExcelFile(uploaded)
    sheet_name = st.selectbox("読み込むシートを選択", xls.sheet_names)
    df = load_data(uploaded, sheet_name)
else:
    # CSV の場合はそのまま
    df = load_data(uploaded)


# -------------------------------------------------------
# ④ Sweetviz レポート生成（キャッシュ付き）
# -------------------------------------------------------
@st.cache_data
def generate_sweetviz_report(df):
    report = sv.analyze(df)
    report_path = "sweetviz_report.html"
    report.show_html(report_path, open_browser=False)

    with open(report_path, "r", encoding="utf-8") as f:
        html = f.read()

    return html

st.info("Sweetviz レポート生成中…")
html = generate_sweetviz_report(df)

components.html(html, height=900, scrolling=True)
