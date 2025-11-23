import streamlit as st
import pandas as pd
from io import StringIO
from mitosheet.streamlit.v1 import spreadsheet
import re
from app import memo_sidebar

memo_sidebar()


st.title("🧪 Mito Data Editor")

uploaded_files = st.file_uploader(
    "CSV / Excel / TXT / TSV ファイルをアップロードしてください",
    type=["csv", "xlsx", "txt", "tsv"],
    accept_multiple_files=True,
)


# -----------------------------------------------------
# 🔧 変数名サニタイズ
# -----------------------------------------------------
def sanitize_df_name(name: str) -> str:
    name = name.replace(" ", "_")
    name = re.sub(r"\W", "_", name)  # 非英数字を "_"
    if re.match(r"^\d", name):  # 数字始まり対策
        name = "df_" + name
    return name


# -----------------------------------------------------
# 🔧 df_names の重複回避ロジック
# -----------------------------------------------------
def make_unique(name: str, existing_names: list) -> str:
    if name not in existing_names:
        return name
    i = 1
    new_name = f"{name}_{i}"
    while new_name in existing_names:
        i += 1
        new_name = f"{name}_{i}"
    return new_name


# -----------------------------------------------------
# 🔧 汎用ファイル読み込み関数（キャッシュ）
# -----------------------------------------------------
@st.cache_data
def load_table_file(uploaded_file, selected_columns=None):
    name = uploaded_file.name.lower()
    content = uploaded_file.read()

    if name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
        if selected_columns:
            df = df[selected_columns]
        return df

    text = content.decode("utf-8", errors="ignore")
    sio = StringIO(text)

    if name.endswith(".tsv"):
        df = pd.read_csv(sio, sep="\t")
    elif name.endswith(".txt"):
        if "\t" in text and text.count("\t") > text.count(","):
            df = pd.read_csv(sio, sep="\t")
        else:
            df = pd.read_csv(sio)
    else:
        df = pd.read_csv(sio)

    if selected_columns:
        df = df[selected_columns]

    return df


# -----------------------------------------------------
# 🔧 全ファイル処理（列100超 → 検索付き列選択 UI）
# -----------------------------------------------------
dfs = {}
df_names = []

if not uploaded_files:
    st.info("まずファイルをアップロードしてください。")
    st.stop()

for uploaded in uploaded_files:
    fname = uploaded.name

    uploaded.seek(0)
    tmp_df = load_table_file(uploaded)

    # 列数100超 → 検索付き列選択
    if tmp_df.shape[1] > 100:
        st.warning(
            f"{fname}: 列数が {tmp_df.shape[1]} 列あります。読み込む列を選択してください。"
        )

        search_text = st.text_input(f"{fname} の列名検索（部分一致）")

        if search_text:
            filtered_cols = [
                col for col in tmp_df.columns if search_text.lower() in col.lower()
            ]
        else:
            filtered_cols = tmp_df.columns.tolist()

        selected_cols = st.multiselect(
            f"{fname} の読み込む列",
            filtered_cols,
            default=filtered_cols[:50] if len(filtered_cols) > 50 else filtered_cols,
        )

        if not selected_cols:
            st.error("最低1つは列を選択してください。")
            st.stop()

        uploaded.seek(0)
        df = load_table_file(uploaded, selected_columns=selected_cols)

    else:
        df = tmp_df

    # ✨ サニタイズ + 重複回避
    safe_name = sanitize_df_name(fname)
    safe_name = make_unique(safe_name, df_names)

    dfs[safe_name] = df
    df_names.append(safe_name)


# -----------------------------------------------------
# 🔧 MitoSheet へ渡す
# -----------------------------------------------------
new_dfs, code = spreadsheet(*dfs.values(), df_names=df_names)


# -----------------------------------------------------
# 🔧 生成コード表示 + コピー
# -----------------------------------------------------
st.subheader("生成されたコード")
st.code(code)
