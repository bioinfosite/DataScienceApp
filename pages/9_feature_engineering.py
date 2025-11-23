import streamlit as st
import pandas as pd
import numpy as np
from app import memo_sidebar

# ============================================
# 📝 メモ Sidebar
# ============================================
memo_sidebar()
st.title("🧪 特徴量生成ツール（Undo / Redo / 時系列 / ラグ / エンコーディング）")

# ------------------------------------------------
# 🔧 Session state 初期化
# ------------------------------------------------
if "feature_history" not in st.session_state:
    st.session_state.feature_history = []  # 変換後の履歴（DataFrame）
if "feature_redo" not in st.session_state:
    st.session_state.feature_redo = []     # Undoした履歴
if "original_df" not in st.session_state:
    st.session_state.original_df = None    # アップロード直後のオリジナル
if "new_columns" not in st.session_state:
    st.session_state.new_columns = []  # 追加された新規列名リスト


# ------------------------------------------------
# 履歴管理
# ------------------------------------------------
def push_history(df_new: pd.DataFrame):
    history = st.session_state.feature_history

    # 履歴が空 → 初期登録（差分なし）
    if len(history) == 0:
        history.append(df_new.copy())
        st.session_state.feature_redo = []
        st.session_state.new_columns = []  # 初期は新規列なし
        return

    # 直前のデータフレームと比較して差分を取る
    old_df = history[-1]
    old_cols = set(old_df.columns)
    new_cols = set(df_new.columns) - old_cols

    # 履歴に追加
    history.append(df_new.copy())
    st.session_state.feature_redo = []
    
    # 追加された列を記録
    st.session_state.new_columns = list(new_cols)


def do_undo():
    if len(st.session_state.feature_history) > 1:
        last = st.session_state.feature_history.pop()
        st.session_state.feature_redo.append(last)


def do_redo():
    if len(st.session_state.feature_redo) > 0:
        restored = st.session_state.feature_redo.pop()
        st.session_state.feature_history.append(restored)

def highlight_new_columns(df: pd.DataFrame):
    new_cols = st.session_state.get("new_columns", [])

    def style(col_name):
        return "background-color: #fff3b0" if col_name in new_cols else ""

    return df.style.apply(lambda col: [style(col.name)] * len(col), axis=0)

# ------------------------------------------------
# 🔼 データアップロード
# ------------------------------------------------
uploaded = st.file_uploader(
    "特徴量生成のベースとなる CSV/Excel をアップロード",
    type=["csv", "xlsx"],
)

if uploaded is None and st.session_state.original_df is None:
    st.info("まずデータをアップロードしてください。")
    st.stop()

if uploaded is not None and st.session_state.original_df is None:
    base_df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
    st.session_state.original_df = base_df.copy()
    st.session_state.feature_history = [base_df.copy()]
    st.session_state.feature_redo = []

orig_df: pd.DataFrame = st.session_state.original_df
current_df: pd.DataFrame = st.session_state.feature_history[-1]

# ------------------------------------------------
# 📄 上側：オリジナル表示
# ------------------------------------------------
st.subheader("📄 オリジナルデータ（head 2）")
st.dataframe(orig_df.head(2))


# ------------------------------------------------
# 🧩 特徴量生成メニュー
# ------------------------------------------------
st.markdown("---")
st.header("📦 特徴量生成メニュー")

feature_action = st.selectbox(
    "生成する特徴量の種類を選択",
    [
        "四則演算で新規特徴量",
        "既存列の変換（log / sqrt / z-score）",
        "組み合わせ特徴量（差分 / 比率）",
        "自動ラグ特徴量（Lag features）",
        "時系列特徴量（Datetime expansion）",
        "カテゴリ encoding（One-hot / Label / Frequency）",
    ]
)

# ================================================================
# 1. 四則演算
# ================================================================
if feature_action == "四則演算で新規特徴量":
    st.subheader("➕ 四則演算による新しい特徴量")

    colA = st.selectbox("特徴量 A", current_df.columns)
    colB = st.selectbox("特徴量 B", current_df.columns)
    op = st.selectbox("演算", ["A + B", "A - B", "A * B", "A / B"])
    default_name = f"{colA}_{op.replace(' ', '')}_{colB}"
    new_name = st.text_input("新しい列名", default_name)

    if st.button("▶ 生成"):
        df_new = current_df.copy()
        if op == "A + B":
            df_new[new_name] = df_new[colA] + df_new[colB]
        elif op == "A - B":
            df_new[new_name] = df_new[colA] - df_new[colB]
        elif op == "A * B":
            df_new[new_name] = df_new[colA] * df_new[colB]
        elif op == "A / B":
            df_new[new_name] = df_new[colA] / df_new[colB]

        push_history(df_new)
        st.success(f"特徴量 {new_name} を追加しました！")
        current_df = df_new


# ================================================================
# 2. 既存列の変換
# ================================================================
elif feature_action == "既存列の変換（log / sqrt / z-score）":
    st.subheader("🔄 既存特徴量の変換")

    colX = st.selectbox("対象列", current_df.columns)
    trans = st.selectbox("変換方法", ["log", "sqrt", "square", "標準化(z-score)"])
    new_name = st.text_input("新しい列名", f"{colX}_{trans}")

    if st.button("▶ 変換実行"):
        df_new = current_df.copy()
        if trans == "log":
            df_new[new_name] = np.log1p(df_new[colX])
        elif trans == "sqrt":
            df_new[new_name] = np.sqrt(df_new[colX])
        elif trans == "square":
            df_new[new_name] = df_new[colX] ** 2
        elif trans == "標準化(z-score)":
            df_new[new_name] = (df_new[colX] - df_new[colX].mean()) / df_new[colX].std()

        push_history(df_new)
        st.success(f"特徴量 {new_name} を追加しました！")
        current_df = df_new


# ================================================================
# 3. 差分 / 比率
# ================================================================
elif feature_action == "組み合わせ特徴量（差分 / 比率）":
    st.subheader("📐 差分 / 比率特徴量")

    colA = st.selectbox("特徴量 A", current_df.columns)
    colB = st.selectbox("特徴量 B", current_df.columns)
    op = st.selectbox("生成タイプ", ["A - B", "A / B"])
    new_name = st.text_input("新しい列名", f"{colA}_{op.replace(' ', '')}_{colB}")

    if st.button("▶ 生成"):
        df_new = current_df.copy()
        if op == "A - B":
            df_new[new_name] = df_new[colA] - df_new[colB]
        else:
            df_new[new_name] = df_new[colA] / df_new[colB]

        push_history(df_new)
        st.success(f"特徴量 {new_name} を作成しました！")
        current_df = df_new


# ================================================================
# 4. 自動ラグ特徴量
# ================================================================
elif feature_action == "自動ラグ特徴量（Lag features）":
    st.subheader("⏳ 自動ラグ特徴量生成")

    time_col = st.selectbox("時系列順に並べる列（日時 or IDなど）", current_df.columns)
    num_col = st.selectbox("ラグを作成する数値列", current_df.columns)
    max_lag = st.number_input("最大ラグ", min_value=1, max_value=60, value=3)

    if st.button("▶ ラグ生成"):
        df_new = current_df.copy().sort_values(time_col)
        for lag in range(1, max_lag + 1):
            df_new[f"{num_col}_lag{lag}"] = df_new[num_col].shift(lag)

        push_history(df_new)
        st.success(f"{max_lag} 個のラグ特徴量を追加しました！")
        current_df = df_new


# ================================================================
# 5. 時系列特徴量
# ================================================================
elif feature_action == "時系列特徴量（Datetime expansion）":
    st.subheader("📅 時系列特徴量の展開")

    dt_col = st.selectbox("Datetime 列", current_df.columns)

    if st.button("▶ 時系列特徴量を生成"):
        df_new = current_df.copy()
        dt_series = pd.to_datetime(df_new[dt_col], errors="coerce")

        df_new[f"{dt_col}_year"] = dt_series.dt.year
        df_new[f"{dt_col}_month"] = dt_series.dt.month
        df_new[f"{dt_col}_day"] = dt_series.dt.day
        df_new[f"{dt_col}_weekday"] = dt_series.dt.weekday
        df_new[f"{dt_col}_hour"] = dt_series.dt.hour
        df_new[f"{dt_col}_quarter"] = dt_series.dt.quarter
        df_new[f"{dt_col}_dayofyear"] = dt_series.dt.dayofyear
        df_new[f"{dt_col}_is_weekend"] = dt_series.dt.weekday >= 5

        push_history(df_new)
        st.success("時系列特徴量を追加しました！")
        current_df = df_new


# ================================================================
# 6. カテゴリ encoding
# ================================================================
elif feature_action == "カテゴリ encoding（One-hot / Label / Frequency）":
    st.subheader("🏷 カテゴリ エンコーディング")

    colX = st.selectbox("カテゴリ列", current_df.columns)
    enc_type = st.selectbox("エンコーディング方式", ["one-hot", "label", "frequency"])

    if st.button("▶ エンコード"):
        df_new = current_df.copy()

        if enc_type == "one-hot":
            df_new = pd.get_dummies(df_new, columns=[colX], prefix=colX)
        elif enc_type == "label":
            df_new[f"{colX}_label"] = df_new[colX].astype("category").cat.codes
        elif enc_type == "frequency":
            freq = df_new[colX].value_counts(normalize=True)
            df_new[f"{colX}_freq"] = df_new[colX].map(freq)

        push_history(df_new)
        st.success("カテゴリ encoding を適用しました！")
        current_df = df_new


# ------------------------------------------------
# 🔧 Undo / Redo → ここに移動（中段から下段へ）
# ------------------------------------------------
st.markdown("---")
st.subheader("⏪ Undo / Redo")

colU, colR = st.columns(2)

with colU:
    st.button(
        "⏪ Undo",
        on_click=do_undo,
        disabled=len(st.session_state.feature_history) <= 1,
    )

with colR:
    st.button(
        "⏩ Redo",
        on_click=do_redo,
        disabled=len(st.session_state.feature_redo) == 0,
    )

current_df = st.session_state.feature_history[-1]


# ------------------------------------------------
# 🧪 下側：変換後のデータ表示
# ------------------------------------------------
st.subheader("🧪 変換後データ（最新）")
highlighted_df = highlight_new_columns(current_df)
st.dataframe(highlighted_df, width="stretch")

# ------------------------------------------------
# 📥 ダウンロード
# ------------------------------------------------
csv = current_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 現在の特徴量データをダウンロード",
    csv,
    file_name="feature_engineered.csv",
    mime="text/csv",
)
