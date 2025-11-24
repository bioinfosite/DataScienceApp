import streamlit as st
import pandas as pd
import numpy as np
from app import memo_sidebar

# ============================================
# 📝 メモ Sidebar
# ============================================
memo_sidebar()

st.title("🧪 特徴量生成ツール（Undo / Redo / 時系列 / ビニング / GroupBy）")

# ------------------------------------------------
# 🔧 Session state 初期化
# ------------------------------------------------
if "feature_history" not in st.session_state:
    st.session_state.feature_history = []
if "feature_redo" not in st.session_state:
    st.session_state.feature_redo = []
if "original_df" not in st.session_state:
    st.session_state.original_df = None
if "new_columns" not in st.session_state:
    st.session_state.new_columns = []


# ------------------------------------------------
# 履歴管理
# ------------------------------------------------
def push_history(df_new: pd.DataFrame):
    history = st.session_state.feature_history

    if len(history) == 0:
        history.append(df_new.copy())
        st.session_state.feature_redo = []
        st.session_state.new_columns = []
        return

    old_df = history[-1]
    old_cols = set(old_df.columns)
    new_cols = set(df_new.columns) - old_cols

    history.append(df_new.copy())
    st.session_state.feature_redo = []
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

    def style(col):
        return (
            ["background-color: #fff3b0"] * len(col)
            if col.name in new_cols
            else [""] * len(col)
        )

    return df.style.apply(style)


# ------------------------------------------------
# 🔼 データアップロード
# ------------------------------------------------
uploaded = st.file_uploader(
    "特徴量生成のベースとなる CSV / Excel をアップロード",
    type=["csv", "xlsx"],
)

if uploaded is None and st.session_state.original_df is None:
    st.info("データをアップロードしてください。")
    st.stop()

if uploaded is not None and st.session_state.original_df is None:
    base_df = (
        pd.read_csv(uploaded)
        if uploaded.name.endswith(".csv")
        else pd.read_excel(uploaded)
    )
    st.session_state.original_df = base_df.copy()
    st.session_state.feature_history = [base_df.copy()]
    st.session_state.feature_redo = []

orig_df: pd.DataFrame = st.session_state.original_df
current_df: pd.DataFrame = st.session_state.feature_history[-1]


# ------------------------------------------------
# 📄 オリジナル表示
# ------------------------------------------------
st.subheader("📄 オリジナルデータ（head 2）")
st.dataframe(orig_df.head(2))


# ------------------------------------------------
# 🧩 特徴量生成メニュー（Expander方式）
# ------------------------------------------------
st.markdown("---")
st.header("📦 特徴量生成メニュー")


# ============================================================
# 1️⃣ 基本操作（算術 / 変換 / 差分）
# ============================================================
with st.expander("🧮 基本操作（算術 / 変換 / 差分）", expanded=False):

    sub_action = st.selectbox(
        "操作を選択",
        [
            "四則演算で新規特徴量",
            "既存列の変換（log / sqrt / z-score）",
            "組み合わせ特徴量（差分 / 比率）",
        ],
    )

    # ---- 四則演算 ----
    if sub_action == "四則演算で新規特徴量":
        colA = st.selectbox("特徴量 A", current_df.columns)
        colB = st.selectbox("特徴量 B", current_df.columns)
        op = st.selectbox("演算", ["A + B", "A - B", "A * B", "A / B"])
        new_name = st.text_input("新しい列名", f"{colA}_{op.replace(' ', '')}_{colB}")

        if st.button("▶ 生成", key="basic_add"):
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
            st.success(f"{new_name} を追加しました！")

    # ---- 変換 ----
    if sub_action == "既存列の変換（log / sqrt / z-score）":
        colX = st.selectbox("対象列", current_df.columns)
        trans = st.selectbox("変換方法", ["log", "sqrt", "square", "標準化(z-score)"])
        new_name = st.text_input("新列名", f"{colX}_{trans}")

        if st.button("▶ 変換実行", key="basic_transform"):
            df_new = current_df.copy()
            if trans == "log":
                df_new[new_name] = np.log1p(df_new[colX])
            elif trans == "sqrt":
                df_new[new_name] = np.sqrt(df_new[colX])
            elif trans == "square":
                df_new[new_name] = df_new[colX] ** 2
            else:
                df_new[new_name] = (df_new[colX] - df_new[colX].mean()) / df_new[
                    colX
                ].std()
            push_history(df_new)
            st.success(f"{new_name} を追加しました！")

    # ---- 差分 / 比率 ----
    if sub_action == "組み合わせ特徴量（差分 / 比率）":
        colA = st.selectbox("特徴量 A", current_df.columns)
        colB = st.selectbox("特徴量 B", current_df.columns)
        op = st.selectbox("生成タイプ", ["A - B", "A / B"])
        new_name = st.text_input("新列名", f"{colA}_{op.replace(' ', '')}_{colB}")

        if st.button("▶ 生成", key="basic_combine"):
            df_new = current_df.copy()
            df_new[new_name] = (
                df_new[colA] - df_new[colB]
                if op == "A - B"
                else df_new[colA] / df_new[colB]
            )
            push_history(df_new)
            st.success(f"{new_name} を生成しました！")


# ============================================================
# 2️⃣ 時系列特徴量（Lag / Rolling / Expanding / Datetime）
# ============================================================
with st.expander(
    "⏳ 時系列特徴量（Lag / Rolling / Expanding / Datetime）", expanded=False
):

    sub_action = st.selectbox(
        "時系列特徴量の種類",
        [
            "自動ラグ特徴量",
            "ローリング特徴量（Rolling）",
            "Expanding（累積）特徴量",
            "Datetime特徴量（year/month/day etc.）",
        ],
    )

    # ---- Lag ----
    if sub_action == "自動ラグ特徴量":
        time_col = st.selectbox("ソート用の列", current_df.columns)
        num_col = st.selectbox("ラグ対象列", current_df.columns)
        max_lag = st.number_input("最大ラグ", 1, 60, 3)

        if st.button("▶ ラグ生成", key="ts_lag"):
            df_new = current_df.copy().sort_values(time_col)
            for lag in range(1, max_lag + 1):
                df_new[f"{num_col}_lag{lag}"] = df_new[num_col].shift(lag)
            push_history(df_new)
            st.success("ラグ特徴量を追加しました！")

    # ---- Rolling ----
    if sub_action == "ローリング特徴量（Rolling）":
        colX = st.selectbox("対象列", current_df.columns)
        window = st.number_input("ウィンドウサイズ", 2, 200, 5)
        funcs = st.multiselect("統計量", ["mean", "std", "min", "max", "sum"])

        if st.button("▶ Rolling 生成", key="ts_roll"):
            df_new = current_df.copy()
            for f in funcs:
                df_new[f"{colX}_roll_{f}{window}"] = df_new[colX].rolling(window).agg(f)
            push_history(df_new)
            st.success("Rolling特徴量を生成しました！")

    # ---- Expanding ----
    if sub_action == "Expanding（累積）特徴量":
        colX = st.selectbox("対象列", current_df.columns)
        funcs = st.multiselect("統計量", ["mean", "std", "min", "max", "sum"])

        if st.button("▶ Expanding 生成", key="ts_exp"):
            df_new = current_df.copy()
            for f in funcs:
                df_new[f"{colX}_exp_{f}"] = df_new[colX].expanding().agg(f)
            push_history(df_new)
            st.success("Expanding特徴量を作成しました！")

    # ---- Datetime ----
    if sub_action == "Datetime特徴量（year/month/day etc.）":
        dt_col = st.selectbox("Datetime 列", current_df.columns)

        if st.button("▶ Datetime 展開", key="ts_dt"):
            df_new = current_df.copy()
            dt = pd.to_datetime(df_new[dt_col], errors="coerce")

            df_new[f"{dt_col}_year"] = dt.dt.year
            df_new[f"{dt_col}_month"] = dt.dt.month
            df_new[f"{dt_col}_day"] = dt.dt.day
            df_new[f"{dt_col}_weekday"] = dt.dt.weekday
            df_new[f"{dt_col}_hour"] = dt.dt.hour
            df_new[f"{dt_col}_quarter"] = dt.dt.quarter
            df_new[f"{dt_col}_dayofyear"] = dt.dt.dayofyear
            df_new[f"{dt_col}_is_weekend"] = dt.dt.weekday >= 5

            push_history(df_new)
            st.success("Datetime特徴量を追加しました！")


# ============================================================
# 3️⃣ Categorical Encoding
# ============================================================
with st.expander("🏷 カテゴリ encoding（one-hot / label / frequency）", expanded=False):

    colX = st.selectbox("カテゴリ列", current_df.columns, key="enc_col")
    enc_type = st.selectbox("方式", ["one-hot", "label", "frequency"], key="enc_type")

    if st.button("▶ エンコード", key="encoding"):
        df_new = current_df.copy()

        if enc_type == "one-hot":
            df_new = pd.get_dummies(df_new, columns=[colX], prefix=colX)
        elif enc_type == "label":
            df_new[f"{colX}_label"] = df_new[colX].astype("category").cat.codes
        elif enc_type == "frequency":
            freq = df_new[colX].value_counts(normalize=True)
            df_new[f"{colX}_freq"] = df_new[colX].map(freq)

        push_history(df_new)
        st.success("カテゴリ encoding を実行しました！")


# ============================================================
# 4️⃣ GroupBy 集約特徴量
# ============================================================
with st.expander("📊 GroupBy 集約特徴量", expanded=False):

    group_col = st.selectbox("GroupBy 対象列", current_df.columns, key="gb_group")
    target_col = st.selectbox("集約する数値列", current_df.columns, key="gb_target")
    funcs = st.multiselect(
        "関数", ["mean", "std", "min", "max", "sum", "count", "nunique"], key="gb_funcs"
    )

    if st.button("▶ 集約特徴量生成", key="gb_run"):
        df_new = current_df.copy()
        gb = df_new.groupby(group_col)[target_col].agg(funcs)

        gb.columns = [f"{target_col}_by_{group_col}_{f}" for f in funcs]
        df_new = df_new.merge(gb, left_on=group_col, right_index=True, how="left")

        push_history(df_new)
        st.success("GroupBy特徴量を生成しました！")


# ============================================================
# 5️⃣ ビニング（離散化）
# ============================================================
with st.expander("🎚 ビニング（等幅 / 等頻度 / KMeans）", expanded=False):

    colX = st.selectbox("対象列", current_df.columns, key="bin_col")
    method = st.selectbox(
        "方式", ["等幅ビニング", "等頻度ビニング", "KMeansビニング"], key="bin_method"
    )
    bins = st.number_input("ビン数（またはクラスタ数）", 2, 50, 5, key="bin_bins")

    if st.button("▶ ビニング実行", key="bin_run"):
        df_new = current_df.copy()

        if method == "等幅ビニング":
            df_new[f"{colX}_bin"] = pd.cut(df_new[colX], bins=bins, labels=False)
        elif method == "等頻度ビニング":
            df_new[f"{colX}_qbin"] = pd.qcut(
                df_new[colX], q=bins, labels=False, duplicates="drop"
            )
        else:
            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=bins, random_state=42)
            df_new[f"{colX}_kbin"] = km.fit_predict(df_new[[colX]])

        push_history(df_new)
        st.success("ビニングを適用しました！")


# ------------------------------------------------
# 🔧 Undo / Redo
# ------------------------------------------------
st.markdown("---")
st.subheader("⏪ Undo / Redo")

colU, colR = st.columns(2)

with colU:
    st.button(
        "⏪ Undo",
        on_click=do_undo,
        disabled=len(st.session_state.feature_history) <= 1,
        key="undo_btn",
    )

with colR:
    st.button(
        "⏩ Redo",
        on_click=do_redo,
        disabled=len(st.session_state.feature_redo) == 0,
        key="redo_btn",
    )

current_df = st.session_state.feature_history[-1]

# ------------------------------------------------
# 🧪 変換後データ表示
# ------------------------------------------------
st.subheader("🧪 変換後データ（最新）")
st.dataframe(highlight_new_columns(current_df), width="stretch")

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

# ------------------------------------------------
# 📘 特徴量生成一覧 & 説明
# ------------------------------------------------
st.markdown("---")
st.header("📘 このページで生成できる特徴量の一覧と説明")

feature_desc_md = """
## 🧮 基本操作（算術 / 変換 / 差分）

|カテゴリ|特徴量|説明|
|-------|------|----|
|四則演算|A + B / A - B / A * B / A / B|2つの数値列の算術演算。非線形モデルでも重要度が上がることが多い。|
|変換|log / sqrt / square / z-score|分布の歪み補正やスケーリング。外れ値に強いモデル化をサポート。|
|組み合わせ特徴量|差分（A-B）/ 比率（A/B）|変化や相対的な強さを表す。時系列・製造データで有効。|

---

## ⏳ 時系列特徴量（Lag / Rolling / Expanding / Datetime）

|カテゴリ|特徴量|説明|
|-------|------|----|
|Lag特徴量|lag1, lag2, ..., lagN|過去の値を特徴量化。時系列モデルで最重要。|
|Rolling特徴量|移動平均 / 移動分散 / 移動合計など|一定期間の傾向を捉える。ノイズ除去としても有効。|
|Expanding特徴量|累積平均 / 累積合計など|初期からの累積傾向を捉える。品質管理データで有効。|
|Datetime特徴量|year / month / day / weekday / quarter / 祝日など|カレンダー要因を持つ需要予測・売上予測で必須。|

---

## 🏷 カテゴリ Encoding

|カテゴリ|特徴量|説明|
|-------|------|----|
|One-hot encoding|col_A, col_B...|カテゴリをダミー変数に変換。木モデルと相性良い。|
|Label encoding|整数ラベル|カテゴリを番号化。順位を持たない点に注意。|
|Frequency encoding|カテゴリ頻度|カテゴリの出現率。One-hot より次元が小さく効果的。|

---

## 📊 GroupBy（集約）特徴量

|カテゴリ|特徴量|説明|
|-------|------|----|
|GroupBy Aggregation|mean / std / min / max / sum / count / nunique|ID や区分ごとの統計量。顧客分析・製造データで強力。|

---

## 🎚 ビニング（離散化）

|カテゴリ|特徴量|説明|
|-------|------|----|
|等幅ビニング|一定幅で区切ったカテゴリ|外れ値に敏感な連続値の単純化。|
|等頻度ビニング|データ数が均等になる区切り|分位点を利用するため分布に強い。|
|KMeansビニング|クラスタに基づくビニング|データの固まりを基準に自然な区切りを作る。|

---

### 🔍 特徴量生成の目的
- モデル性能の向上  
- データの構造をより忠実に表現  
- ノイズを減らしシグナルを強調  
- カテゴリデータや時系列データを ML が扱いやすい形に変換  

"""

st.markdown(feature_desc_md)
