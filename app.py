import streamlit as st
import json
import os
import datetime

# -----------------------------
# Streamlit 基本設定
# -----------------------------
st.set_page_config(
    page_title="Data Science App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# タイトル
st.title("📊 Data Science App")

# -----------------------------
# 説明（必要に応じて編集）
# -----------------------------
st.markdown("""
### 👈 左のサイドバーからページを選択してください

このアプリは以下の分析ステップをサポートします：

- 📁 データアップロード  
- 🔍 探索的データ分析（EDA）  
- 📉 PCA / UMAP  
- 📊 相関可視化（Heatmap, Scatter）  
- 🧮 Metrics（MAPE, RMSE 等）  
- 🔬 Feature Importance  
- 📉 クラスタリング（KMeans / HDBSCAN）  
- 📝 ydata-profiling による自動レポート  
- 🧩 MitoSheet によるデータ加工 GUI  
""")



DB_DIR = "knowledge_db"
os.makedirs(DB_DIR, exist_ok=True)


# -----------------------------------------------------
# 🔧 シンプルメモ保存関数（メモ + コード + 画像）
# -----------------------------------------------------
def save_memo(memo_text, code_text, images):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    record = {
        "timestamp": timestamp,
        "memo": memo_text,
        "code": code_text,
        "image_files": []
    }

    base = f"{DB_DIR}/{timestamp}"

    # 画像保存
    for i, img in enumerate(images):
        img_path = f"{base}_{i+1}.png"
        with open(img_path, "wb") as f:
            f.write(img.getvalue())
        record["image_files"].append(img_path)

    # JSON 保存
    json_path = f"{base}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    return json_path


# -----------------------------------------------------
# 📝 サイドバー共通 UI（どのページでも使える）
# -----------------------------------------------------
def memo_sidebar():
    with st.sidebar:
        st.markdown("## 📝 メモ保存")

        memo_text = st.text_area("メモ内容（Markdown OK）", height=150)
        code_text = st.text_area("コード貼り付け", height=150)

        images = st.file_uploader(
            "画像（任意）",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="memo_images"
        )

        if st.button("💾 保存", key="save_memo"):
            path = save_memo(memo_text, code_text, images or [])
            st.success(f"保存しました: {path}")
