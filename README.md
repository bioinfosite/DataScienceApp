# DataScienceApp

本リポジトリは、データサイエンスに必要な機能をまとめた  
**Streamlit ベースのインタラクティブアプリケーション**です。

特徴量生成、EDA、モデル構築、SHAP解析などをブラウザ上で簡単に実行できます。

メンテナンス： [bioinfosite](https://github.com/bioinfosite)

---

## 🚀 主な機能

- **特徴量エンジニアリング**
  - ラグ特徴量、自動時系列特徴量、カテゴリエンコーディング、数学変換など
  - Undo / Redo による履歴管理
  - 追加された列のハイライト表示

- **EDA**
  - 相関分析、PCA、UMAP、t-SNE
  - 自動レポート（profiling）
  - SHAP + LightGBM モデル学習

- **モデル構築**
  - 回帰 / 分類 LightGBM
  - 5-fold クロスバリデーション（性能一覧）

- **ベクトル指標**
  - 各種誤差指標、距離指標、相関指標の計算

- **メモ保存機能**
  - コード／文章／画像を保存し、ナレッジDBとして蓄積可能

---

## 🧰 必要環境

- **uv**  
- **Git**

---

## 📦 インストール手順

以下では、**uv の公式推奨インストール方法**でセットアップします。

---

## uv のインストール（公式）
### 🟦 Windows の場合
PowerShell を開き、以下を実行：

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 🟩 Linux / macOS の場合

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## リポジトリのクローン

```bash
git clone https://github.com/bioinfosite/DataScienceApp.git
cd DataScienceApp
```

## uvによる環境セットアップ

```bash
uv sync --frozen --native-tls
```

## アプリの起動

```bash
uv run streamlit run app.py
```

🌐 アクセス方法

起動後、自動的に URL が表示されます：

```bash
http://localhost:8501
```

ブラウザでアクセスしてください。

## その他

別ポートで起動したい

```bash
uv run streamlit run app.py --server.port 8510
```