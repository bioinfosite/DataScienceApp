import streamlit as st
from fpdf import FPDF
import io
import os

FONT_PATH = os.path.join(os.path.dirname(__file__), "../utils/fonts/ipaexg.ttf")


def run():

    st.title("EDAチェックリスト")

    # チェックリスト項目
    sections = [
        (
            "1. データ構造の理解",
            [
                "行数・列数を確認",
                "カラム名を確認（意味・命名規則・重複）",
                "データ型（int / float / object / datetime / category）",
                "主キー（ID）のユニーク性を確認",
                "ID列・不要列の洗い出し",
                "カテゴリ名の揺れ（Male/male など）",
            ],
        ),
        (
            "2. 欠損値（Missing Values）分析",
            [
                "欠損値の個数・割合を確認",
                "カラム単位の欠損パターンを確認",
                "行単位の欠損パターンを確認",
                "100%欠損の列がないか確認",
                "欠損が特定カテゴリに偏っていないか確認",
                "欠損補完方針を決定",
                "平均 / 中央値",
                "最頻値",
                "Zero / Unknown",
                "KNN / 回帰補完",
                "行/列削除",
            ],
        ),
        (
            "3. 外れ値（Outliers）分析",
            [
                "箱ひげ図で外れ値を確認",
                "標準偏差・IQR による外れ値検出",
                "分布の歪み（Skewness / Kurtosis）",
                "異常値の原因特定（誤入力 etc）",
            ],
        ),
        (
            "4. 基本統計量",
            [
                "平均・中央値",
                "分散・標準偏差",
                "最小値・最大値",
                "四分位数（Q1〜Q3）",
                "Zero-variance列の有無",
            ],
        ),
        (
            "5. カテゴリ分析",
            [
                "カテゴリ数の確認",
                "頻度の偏り（不均衡）",
                "Rare category の確認",
                "文字列の揺れ",
                "カテゴリ階層の確認",
            ],
        ),
        (
            "6. 相関分析",
            [
                "Pearson / Spearman / Kendall",
                "相関行列",
                "相関ヒートマップ",
                "高相関ペアの特定（多重共線性）",
                "特徴量変換（スケーリング / ログ）",
            ],
        ),
        (
            "7. ターゲット変数分析",
            [
                "y の分布",
                "数値×y の散布図",
                "カテゴリ×y の箱ひげ図",
                "y に強く影響する特徴量を把握",
                "リーク特徴がないか確認",
            ],
        ),
        (
            "8. 時系列データ（必要な場合）",
            [
                "datetime変換",
                "トレンド・季節性",
                "欠損日・異常spike",
                "ラグ特徴量の検討",
            ],
        ),
        (
            "9. 次元削減（PCA / UMAP / t-SNE）",
            [
                "PCA 2D/3D",
                "累積寄与率",
                "UMAP の局所構造確認",
                "t-SNE のクラスタ分離",
            ],
        ),
        (
            "10. クラスタリング",
            [
                "KMeans（適切な k）",
                "HDBSCAN（自然クラスタ）",
                "PCA/UMAP 上での可視化",
                "クラスタ特徴の解釈",
            ],
        ),
        (
            "11. 特徴量重要度",
            [
                "RandomForest重要度の確認",
                "Permutation Importance",
                "SHAP（必要なら）",
                "不要特徴量の除去検討",
            ],
        ),
        (
            "12. 回帰モデルの誤差指標",
            [
                "MAE",
                "MAPE",
                "MSE",
                "RMSE",
                "R²",
                "誤差の大きいサンプル確認",
            ],
        ),
        (
            "13. 分類モデルの指標",
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC/AUC",
                "Confusion Matrix",
            ],
        ),
        (
            "14. データ品質チェック",
            [
                "重複行の確認",
                "無効値（NaN, inf, 空文字）",
                "ラベルの一貫性",
                "数値の異常値（例：Age=300）",
                "カラム名の整形",
            ],
        ),
        (
            "15. 可視化チェック",
            [
                "ヒストグラム",
                "散布図",
                "boxplot / violinplot",
                "barplot / countplot",
                "相関ヒートマップ",
                "密度プロット（KDE）",
            ],
        ),
        (
            "16. EDAレポート（必要に応じて）",
            [
                "要約統計",
                "欠損・外れ値",
                "相関構造",
                "次元削減の知見",
                "クラスタリングの解釈",
                "特徴量重要度まとめ",
            ],
        ),
    ]

    # チェック状態を保存
    checked_dict = {}
    for section, items in sections:
        st.header(section)
        for item in items:
            checked = st.checkbox(item, key=f"{section}_{item}")
            checked_dict[item] = checked

    # PDF生成
    # if st.button("チェックリストをPDFで保存"):
    #     pdf = FPDF()
    #     pdf.add_page()
    #     pdf.add_font("IPAexGothic", "", FONT_PATH, uni=True)
    #     pdf.set_font("IPAexGothic", size=12)
    #     pdf.cell(0, 10, "EDAチェックリスト", ln=True, align="C")
    #     pdf.ln(5)
    #     for section, items in sections:
    #         pdf.set_font("IPAexGothic", style="B", size=12)
    #         pdf.cell(0, 10, section, ln=True)
    #         pdf.set_font("IPAexGothic", size=12)
    #         for item in items:
    #             mark = "[x]" if checked_dict[item] else "[ ]"
    #             pdf.cell(0, 8, f"{mark} {item}", ln=True)
    #         pdf.ln(2)
    #     pdf_output = bytes(pdf.output(dest="S"))
    #     st.download_button(
    #         label="PDFダウンロード",
    #         data=pdf_output,
    #         file_name="eda_checklist.pdf",
    #         mime="application/pdf",
    #     )

run()
