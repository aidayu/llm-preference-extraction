"""評価結果のグラフ化スクリプト

評価結果CSVファイルからMicro/Macro/Weighted F1の棒グラフを生成する
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
import numpy as np
import pandas as pd

# 日本語フォント設定
try:
    import japanize_matplotlib  # pip install japanize-matplotlib
except ImportError:
    # japanize_matplotlibがない場合はシステムフォントを探す
    import matplotlib.font_manager as fm
    # 優先順位で日本語フォントを探す
    jp_fonts = ['IPAGothic', 'IPAPGothic', 'Noto Sans CJK JP', 'Hiragino Sans', 'Yu Gothic', 'MS Gothic']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    font_found = None
    for font in jp_fonts:
        if font in available_fonts:
            font_found = font
            break
    if font_found:
        plt.rcParams['font.family'] = font_found
    else:
        print("Warning: Japanese font not found. Install japanize-matplotlib or a Japanese font.")

# マイナス記号の文字化け対策
plt.rcParams['axes.unicode_minus'] = False

# =====================================================================
# === ユーザー設定 ===
# グラフ化したい評価結果CSVのパスをここに貼り付けてください
# =====================================================================
EVALUATION_CSV_PATH = "data/results/summary/comparison_results_macrof1.csv"
# =====================================================================


def parse_evaluation_csv(filepath: str) -> dict:
    """
    新形式の評価結果CSVファイルをパースする
    
    Returns:
        {
            "info": {"model": str, "timestamp": str, ...},
            "summary": DataFrame (Metric, Micro-F1, Macro-F1, Weighted-F1),
            "detailed": DataFrame (Metric, Type, Precision, Recall, F1)
        }
    """
    data = {"info": {}, "summary": None, "detailed": None}
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # 実験情報をパース
    for i, line in enumerate(lines):
        if line.startswith("Model,"):
            data["info"]["model"] = line.split(",")[1].strip()
        elif line.startswith("Timestamp,"):
            data["info"]["timestamp"] = line.split(",")[1].strip()
        elif line.startswith("Total Test Dialogues,"):
            data["info"]["n_dialogues"] = int(line.split(",")[1].strip())
    
    # サマリーテーブルを探す
    for i, line in enumerate(lines):
        if line.startswith("Metric,Micro-F1,"):
            # ヘッダー行から読み込む
            summary_lines = []
            for j in range(i, len(lines)):
                if lines[j].strip() == "" or lines[j].startswith(",,"):
                    break
                summary_lines.append(lines[j].strip())
            
            if len(summary_lines) > 1:
                header = summary_lines[0].split(",")
                rows = [line.split(",") for line in summary_lines[1:]]
                data["summary"] = pd.DataFrame(rows, columns=header)
            break
    
    # 詳細テーブルを探す
    for i, line in enumerate(lines):
        if line.startswith("Metric,Type,Precision"):
            detailed_lines = []
            for j in range(i, len(lines)):
                if lines[j].strip() == "" or lines[j].startswith(",,"):
                    break
                detailed_lines.append(lines[j].strip())
            
            if len(detailed_lines) > 1:
                header = detailed_lines[0].split(",")
                rows = [line.split(",") for line in detailed_lines[1:]]
                data["detailed"] = pd.DataFrame(rows, columns=header)
            break
    
    return data


def create_f1_comparison_chart(data: dict, output_path: str, title: str = None):
    """
    Micro/Macro/Weighted F1の比較棒グラフを作成
    """
    if data["summary"] is None:
        print("Error: Summary data not found in CSV")
        return
    
    df = data["summary"]
    
    # 数値に変換
    metrics = df["Metric"].tolist()
    micro_f1 = df["Micro-F1"].astype(float).tolist()
    macro_f1 = df["Macro-F1"].astype(float).tolist()
    weighted_f1 = df["Weighted-F1"].astype(float).tolist()
    
    # 表示用の短い名前
    display_names = {
        "Entity": "エンティティ",
        "Axis": "嗜好軸",
        "Sub-Axis": "サブ嗜好軸",
        "Hierarchical Axis": "階層嗜好軸",
        "Polarity": "嗜好極性",
        "Intensity": "嗜好強度",
        "Context": "文脈タグ集合",
        "Perfect Match": "完全一致",
    }
    labels = [display_names.get(m, m) for m in metrics]
    
    # グラフ作成
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 学術論文向けのグレースケール/ハイコントラスト設定
    colors = ['#e0e0e0', '#808080', '#202020']
    
    bars1 = ax.bar(x - width, [v * 100 for v in micro_f1], width, label='Micro-F1', color=colors[0], edgecolor='black')
    bars2 = ax.bar(x, [v * 100 for v in macro_f1], width, label='Macro-F1', color=colors[1], edgecolor='black')
    bars3 = ax.bar(x + width, [v * 100 for v in weighted_f1], width, label='Weighted-F1', color=colors[2], edgecolor='black')
    
    ax.set_xlabel('評価指標', fontsize=14)
    ax.set_ylabel('F1スコア (%)', fontsize=14)
    
    model = data["info"].get("model", "Unknown")
    if title is None:
        title = f'{model} 嗜好抽出評価'
    ax.set_title(title, fontsize=16)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, rotation=45, ha='right')
    ax.legend(loc='lower right', fontsize=12, frameon=True, edgecolor='black', fancybox=False)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
    ax.set_axisbelow(True)
    
    # 値を表示
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            ha='center', va='bottom', fontsize=8, fontfamily='serif')
    
    plt.tight_layout()
    
    # PDFとPNG両方出力
    base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
    plt.savefig(f"{base_path}.pdf", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Chart saved to: {base_path}.pdf, {base_path}.png")
    plt.close()


def create_precision_recall_chart(data: dict, output_path: str):
    """
    Precision/Recall/F1のグループ化棒グラフを作成（Micro平均のみ）
    """
    if data["detailed"] is None:
        print("Error: Detailed data not found in CSV")
        return
    
    df = data["detailed"]
    micro_df = df[df["Type"] == "Micro"].copy()
    
    metrics = micro_df["Metric"].tolist()
    precision = micro_df["Precision"].astype(float).tolist()
    recall = micro_df["Recall"].astype(float).tolist()
    f1 = micro_df["F1"].astype(float).tolist()
    
    display_names = {
        "Entity": "エンティティ", "Axis": "嗜好軸", "Sub-Axis": "サブ嗜好軸",
        "Hierarchical Axis": "階層的嗜好軸", "Polarity": "嗜好極性",
        "Intensity": "嗜好強度", "Context": "文脈タグ集合", "Perfect Match": "完全一致",
    }
    labels = [display_names.get(m, m) for m in metrics]
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 学術論文向けのグレースケール/ハイコントラスト設定
    colors = ['#ffffff', '#808080', '#000000']
    hatches = ['///', '', '']
    
    bars1 = ax.bar(x - width, [v * 100 for v in precision], width, label='適合率', color=colors[0], edgecolor='black', hatch=hatches[0])
    bars2 = ax.bar(x, [v * 100 for v in recall], width, label='再現率', color=colors[1], edgecolor='black', hatch=hatches[1])
    bars3 = ax.bar(x + width, [v * 100 for v in f1], width, label='F1', color=colors[2], edgecolor='black', hatch=hatches[2])
    
    ax.set_xlabel('評価指標', fontsize=14)
    ax.set_ylabel('スコア (%)', fontsize=14)
    ax.set_title('Micro平均: 適合率 / 再現率 / F1', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, rotation=45, ha='right')
    ax.legend(loc='lower right', fontsize=12, frameon=True, edgecolor='black', fancybox=False)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
    ax.set_axisbelow(True)
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            ha='center', va='bottom', fontsize=8, fontfamily='serif')
    
    plt.tight_layout()
    
    # PDFとPNG両方出力
    base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
    plt.savefig(f"{base_path}.pdf", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Chart saved to: {base_path}.pdf, {base_path}.png")
    plt.close()


def parse_comparison_csv(filepath: str) -> dict:
    """
    モデル比較CSVファイルをパースする
    
    Returns:
        {
            "title": str (比較のタイトル),
            "models": list[str] (モデル名のリスト),
            "metrics": list[str] (メトリクス名のリスト),
            "data": DataFrame (Model, Entity, Axis, ...)
        }
    """
    data = {"title": "", "models": [], "metrics": [], "data": None}
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # タイトル行を取得
    if lines:
        data["title"] = lines[0].strip().replace("\r", "")
    
    # ヘッダー行を探す
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Model,"):
            header_idx = i
            break
    
    if header_idx is None:
        print(f"Error: Could not find header row in {filepath}")
        return data
    
    # ヘッダーとデータを読み込む
    header = lines[header_idx].strip().replace("\r", "").split(",")
    data["metrics"] = header[1:]  # Model列を除く
    
    rows = []
    for line in lines[header_idx + 1:]:
        line = line.strip().replace("\r", "")
        if not line:
            continue
        parts = line.split(",")
        if len(parts) >= len(header):
            rows.append(parts[:len(header)])
    
    data["data"] = pd.DataFrame(rows, columns=header)
    data["models"] = data["data"]["Model"].tolist()
    
    return data


def create_model_comparison_chart(data: dict, output_path: str, title: str = None):
    """
    モデル間の性能比較棒グラフを作成（学術論文向けスタイル）
    各メトリクスについて、モデルごとの棒を並べて表示
    """
    if data["data"] is None or len(data["models"]) == 0:
        print("Error: No data found in comparison CSV")
        return
    
    df = data["data"]
    models = data["models"]
    metrics = data["metrics"]
    
    # 表示用の短い名前（日本語）
    display_names = {
        "Entity": "エンティティ",
        "Axis": "嗜好軸",
        "Sub-Axis": "サブ嗜好軸",
        "Hierarchical Axis": "階層的嗜好軸",
        "Polarity": "嗜好極性",
        "Intensity": "嗜好強度",
        "Context": "文脈タグ集合",
        "Perfect Match": "完全一致",
    }
    labels = [display_names.get(m, m) for m in metrics]
    
    # グラフ作成
    x = np.arange(len(labels))
    n_models = len(models)
    width = 0.18  # 固定幅でモデル間のスペースを確保
    
    # 学術論文向けスタイル設定（フォントは既にグローバル設定済み）
    plt.rcParams.update({
        'axes.linewidth': 1.0,
        'axes.edgecolor': 'black',
    })
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # 参照グラフに合わせたカラーパレット: グレートーン + ハッチパターン (5モデル対応)
    colors = ['#ffffff', '#c0c0c0', '#808080', '#404040', '#e0e0e0']
    hatches = ['///', '', '...', 'xxx', '\\\\\\']
    
    all_bars = []
    for i, model in enumerate(models):
        offset = (i - (n_models - 1) / 2) * width
        values = []
        for metric in metrics:
            try:
                val = float(df[df["Model"] == model][metric].iloc[0])
                values.append(val * 100)  # パーセントに変換
            except (ValueError, IndexError):
                values.append(0)
        
        color = colors[i % len(colors)]
        hatch = hatches[i % len(hatches)]
        bars = ax.bar(x + offset, values, width, label=model, 
                      color=color, edgecolor='black', hatch=hatch, linewidth=1.0)
        all_bars.append(bars)
    
    ax.set_xlabel('評価指標', fontsize=12, fontweight='normal')
    ax.set_ylabel('Macro F1スコア (%)', fontsize=12, fontweight='normal')
    
    if title is None:
        title = "嗜好属性抽出のモデル間比較"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=10)
    
    # 凡例を右上に配置
    ax.legend(loc='upper right', fontsize=9, frameon=True, 
              edgecolor='black', fancybox=False, framealpha=1.0)
    
    ax.set_ylim(0, 80)  # 実データ範囲に合わせて調整
    ax.set_yticks(np.arange(0, 81, 10))
    
    # 控えめなグリッド
    ax.grid(axis='y', linestyle='-', alpha=0.3, color='gray', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # スパイン（枠線）の設定
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    
    plt.tight_layout()
    
    # PDFとPNG両方出力
    base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
    plt.savefig(f"{base_path}.pdf", bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Chart saved to: {base_path}.pdf, {base_path}.png")
    plt.close()


def parse_hierarchical_summary_csv(filepath: str) -> dict:
    """
    Hierarchical Axis Macro サマリーCSVをパースする
    
    Returns:
        {
            "title": str,
            "models": list[str],
            "data": DataFrame (Model, Precision, Recall, F1)
        }
    """
    data = {"title": "", "models": [], "data": None}
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # タイトル行を取得
    if lines:
        data["title"] = lines[0].strip().replace("\r", "")
    
    # ヘッダー行を探す
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Model,"):
            header_idx = i
            break
    
    if header_idx is None:
        print(f"Error: Could not find header row in {filepath}")
        return data
    
    # ヘッダーとデータを読み込む
    header = lines[header_idx].strip().replace("\r", "").split(",")
    
    rows = []
    for line in lines[header_idx + 1:]:
        line = line.strip().replace("\r", "")
        if not line:
            continue
        parts = line.split(",")
        if len(parts) >= len(header):
            rows.append(parts[:len(header)])
    
    data["data"] = pd.DataFrame(rows, columns=header)
    data["models"] = data["data"]["Model"].tolist()
    
    return data


def create_hierarchical_comparison_chart(data: dict, output_path: str, title: str = None):
    """
    Hierarchical Axis Macro (hP, hR, hF1) のモデル間比較棒グラフを作成
    X軸: メトリクス (hP, hR, hF1)、各メトリクスにモデルごとのバー
    """
    if data["data"] is None or len(data["models"]) == 0:
        print("Error: No data found in hierarchical summary CSV")
        return
    
    df = data["data"]
    models = data["models"]
    metrics = ["hP", "hR", "hF1"]
    
    # 学術論文向けスタイル設定（フォントは既にグローバル設定済み）
    plt.rcParams.update({
        'axes.linewidth': 1.0,
        'axes.edgecolor': 'black',
    })
    
    # グラフ作成
    x = np.arange(len(metrics))
    n_models = len(models)
    width = 0.18
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 各モデルのメトリクス値を取得
    model_data = {}
    for model in models:
        row = df[df["Model"] == model].iloc[0]
        model_data[model] = [
            float(row["Precision"]) * 100,
            float(row["Recall"]) * 100,
            float(row["F1"]) * 100,
        ]
    
    # グレートーン + ハッチパターン（5モデル対応）
    colors = ['#ffffff', '#c0c0c0', '#808080', '#404040', '#e0e0e0']
    hatches = ['///', '', '...', 'xxx', '\\\\\\\\\\\\']
    
    all_bars = []
    for i, model in enumerate(models):
        offset = (i - (n_models - 1) / 2) * width
        values = model_data[model]
        color = colors[i % len(colors)]
        hatch = hatches[i % len(hatches)]
        bars = ax.bar(x + offset, values, width, label=model, 
                      color=color, edgecolor='black', hatch=hatch, linewidth=1.0)
        all_bars.append(bars)
    
    ax.set_xlabel('評価指標', fontsize=12, fontweight='normal')
    ax.set_ylabel('スコア (%)', fontsize=12, fontweight='normal')
    
    if title is None:
        title = "階層的嗜好軸評価 (Macro)"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11, rotation=0, ha='center')
    ax.tick_params(axis='y', labelsize=10)
    
    # 凡例
    ax.legend(loc='upper right', fontsize=9, frameon=True, 
              edgecolor='black', fancybox=False, framealpha=1.0)
    
    ax.set_ylim(0, 80)
    ax.set_yticks(np.arange(0, 81, 10))
    
    # グリッド
    ax.grid(axis='y', linestyle='-', alpha=0.3, color='gray', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # スパイン（枠線）の設定
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    
    plt.tight_layout()
    
    # PDFとPNG両方出力
    base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
    plt.savefig(f"{base_path}.pdf", bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Chart saved to: {base_path}.pdf, {base_path}.png")
    plt.close()

def main():    
    parser = argparse.ArgumentParser(description="評価結果CSVからグラフを生成")
    parser.add_argument("csv_path", nargs="?", default=EVALUATION_CSV_PATH,
                        help="評価結果CSVファイルのパス（省略時は上記設定を使用）")
    parser.add_argument("-o", "--output", help="出力画像パス（省略時はCSVと同じ場所）")
    parser.add_argument("-t", "--title", help="グラフタイトル")
    parser.add_argument("--type", choices=["f1", "prf", "both"], default="both",
                        help="グラフタイプ: f1=Micro/Macro/Weighted, prf=Precision/Recall/F1, both=両方")
    parser.add_argument("--compare", "-c", action="store_true",
                        help="モデル比較モード: comparison_results.csv形式のファイルから棒グラフを生成")
    parser.add_argument("--hierarchical", "-hi", action="store_true",
                        help="Hierarchical Axis比較モード: HP/HR/HF1をモデル間で比較")
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_path)
    print(f"評価結果ファイル: {csv_path}")
    
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        return
    
    # Hierarchical Axis比較モード
    if args.hierarchical:
        data = parse_hierarchical_summary_csv(str(csv_path))
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output = args.output or str(FIGURES_DIR / csv_path.stem) + "_hierarchical.pdf"
        create_hierarchical_comparison_chart(data, output, args.title)
        return
    
    # モデル比較モード
    if args.compare:
        data = parse_comparison_csv(str(csv_path))
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output = args.output or str(FIGURES_DIR / csv_path.stem) + "_comparison.png"
        create_model_comparison_chart(data, output, args.title)
        return
    
    # 通常モード（単一モデルの評価結果）
    data = parse_evaluation_csv(str(csv_path))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.type in ["f1", "both"]:
        output = args.output or str(FIGURES_DIR / csv_path.stem) + "_f1.png"
        create_f1_comparison_chart(data, output, args.title)
    
    if args.type in ["prf", "both"]:
        output = str(FIGURES_DIR / csv_path.stem) + "_prf.png"
        create_precision_recall_chart(data, output)


if __name__ == "__main__":
    main()

