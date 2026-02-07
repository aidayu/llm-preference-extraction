"""Compare evaluation results across multiple models.

=== USER: Edit the EVALUATION_FILES list below ===

Usage:
    python compare_evaluations.py              # Uses files defined below
    python compare_evaluations.py --metric micro_f1   # Change metric type
    python compare_evaluations.py --hierarchical-summary   # Extract Hierarchical Axis Macro P/R/F1
"""

from pathlib import Path
import csv
import argparse

# =============================================================================
# ★★★ EDIT HERE: Add your evaluation CSV file paths ★★★
# =============================================================================

# Base directory (project root)
BASE_DIR = Path(__file__).resolve().parents[2]  # preference-kg/

# Evaluation files to compare (relative to project root)
EVALUATION_FILES = [
    "preference_kg/results/evaluations/gpt-5.2/20260127_153523/evaluation_20260127_154249_SemEMatch_3F1.csv",
    "preference_kg/results/evaluations/gpt-4o/20260127_130919/evaluation_20260127_145438_SemEMatch_3F1.csv",
    "preference_kg/results/evaluations/gpt-4o-mini/20260127_130447/evaluation_20260127_145623_SemEMatch_3F1.csv",
    "preference_kg/results/evaluations/gemma3:27b/20260127_133341/evaluation_20260127_150038_SemEMatch_3F1.csv",
    "preference_kg/results/evaluations/llama3.1:8b/20260127_132058/evaluation_20260127_150211_SemEMatch_3F1.csv",
]

# Default metric type: "micro_f1", "macro_f1", or "weighted_f1"
DEFAULT_METRIC = "macro_f1"


def get_timestamp() -> str:
    """現在時刻のタイムスタンプを取得"""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# Output file paths (with timestamp)
def get_output_file(name: str) -> Path:
    """タイムスタンプ付きの出力ファイルパスを生成"""
    ts = get_timestamp()
    return BASE_DIR / f"preference_kg/results/evaluations/{name}_{ts}.csv"


# Default output file paths (used if not specified via CLI)
OUTPUT_FILE = BASE_DIR / "preference_kg/results/evaluations/comparison_results.csv"
HIERARCHICAL_OUTPUT_FILE = BASE_DIR / "preference_kg/results/evaluations/hierarchical_axis_macro_summary.csv"
ACCURACY_OUTPUT_FILE = BASE_DIR / "preference_kg/results/evaluations/accuracy_macro_summary.csv"

# =============================================================================
# End of user configuration
# =============================================================================

# Attributes to extract
ATTRIBUTES = [
    "Entity",
    "Axis",
    "Sub-Axis",
    "Hierarchical Axis",
    "Polarity",
    "Intensity",
    "Context",
    "Perfect Match",
]

METRIC_COLUMNS = {
    "micro_f1": "Micro-F1",
    "macro_f1": "Macro-F1",
    "weighted_f1": "Weighted-F1",
}


def extract_model_name(filepath: Path) -> str:
    """Extract model name from file path."""
    parts = filepath.parts
    for i, part in enumerate(parts):
        if part == "evaluations" and i + 1 < len(parts):
            return parts[i + 1]
    return filepath.parent.name or filepath.stem


def parse_detailed_metrics(filepath: Path, metric_name: str, metric_type: str) -> dict:
    """Parse evaluation CSV and extract Precision, Recall, F1 from Detailed Metrics section.
    
    Args:
        filepath: Path to the evaluation CSV file.
        metric_name: The metric to extract (e.g., "Hierarchical Axis").
        metric_type: The type of metric (e.g., "Micro", "Macro", "Weighted").
    
    Returns:
        dict with keys "Precision", "Recall", "F1" or empty dict if not found.
    """
    result = {}
    
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    # Find the "Detailed Metrics" section header
    detailed_section_start = None
    header_row_idx = None
    
    for i, row in enumerate(rows):
        if len(row) > 0 and row[0].strip() == "Detailed Metrics":
            detailed_section_start = i
        if detailed_section_start is not None and len(row) >= 5:
            if "Metric" in row and "Type" in row and "Precision" in row and "Recall" in row and "F1" in row:
                header_row_idx = i
                break
    
    if header_row_idx is None:
        print(f"Warning: Could not find Detailed Metrics section in {filepath}")
        return result
    
    # Find column indices
    header = rows[header_row_idx]
    col_indices = {}
    for j, cell in enumerate(header):
        cell_stripped = cell.strip()
        if cell_stripped in ["Metric", "Type", "Precision", "Recall", "F1"]:
            col_indices[cell_stripped] = j
    
    # Search for the target row
    for row in rows[header_row_idx + 1:]:
        if len(row) <= max(col_indices.values()):
            continue
        row_metric = row[col_indices.get("Metric", 0)].strip()
        row_type = row[col_indices.get("Type", 1)].strip()
        
        if row_metric == metric_name and row_type == metric_type:
            try:
                result["Precision"] = float(row[col_indices["Precision"]])
            except (ValueError, KeyError):
                result["Precision"] = None
            try:
                result["Recall"] = float(row[col_indices["Recall"]])
            except (ValueError, KeyError):
                result["Recall"] = None
            try:
                result["F1"] = float(row[col_indices["F1"]])
            except (ValueError, KeyError):
                result["F1"] = None
            break
    
    return result


def extract_hierarchical_summary(filepaths: list[Path]) -> list[dict]:
    """Extract Hierarchical Axis Macro metrics from multiple evaluation files."""
    results = []
    for filepath in filepaths:
        model_name = extract_model_name(filepath)
        metrics = parse_detailed_metrics(filepath, "Hierarchical Axis", "Macro")
        result = {
            "Model": model_name,
            "Precision": metrics.get("Precision"),
            "Recall": metrics.get("Recall"),
            "F1": metrics.get("F1"),
        }
        results.append(result)
    return results


def save_hierarchical_summary_csv(results: list[dict], output_path: Path):
    """Save Hierarchical Axis Macro summary to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Hierarchical Axis - Macro"])
        writer.writerow([])
        writer.writerow(["Model", "Precision", "Recall", "F1"])
        
        for result in results:
            row = [
                result.get("Model", ""),
                f"{result['Precision']:.4f}" if result.get("Precision") is not None else "",
                f"{result['Recall']:.4f}" if result.get("Recall") is not None else "",
                f"{result['F1']:.4f}" if result.get("F1") is not None else "",
            ]
            writer.writerow(row)
    
    print(f"Saved: {output_path}")


# Accuracy attributes to extract
ACCURACY_ATTRIBUTES = [
    "Axis",
    "Sub-Axis",
    "H-Axis",
    "Polarity",
    "Intensity",
    "Context",
    "Perfect",
]


def parse_accuracy_metrics(filepath: Path, accuracy_type: str = "Macro-Accuracy") -> dict:
    """Parse evaluation CSV and extract Conditional Classification Accuracy.
    
    Args:
        filepath: Path to the evaluation CSV file.
        accuracy_type: "Micro-Accuracy" or "Macro-Accuracy"
    
    Returns:
        dict with attribute names as keys and accuracy values.
        Also includes "_matched_pairs" key for total matched pairs count.
    """
    result = {}
    
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    # Find Total Matched Pairs (usually near the top of the CSV)
    for row in rows:
        if len(row) >= 2:
            if "Total Matched Pairs" in row[0]:
                try:
                    result["_matched_pairs"] = int(row[1].strip())
                except ValueError:
                    pass
                break
    
    # Find the "Conditional Classification Accuracy" section
    accuracy_section_start = None
    header_row_idx = None
    
    for i, row in enumerate(rows):
        if len(row) > 0 and "Conditional Classification Accuracy" in row[0]:
            accuracy_section_start = i
        if accuracy_section_start is not None and len(row) >= 3:
            if "Attribute" in row and accuracy_type in row:
                header_row_idx = i
                break
    
    if header_row_idx is None:
        print(f"Warning: Could not find Accuracy section in {filepath}")
        return result
    
    # Find column index for the accuracy type
    header = rows[header_row_idx]
    acc_col_idx = None
    for j, cell in enumerate(header):
        if cell.strip() == accuracy_type:
            acc_col_idx = j
            break
    
    if acc_col_idx is None:
        print(f"Warning: Could not find {accuracy_type} column in {filepath}")
        return result
    
    # Extract accuracy values for each attribute (stop at empty row = end of section)
    for row in rows[header_row_idx + 1:]:
        # Stop at empty row (end of accuracy section)
        if len(row) == 0 or (len(row) > 0 and not row[0].strip()):
            break
        if len(row) <= acc_col_idx:
            continue
        attr = row[0].strip()
        if attr in ACCURACY_ATTRIBUTES:
            try:
                result[attr] = float(row[acc_col_idx])
            except (ValueError, IndexError):
                result[attr] = None
    
    return result


def extract_accuracy_summary(filepaths: list[Path]) -> list[dict]:
    """Extract Macro accuracy for all attributes from multiple evaluation files."""
    results = []
    for filepath in filepaths:
        model_name = extract_model_name(filepath)
        accuracy = parse_accuracy_metrics(filepath, "Macro-Accuracy")
        result = {"Model": model_name}
        result["Matched Pairs"] = accuracy.get("_matched_pairs")
        for attr in ACCURACY_ATTRIBUTES:
            result[attr] = accuracy.get(attr)
        results.append(result)
    return results


def save_accuracy_summary_csv(results: list[dict], output_path: Path):
    """Save Conditional Classification Accuracy (Macro) summary to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Conditional Classification Accuracy - Macro"])
        writer.writerow([])
        writer.writerow(["Model", "Matched Pairs"] + ACCURACY_ATTRIBUTES)
        
        for result in results:
            row = [result.get("Model", "")]
            matched = result.get("Matched Pairs")
            row.append(str(matched) if matched is not None else "")
            for attr in ACCURACY_ATTRIBUTES:
                val = result.get(attr)
                row.append(f"{val:.4f}" if val is not None else "")
            writer.writerow(row)
    
    print(f"Saved: {output_path}")

def parse_evaluation_csv(filepath: Path, metric_column: str) -> dict:
    """Parse evaluation CSV and extract F1 scores for all attributes."""
    scores = {}
    
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    # Find the header row
    header_row_idx = None
    metric_col_idx = None
    
    for i, row in enumerate(rows):
        if len(row) >= 4 and "Micro-F1" in row:
            header_row_idx = i
            for j, cell in enumerate(row):
                if cell == metric_column:
                    metric_col_idx = j
                    break
            break
    
    if header_row_idx is None or metric_col_idx is None:
        print(f"Warning: Could not find metric column '{metric_column}' in {filepath}")
        return scores
    
    # Extract scores
    for attr in ATTRIBUTES:
        for row in rows[header_row_idx + 1:]:
            if len(row) > metric_col_idx and row[0].strip() == attr:
                try:
                    scores[attr] = float(row[metric_col_idx])
                except (ValueError, IndexError):
                    scores[attr] = None
                break
    
    return scores


def compare_evaluations(filepaths: list[Path], metric_type: str) -> list[dict]:
    """Compare evaluation results from multiple files."""
    metric_column = METRIC_COLUMNS.get(metric_type)
    if not metric_column:
        raise ValueError(f"Unknown metric: {metric_type}")
    
    results = []
    for filepath in filepaths:
        model_name = extract_model_name(filepath)
        scores = parse_evaluation_csv(filepath, metric_column)
        result = {"Model": model_name}
        result.update(scores)
        results.append(result)
    
    return results


def save_comparison_csv(results: list[dict], output_path: Path, metric_type: str):
    """Save comparison results to CSV."""
    header = ["Model"] + ATTRIBUTES
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"Comparison: {metric_type.replace('_', ' ').title()}"])
        writer.writerow([])
        writer.writerow(header)
        
        for result in results:
            row = [result.get("Model", "")]
            for attr in ATTRIBUTES:
                value = result.get(attr)
                row.append(f"{value:.4f}" if value is not None else "")
            writer.writerow(row)
    
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results")
    parser.add_argument(
        "--metric", "-m",
        choices=["micro_f1", "macro_f1", "weighted_f1"],
        default=DEFAULT_METRIC,
        help=f"Metric type (default: {DEFAULT_METRIC})"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,  # None = auto-generate with timestamp
        help="Output CSV path (default: auto-generate with timestamp)"
    )
    parser.add_argument(
        "--hierarchical-summary", "-hs",
        action="store_true",
        help="Extract Hierarchical Axis Macro Precision/Recall/F1 for each model"
    )
    parser.add_argument(
        "--hierarchical-output", "-ho",
        type=Path,
        default=None,  # None = auto-generate with timestamp
        help="Output path for hierarchical summary CSV (default: auto-generate with timestamp)"
    )
    parser.add_argument(
        "--accuracy-summary", "-as",
        action="store_true",
        help="Extract Conditional Classification Accuracy (Macro) for each model"
    )
    parser.add_argument(
        "--accuracy-output", "-ao",
        type=Path,
        default=None,  # None = auto-generate with timestamp
        help="Output path for accuracy summary CSV (default: auto-generate with timestamp)"
    )
    args = parser.parse_args()
    
    # Generate timestamped output paths if not specified
    output_path = args.output or get_output_file("comparison_results")
    hierarchical_output_path = args.hierarchical_output or get_output_file("hierarchical_axis_macro_summary")
    accuracy_output_path = args.accuracy_output or get_output_file("accuracy_macro_summary")
    
    # Resolve paths
    filepaths = [BASE_DIR / f for f in EVALUATION_FILES]
    
    # Validate files
    missing = [f for f in filepaths if not f.exists()]
    if missing:
        print("Error: Files not found:")
        for f in missing:
            print(f"  - {f}")
        return
    
    # Handle accuracy summary mode
    if args.accuracy_summary:
        print(f"=== Extracting Conditional Classification Accuracy (Macro) ({len(filepaths)} models) ===")
        for f in filepaths:
            print(f"  - {extract_model_name(f)}")
        print()
        
        results = extract_accuracy_summary(filepaths)
        save_accuracy_summary_csv(results, accuracy_output_path)
        
        # Print summary
        print("\n=== Conditional Classification Accuracy - Macro ===")
        print(f"{'Model':<15}{'Matched':<10}", end="")
        for attr in ACCURACY_ATTRIBUTES:
            print(f"{attr[:8]:<10}", end="")
        print()
        print("-" * (25 + 10 * len(ACCURACY_ATTRIBUTES)))
        for result in results:
            matched = result.get("Matched Pairs")
            print(f"{result['Model']:<15}{matched if matched else 'N/A':<10}", end="")
            for attr in ACCURACY_ATTRIBUTES:
                v = result.get(attr)
                print(f"{v:.4f}    " if v is not None else "N/A       ", end="")
            print()
        return
    
    # Handle hierarchical summary mode
    if args.hierarchical_summary:
        print(f"=== Extracting Hierarchical Axis Macro metrics ({len(filepaths)} models) ===")
        for f in filepaths:
            print(f"  - {extract_model_name(f)}")
        print()
        
        results = extract_hierarchical_summary(filepaths)
        save_hierarchical_summary_csv(results, hierarchical_output_path)
        
        # Print summary
        print("\n=== Hierarchical Axis - Macro ===")
        print(f"{'Model':<15} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 51)
        for result in results:
            p = result.get("Precision")
            r = result.get("Recall")
            f = result.get("F1")
            print(
                f"{result['Model']:<15} "
                f"{f'{p:.4f}' if p is not None else 'N/A':<12} "
                f"{f'{r:.4f}' if r is not None else 'N/A':<12} "
                f"{f'{f:.4f}' if f is not None else 'N/A':<12}"
            )
        return
    
    # Standard comparison mode
    print(f"=== Comparing {len(filepaths)} evaluations ({args.metric}) ===")
    for f in filepaths:
        print(f"  - {extract_model_name(f)}")
    print()
    
    results = compare_evaluations(filepaths, args.metric)
    save_comparison_csv(results, output_path, args.metric)
    
    # Print summary
    print("\n=== Results ===")
    print(f"{'Model':<15}", end="")
    for attr in ATTRIBUTES:
        print(f"{attr[:8]:<10}", end="")
    print()
    
    for result in results:
        print(f"{result['Model']:<15}", end="")
        for attr in ATTRIBUTES:
            v = result.get(attr)
            print(f"{v:.4f}    " if v else "N/A       ", end="")
        print()


if __name__ == "__main__":
    main()
