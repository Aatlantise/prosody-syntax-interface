import pandas as pd
import numpy as np
from pathlib import Path
import sys
from scipy import stats


def load_and_align(path_dict):
    """
    Loads multiple CSVs and merges them into a single DataFrame aligned by 'original_index'.
    """
    merged_df = None
    dfs = []
    for name, path in path_dict.items():
        if not Path(path).exists():
            print(f"Warning: File not found: {path}")
            continue

        df = pd.read_csv(path)
        # We only need the index and the score
        df = df[['original_index', 'surprisal']].rename(columns={'surprisal': name})

        if len(dfs) == 0:
            merged_df = df
            dfs.append(df)
        else:
            # Inner join ensures we only compare sequences that exist in both models
            print(f"Loaded and merging {name}")
            merged_df = pd.merge(merged_df, df, on='original_index', how='inner')

    return merged_df


def paired_wilcoxon_test(data, col_a, col_b):
    """
    Runs a paired Wilcoxon signed-rank test between two aligned sequence columns.

    Returns:
        mean_diff: Mean difference (A - B) in nats. Positive means B achieves lower surprisal.
        p_value: Two-sided p-value testing distributional shift.
        r_rb: Rank-Biserial Correlation effect size [-1.0 to +1.0].
    """
    # Difference vector: (A - B)
    diffs = data[col_a] - data[col_b]
    mean_diff = np.mean(diffs)

    # Run two-sided paired Wilcoxon signed-rank test
    res = stats.wilcoxon(data[col_a], data[col_b], alternative='two-sided', method='auto')

    # Compute Rank-Biserial Correlation (r_rb) effect size
    # r_rb = (W_pos - W_neg) / (W_pos + W_neg)
    # Filter zero differences as scipy does by default
    non_zero_diffs = diffs[diffs != 0]
    ranks = stats.rankdata(np.abs(non_zero_diffs))

    w_pos = np.sum(ranks[non_zero_diffs > 0])
    w_neg = np.sum(ranks[non_zero_diffs < 0])
    w_total = w_pos + w_neg

    r_rb = (w_pos - w_neg) / w_total if w_total > 0 else 0.0

    return mean_diff, res.pvalue, r_rb


def main(dataset, cond):
    # 1. Define your file paths
    files = {
        "baseline": f"outputs/{dataset}_{cond}/cross_validation_results.csv",
        "text": f"outputs/{dataset}_text_{cond}/cross_validation_results.csv",
        "pause": f"outputs/{dataset}_pause_{cond}/cross_validation_results.csv",
        "duration": f"outputs/{dataset}_duration_{cond}/cross_validation_results.csv",
        "pause_text": f"outputs/{dataset}_text_pause_{cond}/cross_validation_results.csv",
        "duration_text": f"outputs/{dataset}_text_duration_{cond}/cross_validation_results.csv",
    }

    print("Loading and aligning data...")
    df = load_and_align(files)
    print(f"Aligned {len(df)} sequences across all conditions.\n")

    # 2. Define comparisons
    comparisons = [
        # (Model A, Model B) -> Is B significantly different from A?
        ("baseline", "text"),
        ("baseline", "pause"),
        ("baseline", "duration"),
        ("text", "pause_text"),
        ("text", "duration_text")
    ]

    print(f"{'Comparison':<30} | {'Mean Diff':<10} | {'r_rb':<8} | {'P-Value':<12} | {'Result'}")
    print("-" * 75)

    for model_a, model_b in comparisons:
        if model_a not in df.columns or model_b not in df.columns:
            print(f"Skipping {model_a} vs {model_b} (data missing)")
            continue

        # Run Wilcoxon Test
        diff, p, r_rb = paired_wilcoxon_test(df, model_a, model_b)

        # Significance tags
        if p < 0.001:
            significance = "***"
        elif p < 0.01:
            significance = "**"
        elif p < 0.05:
            significance = "*"
        else:
            significance = "n.s."

        comp_name = f"{model_a} vs {model_b}"
        print(f"{comp_name:<30} | {diff:>10.4f} | {r_rb:>8.4f} | {p:>12.4e} | {significance}")


if __name__ == '__main__':
    # Usage example: python -m significance libri nopunct
    assert len(sys.argv) == 3
    main(sys.argv[1], sys.argv[2])