import pandas as pd
import numpy as np
from pathlib import Path

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
            # (If your cross-validation was identical, this should preserve all rows)
            print(f"Loaded and merging {name}")
            merged_df = pd.merge(merged_df, df, on='original_index', how='inner')

    return merged_df

def paired_permutation_test(data, col_a, col_b, n_permutations=10000):
    """
    Runs a paired permutation test between two columns.
    Null Hypothesis: The mean difference between the two models is zero.
    """
    # 1. Calculate observed difference in means (A - B)
    # If A is baseline and B is better, this should be positive (Surprisal decreases)
    # If we want "Is B better than A?", we look for a significant reduction.
    diffs = data[col_a] - data[col_b]
    obs_diff = np.mean(diffs)
    
    # 2. Permutation Step
    # We randomly flip the sign of the differences.
    # This simulates the null hypothesis where Model A and Model B are exchangeable.
    
    # Create a matrix of random signs [-1, 1]
    # Shape: (n_permutations, n_samples)
    signs = np.random.choice([-1, 1], size=(n_permutations, len(diffs)))
    
    # Multiply differences by signs and compute mean across samples for each permutation
    # resulting shape: (n_permutations,)
    perm_means = np.mean(diffs.values * signs, axis=1)
    
    # 3. Calculate p-value
    # Two-sided test: proportion of permutations where abs(perm_mean) >= abs(obs_diff)
    p_value = np.mean(perm_means <= obs_diff)


    return obs_diff, p_value

def main():
    # 1. Define your file paths
    files = {
        "baseline": "/home/jm3743/prosody-syntax-interface/outputs/cross_validation_results.csv",
        "text": "/home/jm3743/prosody-syntax-interface/outputs/text/cross_validation_results.csv",
        "pause": "/home/jm3743/prosody-syntax-interface/outputs/pause/cross_validation_results.csv",
        "duration": "/home/jm3743/prosody-syntax-interface/outputs/duration/cross_validation_results.csv",
        # "pause_text":    "/home/jm3743/prosody-syntax-interface/outputs/pause_text/cross_validation_results.csv",
        # "duration_text": "/home/jm3743/prosody-syntax-interface/outputs/duration_text/cross_validation_results.csv",
    }

    print("Loading and aligning data...")
    df = load_and_align(files)
    print(f"Aligned {len(df)} sequences across all conditions.\n")

    # 2. Define the comparisons you asked for
    comparisons = [
        # (Model A, Model B) -> Is B significantly different from A?
        ("baseline", "text"),
        ("baseline", "pause"),
        ("baseline", "duration"),
        # ("text", "pause_text"),
        # ("text", "duration_text")
    ]

    print(f"{'Comparison':<30} | {'Mean Diff':<12} | {'P-Value':<10} | {'Result'}")
    print("-" * 70)

    for model_a, model_b in comparisons:
        if model_a not in df.columns or model_b not in df.columns:
            print(f"Skipping {model_a} vs {model_b} (data missing)")
            continue

        # Run Test
        diff, p = paired_permutation_test(df, model_a, model_b)

        # Interpretation
        # Diff is (Mean A - Mean B). Positive diff means A has HIGHER surprisal (B is better).
        significance = "*" if p < 0.05 else "n.s."
        if p < 0.001:
            significance = "***"
        elif p < 0.01:
            significance = "**"

        print(f"{model_a:<12} vs {model_b:<12} | {diff:>12.4f} | {p:>10.4f} | {significance}")


if __name__ == '__main__':
    main()