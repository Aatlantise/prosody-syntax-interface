import os
import sys
import json
import re
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm


def analyze_sic_vs_features(jsonl_path, csv_baseline_path, csv_prosody_path):
    print("--- Step 1: Parsing Structural Features from JSONL ---")
    features_list = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        idx = 0
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            # Replicate your exact cross-validation filter criteria
            if 'candor' in jsonl_path and data['text'][0].islower():
                continue

            # Feature A: Word Count (Regex-cleaned)
            clean_text = re.sub(r'[",?.!:-;]', ' ', data['text'])
            word_count = len(clean_text.split())

            # Feature B: Parse Tree Depth
            current_depth = 0
            parse_depth = 0
            for char in data['parse']:
                if char == '(':
                    current_depth += 1
                    if current_depth > parse_depth:
                        parse_depth = current_depth
                elif char == ')':
                    current_depth -= 1

            features_list.append({
                'original_index': idx,
                'word_count': word_count,
                'parse_depth': parse_depth
            })
            idx += 1

    df_features = pd.DataFrame(features_list)
    print(f"Extracted features for {len(df_features)} sentences.")

    print("\n--- Step 2: Aligning Baselines and Calculating SIC ---")
    df_base = pd.read_csv(csv_baseline_path)
    df_pros = pd.read_csv(csv_prosody_path)

    # Core merge ensuring perfect sequence alignment
    merged_models = pd.merge(
        df_base, df_pros,
        on="original_index",
        suffixes=('_baseline', '_prosody')
    )

    # Final merge attaching the parsed features
    final_df = pd.merge(df_features, merged_models, on="original_index")
    print(f"Successfully aligned {len(final_df)} instances for multi-variable SIC analysis.")

    # Calculate Syntactic Information Content (SIC): H(S) - H(S|P)
    # A positive delta indicates the information payload prosody provided to lower uncertainty.
    final_df['sic'] = final_df['surprisal_baseline'] - final_df['surprisal_prosody']

    print("\n=======================================================")
    print(" 1. CORRELATION ANALYSIS (SIC vs. INDIVIDUAL FEATURES)")
    print("=======================================================")
    for feat in ['word_count', 'parse_depth']:
        spearman_r, p_s = stats.spearmanr(final_df[feat], final_df['sic'])

        print(f"\nSIC vs. {feat.upper().replace('_', ' ')}:")
        print(f"  Spearman r: {spearman_r:.4f} (p = {p_s:.4e})")


    return final_df


if __name__ == "__main__":
    # Execution requires identifying the baseline and experimental systems via terminal
    if len(sys.argv) != 3:
        print("Usage: python analyze_sic_vs_features.py [baseline_dir] [prosody_dir]")
        sys.exit(1)

    arg1, arg2 = sys.argv[1], sys.argv[2]
    csv_1 = f"outputs/{arg1}/cross_validation_results.csv"
    csv_2 = f"outputs/{arg2}/cross_validation_results.csv"

    jsonl_data = "data/constituency_corpus.json" if 'libri' in arg1 else "data/candor_corpus.json"

    analysis_df = analyze_sic_vs_features(jsonl_data, csv_1, csv_2)

    analysis_df.to_csv(f"corr_{arg1}_{arg2}.csv")