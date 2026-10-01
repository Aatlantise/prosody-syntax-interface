import os
import sys
import json
import re
import pandas as pd
import scipy.stats as stats


def detect_disfluencies_from_text(text):
    """Detects repetitions, false starts, and filled pauses directly on sentence text."""
    words = re.findall(r'\b[\w-]+\b', str(text).lower())

    repetitions = [words[i] for i in range(len(words) - 1)
                   if words[i] == words[i + 1] and words[i] not in ['um', 'uh', 'er', 'ah', 'hmm']]
    false_starts = re.findall(r'\b\w+-\b', str(text).lower())
    filled_pauses = re.findall(r'\b(um|uh|er|ah|hmm)\b', str(text).lower())

    total = len(repetitions) + len(false_starts) + len(filled_pauses)
    return total == 0, total


def analyze_sic_vs_features(jsonl_path, csv_baseline_path, csv_prosody_path):
    print("--- Step 1: Parsing Structural Features & Text from JSONL ---")
    features_list = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        idx = 0
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            # Replicate exact filter criteria
            if 'candor' in jsonl_path and data['text'][0].islower():
                continue

            text_val = data.get('text', '')
            is_fluent, n_disfluencies = detect_disfluencies_from_text(text_val)

            # Feature A: Word Count (Regex-cleaned)
            clean_text = re.sub(r'[",?.!:-;]', ' ', text_val)
            word_count = len(clean_text.split())

            # Feature B: Parse Tree Depth
            current_depth = 0
            parse_depth = 0
            for char in data.get('parse', ''):
                if char == '(':
                    current_depth += 1
                    if current_depth > parse_depth:
                        parse_depth = current_depth
                elif char == ')':
                    current_depth -= 1

            features_list.append({
                'original_index': idx,
                'turn_id': data.get('turn_id', idx),
                'text': text_val,
                'is_fluent': is_fluent,
                'total_disfluencies': n_disfluencies,
                'word_count': word_count,
                'parse_depth': parse_depth
            })
            idx += 1

    df_features = pd.DataFrame(features_list)
    print(f"Extracted features for {len(df_features):,} sentences.")

    print("\n--- Step 2: Aligning Baselines and Calculating SIC ---")
    df_base = pd.read_csv(csv_baseline_path)
    df_pros = pd.read_csv(csv_prosody_path)

    merged_models = pd.merge(
        df_base, df_pros,
        on="original_index",
        suffixes=('_baseline', '_prosody')
    )

    final_df = pd.merge(df_features, merged_models, on="original_index")
    print(f"Successfully aligned {len(final_df):,} instances for multi-variable SIC analysis.")

    # Calculate SIC
    final_df['sic'] = final_df['surprisal_baseline'] - final_df['surprisal_prosody']

    print("\n=======================================================")
    print(" 1. CORRELATION ANALYSIS (SIC vs. INDIVIDUAL FEATURES)")
    print("=======================================================")
    for feat in ['word_count', 'parse_depth']:
        # spearman_r, p_s = stats.spearmanr(final_df[feat], final_df['sic'])
        pearson_r, p_re = stats.pearsonr(final_df[feat], final_df['sic'])

        print(f"\nSIC vs. {feat.upper().replace('_', ' ')}:")
        # print(f"  Spearman r: {spearman_r:.4f} (p = {p_s:.4e})")
        print(f" Pearson r:    {pearson_r:.4f} (p = {p_re:.4f})")


    return final_df


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python analyze_sic_vs_features.py [baseline_dir] [prosody_dir]")
        sys.exit(1)

    arg1, arg2 = sys.argv[1], sys.argv[2]
    csv_1 = f"outputs/{arg1}/cross_validation_results.csv"
    csv_2 = f"outputs/{arg2}/cross_validation_results.csv"

    jsonl_data = "data/constituency_corpus.json" if 'libri' in arg1 else "data/candor_corpus.json"

    analysis_df = analyze_sic_vs_features(jsonl_data, csv_1, csv_2)
    analysis_df.to_csv(f"corr_{arg1}_{arg2}.csv", index=False)