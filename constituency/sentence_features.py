import scipy.stats as stats
import statsmodels.api as sm
import os

import json
import pandas as pd
import re
from constituency.util import remove_punctuation, load_jsonl_data
import argparse


def analyze_surprisal_vs_features(jsonl_path, csv_path):
    # 1. Parse sentence features from the single JSONL file line-by-line
    print("Parsing features from JSONL file...")
    features_list = []

    # args = argparse.Namespace()
    # args.data = jsonl_path
    # args.dyck = 'dyck' in csv_path
    # args.nopunct = 'nopunct' in csv_path
    # args.debug = False
    #
    # data = load_jsonl_data(args)
    # print(f"Read data with {len(data)} rows...")

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        idx = 0
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            if 'candor'in jsonl_path and (
                    # data['text'][-1] not in ['.', '?', '!'] or
                    # len(data['text'].split()) < 5 or
                    data['text'][0].islower()
            ):
                continue

            # Feature A: Sentence length in words (split clean text)
            # Stripping punctuation/quotes to get true word tokens
            clean_text = remove_punctuation(data['text'])
            word_count = len(clean_text.split())

            # Feature B: Length in syntactic parse tokens
            # We split the 'parse' string by spaces to count every bracket and POS tag
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
                'text': clean_text,
                'word_count': word_count,
                'parse_depth': parse_depth
            })
            idx += 1

    print(len(features_list))

    df_features = pd.DataFrame(features_list)
    print(f"Extracted features for {len(df_features)} sentences.")

    # 2. Load and merge your surprisal CSV data
    print(f"Loading surprisal data from {csv_path}...")
    df_surprisal = pd.read_csv(csv_path)

    # Merge using 'original_index' to guarantee exact sentence alignment
    merged_df = pd.merge(df_features, df_surprisal, on='original_index')
    print(f"Successfully aligned {len(merged_df)} items for analysis.\n")

    # 3. Calculate Correlation Coefficients
    features_to_test = ['word_count', 'parse_depth']

    print("=== Correlation Analysis ===")
    for feat in features_to_test:
        pearson_r, p_p = stats.pearsonr(merged_df[feat], merged_df['surprisal'])
        spearman_r, p_s = stats.spearmanr(merged_df[feat], merged_df['surprisal'])

        print(f"\nSurprisal vs. {feat.upper().replace('_', ' ')}:")
        print(f"  Pearson  r: {pearson_r:.4f} (p = {p_p})")
        print(f"  Spearman r: {spearman_r:.4f} (p = {p_s})")
        print(f"  *Interpretation: Explains roughly {pearson_r ** 2 * 100:.1f}% of individual surprisal variance.*")

    # 4. Run a Multiple Linear Regression Analysis
    print("\n=== Multiple Linear Regression ===")
    print("Determining unique contribution of words vs. parse tokens to total surprisal:")

    X = merged_df[['word_count', 'parse_depth']]
    X = sm.add_constant(X)  # Adds an intercept to the linear model
    y = merged_df['surprisal']

    model = sm.OLS(y, X).fit()
    print(model.summary().tables[1])  # Print the coefficient table
    print(f"\nOverall Model R-squared: {model.rsquared:.4f}")

    return merged_df


# --- How to execute ---
if __name__ == "__main__":
    # Point these to your local file paths
    models = os.listdir("outputs/")

    for model in models:
        if '.txt' in model:
            continue
        print(f"Analyzing {model}...")
        csv_data = f"outputs/{model}/cross_validation_results.csv"

        if 'libri' in model:
            jsonl_data = "data/constituency_corpus_reldur.json"
        else:
            jsonl_data = "data/candor_corpus.json"
            continue

        analysis_df = analyze_surprisal_vs_features(jsonl_data, csv_data)
        # print(analysis_df[['text', 'word_count', 'surprisal']].head(10))