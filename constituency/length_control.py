import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


def load_and_tag(file_path, dataset_label, feature_label, task_label):
    """Loads a condition CSV and tags its dataset, feature, and task metadata."""
    df = pd.read_csv(file_path)
    # Retain core fields and attach condition tags
    df = df[['original_index', 'word_count', 'parse_depth', 'sic']].copy()
    df['dataset'] = dataset_label  # 'CANDOR' or 'LibriTTS'
    df['feature'] = feature_label  # 'Pause' or 'Duration'
    df['task'] = task_label  # 'Full Parse' or 'Dyck Brackets'
    return df


# =====================================================================
# 1. LOAD YOUR CSV FILES (Update filenames to match your local paths)
# =====================================================================
# Spontaneous Speech (CANDOR)
c_pause_dk = load_and_tag('corr_candor_dyck_candor_pause_dyck.csv', 'CANDOR', 'Pause', 'Dyck Brackets')
c_dur_dk = load_and_tag('corr_candor_dyck_candor_duration_dyck.csv', 'CANDOR', 'Duration', 'Dyck Brackets')
c_pause_np = load_and_tag('corr_candor_nopunct_candor_pause_nopunct.csv', 'CANDOR', 'Pause', 'Full Parse')
c_dur_np = load_and_tag('corr_candor_nopunct_candor_duration_nopunct.csv', 'CANDOR', 'Duration', 'Full Parse')

# Read Speech (LibriTTS)
l_pause_dk = load_and_tag('corr_libri_dyck_libri_pause_dyck.csv', 'LibriTTS', 'Pause', 'Dyck Brackets')
l_dur_dk = load_and_tag('corr_libri_dyck_libri_duration_dyck.csv', 'LibriTTS', 'Duration', 'Dyck Brackets')
l_pause_np = load_and_tag('corr_libri_nopunct_libri_pause_nopunct.csv', 'LibriTTS', 'Pause', 'Full Parse')
l_dur_np = load_and_tag('corr_libri_nopunct_libri_duration_nopunct.csv', 'LibriTTS', 'Duration', 'Full Parse')

# Master Dataframe combining all conditions
master_df = pd.concat([
    c_pause_dk, c_dur_dk, c_pause_np, c_dur_np,
    l_pause_dk, l_dur_dk, l_pause_np, l_dur_np
], ignore_index=True)

# Calculate SIC Density (Information bits/nats per word)
master_df['sic_density'] = master_df['sic'] / master_df['word_count']


# =====================================================================
# 2. ANALYSIS 1: EXACT LENGTH-MATCHED SUBSAMPLING (CANDOR vs LibriTTS)
# =====================================================================
def run_exact_length_matching(df, task_filter='Full Parse', feature_filter='Pause', min_w=5, max_w=30, seed=42):
    """
    Subsamples exact 1-to-1 matching word counts between CANDOR and LibriTTS
    to evaluate if cross-dataset SIC differences survive length matching.
    """
    np.random.seed(seed)

    subset = df[(df['task'] == task_filter) & (df['feature'] == feature_filter)]
    candor = subset[subset['dataset'] == 'CANDOR']
    libri = subset[subset['dataset'] == 'LibriTTS']

    candor_matched = []
    libri_matched = []

    for w in range(min_w, max_w + 1):
        c_w = candor[candor['word_count'] == w]
        l_w = libri[libri['word_count'] == w]

        n_match = min(len(c_w), len(l_w))
        if n_match > 0:
            candor_matched.append(c_w.sample(n_match, random_state=seed))
            libri_matched.append(l_w.sample(n_match, random_state=seed))

    c_matched_df = pd.concat(candor_matched, ignore_index=True)
    l_matched_df = pd.concat(libri_matched, ignore_index=True)

    # Statistical Comparisons
    t_stat, p_val = stats.ttest_ind(c_matched_df['sic'], l_matched_df['sic'])

    print(f"=== EXACT LENGTH MATCHING ({feature_filter} | {task_filter}) ===")
    print(f"Total Matched Sentences per Dataset: {len(c_matched_df):,}")
    print(f"CANDOR   Mean SIC: {c_matched_df['sic'].mean():.4f} (Density: {c_matched_df['sic_density'].mean():.4f})")
    print(f"LibriTTS Mean SIC: {l_matched_df['sic'].mean():.4f} (Density: {l_matched_df['sic_density'].mean():.4f})")
    print(f"Independent t-test on Matched Data: t = {t_stat:.4f}, p = {p_val:.4e}\n")


# Run length matching on Pause & Duration conditions
run_exact_length_matching(master_df, task_filter='Full Parse', feature_filter='Pause')
run_exact_length_matching(master_df, task_filter='Full Parse', feature_filter='Duration')


# =====================================================================
# 3. ANALYSIS 2: OLS SIC RESIDUALIZATION (Holding Length Constant)
# =====================================================================
def compute_length_residuals(df):
    """
    Residualizes SIC against word_count across the full corpus to yield
    length-independent excess Syntactic Information Content.
    """
    X = sm.add_constant(df['word_count'])
    y = df['sic']
    model = sm.OLS(y, X).fit()
    df['sic_residual'] = model.resid
    return df


master_df = compute_length_residuals(master_df)

# Grouped Summary comparing Length-Controlled Residuals across Datasets
residual_summary = master_df.groupby(['dataset', 'feature', 'task'])[['sic', 'sic_density', 'sic_residual']].mean()
print("=== OVERALL METRICS CONTROLLED FOR LENGTH ===")
print(residual_summary)