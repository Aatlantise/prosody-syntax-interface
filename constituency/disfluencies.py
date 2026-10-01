import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.mediation import Mediation


def load_candor_condition(file_path, feature, task):
    df = pd.read_csv(file_path)
    df['feature'] = feature
    df['task'] = task
    df['sic_density'] = df['sic'] / df['word_count']
    return df


def exact_length_matching(df):
    """Sub-samples fluent and non-fluent sets to have identical word_count distributions."""
    fluent_df = df[df['is_fluent'] == True]
    non_fluent_df = df[df['is_fluent'] == False]

    # Find common word counts present in both sets
    counts_fl = fluent_df['word_count'].value_counts()
    counts_nf = non_fluent_df['word_count'].value_counts()

    common_lengths = set(counts_fl.index).intersection(set(counts_nf.index))

    matched_fluent_list = []
    matched_nf_list = []

    np.random.seed(42)  # For reproducibility

    for length in common_lengths:
        n_match = min(counts_fl[length], counts_nf[length])

        fl_sub = fluent_df[fluent_df['word_count'] == length].sample(n=n_match, random_state=42)
        nf_sub = non_fluent_df[non_fluent_df['word_count'] == length].sample(n=n_match, random_state=42)

        matched_fluent_list.append(fl_sub)
        matched_nf_list.append(nf_sub)

    matched_fl = pd.concat(matched_fluent_list, ignore_index=True)
    matched_nf = pd.concat(matched_nf_list, ignore_index=True)

    return matched_fl, matched_nf


def compute_fluency_variance_explained_by_length(sub):
    sub = sub.copy()
    sub['is_non_fluent'] = (~sub['is_fluent']).astype(int)

    # 1. Variance Decomposition (R-squared Comparisons)
    m_fluency_only = smf.ols('sic ~ is_non_fluent', data=sub).fit()
    m_length_only = smf.ols('sic ~ word_count', data=sub).fit()
    m_full = smf.ols('sic ~ is_non_fluent + word_count', data=sub).fit()

    r2_fluency = m_fluency_only.rsquared
    r2_length = m_length_only.rsquared
    r2_full = m_full.rsquared

    # Delta R-squared: Explanatory gain of adding fluency to a length model
    delta_r2_fluency = r2_full - r2_length

    # 2. Linear Mediation Analysis (Baron & Kenny / Sobel Formulation)
    # Step a: Effect of Exposure (Disfluency) on Mediator (Word Count)
    m_a = smf.ols('word_count ~ is_non_fluent', data=sub).fit()
    a = m_a.params['is_non_fluent']

    # Step b & c': Direct & Mediator effects on Outcome (SIC)
    m_b = smf.ols('sic ~ is_non_fluent + word_count', data=sub).fit()
    c_prime = m_b.params['is_non_fluent']  # Direct effect (ADE)
    b = m_b.params['word_count']           # Mediator effect

    # Step c: Total Effect of Exposure on Outcome
    c = m_fluency_only.params['is_non_fluent']

    # Indirect Effect (ACME) = a * b
    indirect_effect = a * b
    total_effect = c

    # Proportion Mediated = Indirect / Total
    prop_mediated = (indirect_effect / total_effect) * 100 if total_effect != 0 else 0.0

    return {
        'R2 (Fluency Only)': f"{r2_fluency:.4f}",
        'R2 (Length Only)': f"{r2_length:.4f}",
        'R2 (Full Model)': f"{r2_full:.4f}",
        'Delta R2 (Adding Fluency)': f"{delta_r2_fluency:.6f}",
        'Total Effect (nats)': f"{total_effect:+.4f}",
        'Direct Effect ADE (nats)': f"{c_prime:+.4f}",
        'Indirect Effect ACME (nats)': f"{indirect_effect:+.4f}",
        'Proportion Mediated by Length': f"{prop_mediated:.2f}%"
    }



c_pause_dk = load_candor_condition('corr_candor_dyck_candor_pause_dyck.csv', 'Pause', 'Dyck Brackets')
c_dur_dk = load_candor_condition('corr_candor_dyck_candor_duration_dyck.csv', 'Duration', 'Dyck Brackets')
c_pause_np = load_candor_condition('corr_candor_nopunct_candor_pause_nopunct.csv', 'Pause', 'Full Parse')
c_dur_np = load_candor_condition('corr_candor_nopunct_candor_duration_nopunct.csv', 'Duration', 'Full Parse')

master_candor = pd.concat([c_pause_dk, c_dur_dk, c_pause_np, c_dur_np], ignore_index=True)

print("===========================================================")
print("ACCURATE UNCONSTRAINED FLUENCY ANALYSIS")
print("===========================================================")

for (feat, task), sub in master_candor.groupby(['feature', 'task']):
    fluent = sub[sub['is_fluent'] == True]
    non_fluent = sub[sub['is_fluent'] == False]

    n_sub = len(sub)
    n_fl = len(fluent)
    n_nf = len(non_fluent)

    len_fl = fluent['word_count'].mean()
    len_nf = non_fluent['word_count'].mean()

    sic_fl = fluent['sic'].mean()
    sic_nf = non_fluent['sic'].mean()
    t_sic, p_sic = stats.ttest_ind(fluent['sic'], non_fluent['sic'])

    den_fl = fluent['sic_density'].mean()
    den_nf = non_fluent['sic_density'].mean()
    t_den, p_den = stats.ttest_ind(fluent['sic_density'], non_fluent['sic_density'])

    print(f"\n--- Condition: {feat} | {task} (N = {n_sub:,}) ---")
    print(f"  • Fluent Ratio: {n_fl / n_sub * 100:.2f}% (N = {n_fl:,}) | Mean Words: {len_fl:.2f}")
    print(f"  • Non-Fluent:   {(n_nf / n_sub * 100):.2f}% (N = {n_nf:,}) | Mean Words: {len_nf:.2f}")
    print(f"  • Raw SIC:      Fluent = {sic_fl:.4f} nats | Non-Fluent = {sic_nf:.4f} nats (t = {t_sic:.2f}, p = {p_sic:.4e})")
    print(f"  • SIC Density:  Fluent = {den_fl:.4f} nats/w | Non-Fluent = {den_nf:.4f} nats/w (t = {t_den:.2f}, p = {p_den:.4e})")

print("===========================================================")
print("ACCURATE MATCHED-LENGTH FLUENCY ANALYSIS (1:1 Exact Matching)")
print("===========================================================")

for (feat, task), sub in master_candor.groupby(['feature', 'task']):
    # Perform exact length matching
    m_fl, m_nf = exact_length_matching(sub)

    n_matched = len(m_fl)
    mean_len = m_fl['word_count'].mean()

    sic_fl = m_fl['sic'].mean()
    sic_nf = m_nf['sic'].mean()
    t_sic, p_sic = stats.ttest_ind(m_fl['sic'], m_nf['sic'])

    h_s_fl = m_fl['surprisal_baseline'].mean()
    h_s_nf = m_nf['surprisal_baseline'].mean()

    u_fl = (sic_fl / h_s_fl) * 100
    u_nf = (sic_nf / h_s_nf) * 100

    print(f"\n--- Condition: {feat} | {task} (Matched Pair N = {n_matched:,} per group) ---")
    print(f"  • Matched Mean Words: {mean_len:.2f} (Identical across groups)")
    print(
        f"  • Matched Raw SIC:    Fluent = {sic_fl:.2f} nats | Non-Fluent = {sic_nf:.2f} nats (t = {t_sic:.2f}, p = {p_sic:.4e})")
    print(f"  • Matched Unc Coeff:  Fluent = {u_fl:.2f}% | Non-Fluent = {u_nf:.2f}%")

    print(compute_fluency_variance_explained_by_length(sub))