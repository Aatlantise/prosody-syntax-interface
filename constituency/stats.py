libri = {
    "p": [58.1922858116721, 57.179307779157945, 63.19266932522599, 58.6766439725217, 57.091733962972135, 57.3192792484124, 58.59848131905588, 58.18474101721763, 55.68267343180855, 64.84323365999637],
    "null": [59.02903351507332, 57.79125732250244, 57.923196211522, 61.14501842378671, 64.65802358035336, 58.600799644208735, 58.68399031882479, 61.84898447163067, 58.62005393510004, 58.92948400698945],
    "d": [48.896244950549885, 49.08022327461568, 47.980937657639004, 50.624046914088424, 51.17397725123625, 49.83302959300332, 51.22300919985924, 48.110421665777, 50.32697641647561, 55.67185707668681],
    "w": [15.539112633118469, 15.576098457895425, 15.477759756868556, 15.838496414278783, 15.799978020621634, 15.45552759452472, 14.824296511926068, 14.923157575721058, 15.856881477148288, 15.663684770659128],
    "wd": [15.178778196829331, 16.165050371881073, 14.72947495086835, 17.280038001336457, 15.027902464167429, 14.88771725050941, 15.276162951516037, 15.374194131225973, 16.406787824574792, 16.16053740557747],
    "wp": [14.165557611572522, 14.154570144537004, 14.57316129678526, 14.746505408906021, 14.947082131814497, 15.32884250698618, 15.455977677152708, 14.470964778068, 14.450793612823437, 14.291718571805037],
}

candor = {
    "null": [41.65105414019848, 43.260830887446005, 43.144829116094726, 40.79891245939042, 41.15904525548067, 41.82155974560168, 44.601000901479814, 40.642021814169844, 43.2946946575687, 41.70371954016058],
    "p": [39.95251705456615, 39.08281176487085, 42.76269297084351, 40.883911652291, 44.03629241255208, 39.157586853873816, 38.54042364327972, 40.77726078524401, 42.08693646460954, 38.38985396793745],
    "d": [44.171140657610245, 40.25962461777989, 37.964247619694014, 39.183047845845124, 44.120012933592605, 41.25399442216491, 39.413042085126655, 40.55519746828987, 38.32605699307008, 38.67704684619788],
    "w": [4.475936032930449, 4.41629189672072, 4.373160111884963, 4.309881421991609, 4.450303030937127],
    "wd": [5.984729849132943, 5.437787234324029, 5.136712109357758, 5.2048087364332565, 4.92834643032445],
    "wp": [5.013073527471451, 4.5367043146032255, 4.542161204447006, 4.493724982414664, 4.725127554509398],
}

import pandas as pd
import numpy as np
from scipy import stats


def fast_paired_permutation(diffs, n_resamples=10000, batch_size=100):
    obs_mean = np.mean(diffs)
    n = len(diffs)
    count_extreme = 0

    # Process in batches to keep memory usage tiny
    for i in range(0, n_resamples, batch_size):
        current_batch = min(batch_size, n_resamples - i)

        # Generate random signs (1 or -1) of shape (batch_size, n)
        signs = np.random.choice([-1, 1], size=(current_batch, n))

        # Multiply diffs by signs and take the mean across the instances
        perm_means = np.mean(signs * diffs, axis=1)

        # Count how many permuted means are as extreme as the observed mean
        count_extreme += np.sum(np.abs(perm_means) >= np.abs(obs_mean))

    return count_extreme / n_resamples

def analyze_conditions(condition_a, condition_b):
    condition_A_path = f"/home/jm3743/prosody-syntax-interface/outputs/{condition_a}/cross_validation_results.csv"
    condition_B_path = f"/home/jm3743/prosody-syntax-interface/outputs/{condition_b}/cross_validation_results.csv"

    # 1. Load the data
    df_A = pd.read_csv(condition_A_path)
    df_B = pd.read_csv(condition_B_path)

    # 2. Align the data by original_index (this solves the inconsistent fold issue)
    # Using an inner merge ensures we only compare sequences present in both files
    merged = pd.merge(df_A, df_B, on='original_index', suffixes=('_A', '_B'))

    # Extract the paired surprisal lists
    surprisal_A = merged['surprisal_A'].values
    surprisal_B = merged['surprisal_B'].values

    # 3. Calculate instance-level Mutual Information (MI)
    # Assuming MI(A, B) = H(A) - H(B)
    mi_instances = surprisal_A - surprisal_B
    a_mean = np.mean(surprisal_A)
    b_mean = np.mean(surprisal_B)
    mi_mean = np.mean(mi_instances)

    # 4. Calculate the Standard Error (SE) of the MI
    # Analytically: Standard deviation of the differences divided by sqrt(N)
    n = len(mi_instances)
    mi_se = np.std(mi_instances, ddof=1) / np.sqrt(n)

    # 5. Calculate Statistical Significance

    # Option 1: Wilcoxon Signed-Rank Test (Fast, standard non-parametric paired test)
    stat_wilcoxon, p_val_wilcoxon = stats.wilcoxon(surprisal_A, surprisal_B)

    # Option 2: Paired Permutation Test (Gold standard for NLP surprisal)
    # We test if the mean difference (MI) is significantly different from 0

    p_val_permutation = fast_paired_permutation(mi_instances)

    # 6. Print Results
    print(f"Number of aligned sequences: {n}")
    print(f"Estimated {condition_a} Entropy: {a_mean:.4f} nats")
    print(f"Estimated {condition_b} Entropy: {b_mean:.4f} nats")
    print(f"Estimated MI (Mean Difference): {mi_mean:.4f} nats")
    print(f"Standard Error of MI: ±{mi_se:.4f}")
    print("-" * 30)
    print(f"Wilcoxon p-value: {p_val_wilcoxon:.4e}")
    print(f"Permutation test p-value: {p_val_permutation:.4e}")

    return mi_mean, mi_se, p_val_permutation


def calculate_aggregate_mi(cond_a, cond_b):
    """
    folds_A: list of 10 entropy estimates (e.g., text only)
    folds_B: list of 5 entropy estimates (e.g., text + pause)
    """

    def get_fold_from_cond(cond):
        if 'autoreg' in cond:
            _key = 'null'
        elif 'text_pause' in cond or 'pause_text' in cond:
            _key = 'wp'
        elif 'text_dur' in cond or 'duration_text' in cond:
            _key = 'wd'
        elif 'text' in cond:
            _key = 'w'
        elif 'pause' in cond:
            _key = 'p'
        else:
            _key = 'd'

        if 'candor' in cond:
            return candor[_key]
        else:
            return libri[_key]

    folds_A = get_fold_from_cond(cond_a)
    folds_B = get_fold_from_cond(cond_b)

    n_a = len(folds_A)
    n_b = len(folds_B)

    mean_a = np.mean(folds_A)
    mean_b = np.mean(folds_B)

    var_a = np.var(folds_A, ddof=1)  # ddof=1 for sample variance
    var_b = np.var(folds_B, ddof=1)

    # 1. Calculate Mutual Information
    mi = mean_a - mean_b

    # 2. Calculate Standard Error of the MI
    se_mi = np.sqrt((var_a / n_a) + (var_b / n_b))

    # 3. Calculate Significance using Welch's t-test
    # equal_var=False enforces Welch's t-test instead of Student's
    t_stat, p_val = stats.ttest_ind(folds_A, folds_B, equal_var=False)

    print(f"Condition A Mean: {mean_a:.4f} (n={n_a})")
    print(f"Condition B Mean: {mean_b:.4f} (n={n_b})")
    print("-" * 30)
    print(f"Estimated MI: {mi:.4f} nats/bits")
    print(f"Standard Error: ±{se_mi:.4f}")
    print(f"Welch's p-value: {p_val:.4e}")

    return mi, se_mi, p_val

# --- Example Usage ---
condition_1 = "libri_autoreg"
condition_2 = "libri_pause"
# analyze_conditions(condition_1, condition_2)
calculate_aggregate_mi(condition_1, condition_2)