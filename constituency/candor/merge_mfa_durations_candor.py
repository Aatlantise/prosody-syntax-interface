import os
import pandas as pd
from pathlib import Path
import string
from glob import glob
from tqdm.contrib.concurrent import process_map
import traceback
import re


def clean_words_vectorized(series):
    """Vectorized cleaning: Lowercases, strips whitespace, and removes punctuation."""
    # Create a regex pattern of all punctuation characters
    punct_pattern = f"[{re.escape(string.punctuation)}]"
    return series.fillna("").astype(str).str.strip().str.lower().str.replace(punct_pattern, "", regex=True)


def merge_mfa_timestamps(features_path, mfa_base_dir):
    features_df = pd.read_csv(features_path)

    features_df['mfa_start'] = features_df['word_start']
    features_df['mfa_stop'] = features_df['word_stop']

    mfa_dir = Path(mfa_base_dir)
    mismatched_turns = []
    mismatched_tokens = []
    missing_files = []

    # Pre-clean the entire STT word column at once (massive speedup)
    import re
    features_df['clean_word'] = clean_words_vectorized(features_df['word'])

    grouping_cols = ['transcript_name', 'speaker', 'turn_id']
    for (convo_id, speaker_id, turn_id), turn_group in features_df.groupby(grouping_cols):
        mfa_filename = f"{convo_id}_{turn_id}.csv"
        mfa_filepath = mfa_dir / str(speaker_id) / mfa_filename

        if not mfa_filepath.exists():
            missing_files.append(mfa_filepath)
            continue

        # Load MFA file
        mfa_df = pd.read_csv(mfa_filepath)
        mfa_df = mfa_df[mfa_df['Type'] == 'words'].reset_index(drop=True)

        if len(turn_group) == len(mfa_df):
            # Extract clean arrays for instant comparison
            stt_words = turn_group['clean_word'].values
            mfa_labels = mfa_df['Label'].fillna("").astype(str).str.strip().values

            # Create a boolean mask of matches
            match_mask = (mfa_labels == stt_words) | (mfa_labels == '<unk>')

            # 1. Update matching rows instantly using the mask
            matched_indices = turn_group.index[match_mask]
            features_df.loc[matched_indices, 'mfa_start'] = mfa_df['Begin'].values[match_mask]
            features_df.loc[matched_indices, 'mfa_stop'] = mfa_df['End'].values[match_mask]

            # 2. Log mismatched tokens using the inverted mask
            mismatched_idx = ~match_mask
            if mismatched_idx.any():
                for stt_w, mfa_l in zip(stt_words[mismatched_idx], mfa_labels[mismatched_idx]):
                    mismatched_tokens.append({
                        'transcript_name': convo_id,
                        'turn_id': turn_id,
                        'stt_word': stt_w,
                        'mfa_label': mfa_l
                    })
        else:
            mismatched_turns.append({
                'transcript_name': convo_id,
                'turn_id': turn_id,
                'speaker': speaker_id,
                'features_len': len(turn_group),
                'mfa_len': len(mfa_df)
            })

    # Drop the temporary cleaning column before returning
    features_df = features_df.drop(columns=['clean_word'])

    return features_df, mismatched_turns, mismatched_tokens, missing_files


def process_single_convo(convo_path):
    """Wrapper function for multiprocessing."""
    try:
        convo_id = convo_path.split("/")[-1]
        features_path = f"{convo_path}/word_level_features.csv"
        mfa_base_dir = f"/home/scratch/jm3743/candor/mfa/post_alignment/{convo_id}"

        # Skip if features file doesn't exist
        if not os.path.exists(features_path):
            return {"convo": convo_id, "status": "skipped - no features file"}

        merged_df, turn_errors, token_errors, missing = merge_mfa_timestamps(features_path, mfa_base_dir)
        merged_df.to_csv(f"{convo_path}/features_mfa_merged.csv", index=False)

        return {
            "convo": convo_id,
            "status": "success",
            "missing_mfa": len(missing),
            "turn_errors": len(turn_errors),
            "token_errors": len(token_errors)
        }
    except Exception as e:
        return {"convo": convo_id, "status": "failed", "error": traceback.format_exc()}


def main():
    convo_paths = glob("/home/scratch/jm3743/candor_full_media/*")

    # Check Slurm allocation, default to 4
    slurm_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))

    print(f"Merging MFA timestamps across {len(convo_paths)} conversations using {slurm_cores} cores...")

    results = process_map(
        process_single_convo,
        convo_paths,
        max_workers=slurm_cores,
        chunksize=10
    )

    # Quick Summary Report
    failures = [r for r in results if r["status"] == "failed"]
    if failures:
        print(f"\n[WARNING] {len(failures)} conversations failed to merge.")
        with open("mfa_merge_errors.log", "w") as f:
            for fail in failures:
                f.write(f"--- {fail['convo']} ---\n{fail['error']}\n\n")
    else:
        print("\n[SUCCESS] All available conversations merged perfectly.")


if __name__ == "__main__":
    main()