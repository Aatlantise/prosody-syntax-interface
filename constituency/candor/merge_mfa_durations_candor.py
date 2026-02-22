import pandas as pd
from pathlib import Path
import string
from glob import glob


def clean_word(word):
    """Lowercases, strips whitespace, and removes punctuation for comparison."""
    if pd.isna(word):
        return ""
    return str(word).strip().lower().translate(str.maketrans('', '', string.punctuation))


def merge_mfa_timestamps(features_path, mfa_base_dir):
    # 1. Load the main features file
    features_df = pd.read_csv(features_path)

    # 2. Initialize MFA columns with STT fallback timestamps
    features_df['mfa_start'] = features_df['word_start']
    features_df['mfa_stop'] = features_df['word_stop']

    mfa_dir = Path(mfa_base_dir)
    mismatched_turns = []
    mismatched_tokens = []
    missing_files = []

    # 3. Group by transcript_name (convo_id), speaker, and turn_id
    grouping_cols = ['transcript_name', 'speaker', 'turn_id']
    for (convo_id, speaker_id, turn_id), turn_group in features_df.groupby(grouping_cols):

        # Construct the expected MFA file path
        # Format: mfa/speaker_id/convo_id_turn_id.csv
        mfa_filename = f"{convo_id}_{turn_id}.csv"
        mfa_filepath = mfa_dir / str(speaker_id) / mfa_filename

        if not mfa_filepath.exists():
            missing_files.append(mfa_filepath)
            continue

        # Load the MFA file (filtering for 'words' just in case)
        mfa_df = pd.read_csv(mfa_filepath)
        mfa_df = mfa_df[mfa_df['Type'] == 'words'].reset_index(drop=True)

        # 4. Check if the word counts are equivalent for the turn
        if len(turn_group) == len(mfa_df):
            # 5. Token-by-token alignment within the matching turn
            for idx, (f_idx, f_row) in enumerate(turn_group.iterrows()):
                mfa_row = mfa_df.iloc[idx]
                mfa_label = str(mfa_row['Label']).strip()
                stt_clean = clean_word(f_row['word'])

                # Check for match or <unk>
                if mfa_label == stt_clean or mfa_label == '<unk>':
                    features_df.at[f_idx, 'mfa_start'] = mfa_row['Begin']
                    features_df.at[f_idx, 'mfa_stop'] = mfa_row['End']
                else:
                    mismatched_tokens.append({
                        'transcript_name': convo_id,
                        'turn_id': turn_id,
                        'stt_word': stt_clean,
                        'mfa_label': mfa_label
                    })
        else:
            # Turn length mismatch: Flag and skip (leaves STT timestamps)
            mismatched_turns.append({
                'transcript_name': convo_id,
                'turn_id': turn_id,
                'speaker': speaker_id,
                'features_len': len(turn_group),
                'mfa_len': len(mfa_df)
            })

    # 6. Report on the alignment health
    print(f"Processed {len(features_df.groupby(['transcript_name', 'turn_id']))} turns.")
    if missing_files:
        print(f"Warning: {len(missing_files)} MFA files were not found.")
    if mismatched_turns:
        print(f"Warning: {len(mismatched_turns)} turns had mismatched word counts and kept STT timings.")
    if mismatched_tokens:
        print(f"Warning: {len(mismatched_tokens)} individual tokens mismatched and kept STT timings.")

    return features_df, mismatched_turns, mismatched_tokens

if __name__ == "__main__":
    convo_paths = glob("/home/jm3743/data/candor_full_media/*")
    for convo_path in convo_paths:
        convo_id = convo_path.split("/")[-1]
        features_path = f"{convo_path}/word_level_features.csv"
        mfa_base_dir = f"/home/jm3743/data/candor/mfa/post_alignment/{convo_id}"

        merged_df, turn_errors, token_errors = merge_mfa_timestamps(features_path, mfa_base_dir)
        merged_df.to_csv(f"{convo_path}/features_mfa_merged.csv", index=False)
