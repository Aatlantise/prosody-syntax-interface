import pandas as pd
import re

def load_data(filepath):
    """
    Tabular prosody data to df

    :param filepath:
    :return: dataframe
    """
    df = pd.read_csv(filepath, sep='\t', names=["start", "end", "token"], keep_default_na=False)
    return df

def load_celex_syllables(celex_path="/home/jm3743/prosody-syntax-interface/data/celex.txt"):
    """
    Parses the CELEX .txt file to create a word -> syllable_count mapping.
    Assumes columns: Head\Class\StrsPat\...
    """
    syllable_map = {}

    print(f"Loading CELEX from {celex_path}...")

    with open(celex_path, 'r', encoding='utf-8') as f:
        # Skip header if it exists (assuming header starts with 'Head')
        first_line = f.readline()
        if not first_line.startswith('Head'):
            # If no header, reset pointer (or process line)
            f.seek(0)

        for line in f:
            parts = line.strip().split('\\')
            if len(parts) < 3:
                continue

            word = parts[0]
            strs_pat = parts[2]  # The pattern like "010"

            # Syllable count is simply the length of the stress pattern
            count = len(strs_pat)

            # Store lowercase for better matching
            syllable_map[word.lower()] = count

    print(f"Loaded {len(syllable_map)} words from CELEX.")
    return syllable_map

def count_syllables_heuristic(word):
    """
    Fallback function: Estimates syllables by counting vowel groups.
    Simple but effective for OOV words.
    """
    word = word.lower()
    if len(word) <= 3: return 1
    # Remove trailing 'e' (likely silent)
    if word.endswith('e'):
        word = word[:-1]
    elif word.endswith('ed'):
        word = word[:-2]
    # Count vowel groups (e.g., "oa" in "boat" counts as 1)
    vowels = re.findall(r'[aeiouy]+', word)
    return max(1, len(vowels))


def extract_examples_from_sent(df, syll_map):
    """
    Df to dictionary, including token, pause, and duration data

    :param df:
    :return:
    """
    current_context = []

    for i, row in df.iterrows():
        start, end, token = row["start"], row["end"], row["token"]
        if not token: # do not consider pause at this time
            if current_context:
                current_context[-1]["pause"] = row['end'] - row['start']
        else:
            current_context.append({
                "start": float(start),
                "end": float(end),
                "token": token,
                "pause": 0. # to be corrected in the following pause token when applicable
            })

    # Flatten into list of examples
    examples = []
    max_duration = 0.
    max_pause = 0.
    for i in range(len(current_context)):
        duration = current_context[i]["end"] - current_context[i]["start"]
        pause = current_context[i]["pause"]

        word = current_context[i]["token"]
        normalizer = syll_map.get(word, count_syllables_heuristic(word))
        rel_dur = duration / normalizer

        # get max duration and pause for future normalization purposes
        max_duration = max(max_duration, duration)
        max_pause = max(max_pause, pause)

        examples.append({
            "text": current_context[i]["token"],
            "duration": round(duration, 3),
            "pause": round(pause, 3),
            "rel_dur": round(rel_dur, 3),
        })
    return examples, max_duration, max_pause
