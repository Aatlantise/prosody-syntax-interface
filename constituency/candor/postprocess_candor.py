import pandas as pd
import numpy as np
import argparse
import functools
import nltk
import pyphen
from nltk.corpus import cmudict


# get the number of syllables in a word
# if the word is not in the CMU dictionary, approximate it using pyphen hyphenation
@functools.lru_cache
def get_syllable_count(word: str) -> int:
    if word in cmudic:
        return [
            len(list(y for y in x if y[-1].isdigit())) for x in cmudic[word.lower()]
        ][0]
    else:
        return len(pyphendic.inserted(word).split("-"))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--surp_csv")
    parser.add_argument("--out_csv")
    args = parser.parse_args()

    df = pd.read_csv(args.surp_csv)
    df = df.rename(columns={"word_start": "start_time", "word_stop": "end_time"})

    # add feature: semitones above 50 Hz
    df["max_pitch_semitones_above_50_hz"] = 12 * np.log2(df.max_pitch / 50.0)
    df["min_pitch_semitones_above_50_hz"] = 12 * np.log2(df.min_pitch / 50.0)
    df["mean_pitch_semitones_above_50_hz"] = 12 * np.log2(df.mean_pitch / 50.0)

    # Change in max pitch from one word to the next within a turn
    df["delta_max_pitch_semitones_above_50_hz"] = df.groupby(
        ["transcript_name", "turn_id"]
    )["max_pitch_semitones_above_50_hz"].diff()

    # Pitch range within a word (in semitones)
    df["range_pitch_semitones"] = 12 * np.log2(df.max_pitch / df.min_pitch)

    # Pitch range scaled by word duration (~rate of pitch change)
    df["range_pitch_semitones_per_second"] = df.range_pitch_semitones / df.duration

    # Change in pitch range from one word to the next within a turn
    df["delta_range_pitch_semitones"] = df.groupby(["transcript_name", "turn_id"])[
        "range_pitch_semitones"
    ].diff()

    # Change in rate of pitch change from one word to the next within a turn
    df["delta_range_pitch_semitones_per_second"] = df.groupby(
        ["transcript_name", "turn_id"]
    )["range_pitch_semitones_per_second"].diff()

    # get syllable count for each word
    df["syllable_count"] = df["word"].apply(lambda x: get_syllable_count(x))

    df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":

    # load CMU dictionary and pyphen dictionary as global variables
    nltk.download("cmudict")
    cmudic = cmudict.dict()
    pyphendic = pyphen.Pyphen(lang="en_US")

    main()
