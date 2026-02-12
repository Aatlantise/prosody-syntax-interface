import math
import pandas as pd
import numpy as np
import argparse
import json
import string
from intervaltree import IntervalTree, Interval
from sortedcontainers import SortedDict
import parselmouth

PUNCT = set(string.punctuation)


def validate_metadata(metadata: dict) -> bool:
    assert "speakers" in metadata
    assert len(metadata["speakers"]) == 2
    assert "channel" in metadata["speakers"][0]
    assert metadata["speakers"][0]["channel"] == "L"
    assert "user_id" in metadata["speakers"][0]
    assert "channel" in metadata["speakers"][1]
    assert metadata["speakers"][1]["channel"] == "R"
    assert "user_id" in metadata["speakers"][1]
    return True


def build_time_word_lookup(
    raw_transcript_file: str, metadata_file: str
) -> pd.DataFrame:

    print("build_time_word_lookup")

    with open(metadata_file) as f:
        metadata = json.load(f)
    valid = validate_metadata(metadata)

    ch_0_speaker = metadata["speakers"][0]["user_id"]
    ch_1_speaker = metadata["speakers"][1]["user_id"]

    records = []
    with open(raw_transcript_file) as f:
        data = json.load(f)
    for channel in data["results"]["channel_labels"]["channels"]:
        channel_label = channel["channel_label"]
        for item in channel["items"]:
            if item["type"] != "pronunciation":
                continue
            records.append(
                {
                    "start_time": float(item["start_time"]),
                    "end_time": float(item["end_time"]),
                    "duration": float(item["end_time"]) - float(item["start_time"]),
                    "word": item["alternatives"][0]["content"],
                    "channel": 0 if channel_label == "ch_0" else 1,
                    "speaker": (
                        ch_0_speaker if channel_label == "ch_0" else ch_1_speaker
                    ),
                    "confidence": float(item["alternatives"][0]["confidence"]),
                }
            )

    df = pd.DataFrame(records)

    # sorting by start time to ensure that post-word pauses are compared to chronologically next words
    df = df.sort_values("start_time").copy()

    df["pre_word_pause"] = (df["start_time"] - df["end_time"].shift(1)).copy()
    df["pre_word_pause"].iloc[0] = 0.0

    df["post_word_pause"] = (df["start_time"].shift(-1) - df["end_time"]).copy()
    df["post_word_pause"].iloc[-1] = 0.0

    return df


def posfinitemean(a):
    a = np.array(a)
    mask = np.isfinite(a) & (a > 0)
    return np.nan if len(a[mask]) == 0 else np.mean(a[mask])


def posfinitemin(a):
    a = np.array(a)
    mask = np.isfinite(a) & (a > 0)
    return np.nan if len(a[mask]) == 0 else np.min(a[mask])


def posfinitemax(a):
    a = np.array(a)
    mask = np.isfinite(a) & (a > 0)
    return np.nan if len(a[mask]) == 0 else np.max(a[mask])


def get_sound_features(df_surprisals, sound_file):
    def get_values(d, start, end):
        # Get keys in the interval [start, end], inclusive
        keys_in_range = d.irange(start, end, inclusive=(True, True))

        # Retrieve corresponding values
        values_in_range = [d[k] for k in keys_in_range]
        return values_in_range

    print("get_sound_features")

    snd = parselmouth.Sound(sound_file)

    ## Pitch
    pitch_dict0 = SortedDict()
    pitch_dict1 = SortedDict()

    # extract channel 0 (left)
    pitch = snd.extract_left_channel().to_pitch()
    pitch_values = pitch.selected_array["frequency"]
    pitch_times = pitch.xs()
    for time, val in zip(pitch_times, pitch_values):
        pitch_dict0[time] = val

    # extract channel 1 (right)
    pitch = snd.extract_right_channel().to_pitch()
    pitch_values = pitch.selected_array["frequency"]
    pitch_times = pitch.xs()
    for time, val in zip(pitch_times, pitch_values):
        pitch_dict1[time] = val

    ## Intensity
    intensity_dict0 = SortedDict()
    intensity_dict1 = SortedDict()

    # extract channel 0 (left)
    intensity = snd.extract_left_channel().to_intensity()
    intensity_values = intensity.values[0]
    intensity_times = intensity.xs()
    for time, val in zip(intensity_times, intensity_values):
        intensity_dict0[time] = val

    # extract channel 1 (right)
    intensity = snd.extract_right_channel().to_intensity()
    intensity_values = intensity.values[0]
    intensity_times = intensity.xs()
    for time, val in zip(intensity_times, intensity_values):
        intensity_dict1[time] = val

    df_surprisals["mean_pitch"] = list(
        map(
            lambda x, y, z: posfinitemean(
                get_values(pitch_dict0 if z == 0 else pitch_dict1, x, y)
            ),
            df_surprisals.word_start,
            df_surprisals.word_stop,
            df_surprisals.channel,
        )
    )

    df_surprisals["max_pitch"] = list(
        map(
            lambda x, y, z: posfinitemax(
                get_values(pitch_dict0 if z == 0 else pitch_dict1, x, y)
            ),
            df_surprisals.word_start,
            df_surprisals.word_stop,
            df_surprisals.channel,
        )
    )

    df_surprisals["min_pitch"] = list(
        map(
            lambda x, y, z: posfinitemin(
                get_values(pitch_dict0 if z == 0 else pitch_dict1, x, y)
            ),
            df_surprisals.word_start,
            df_surprisals.word_stop,
            df_surprisals.channel,
        )
    )

    df_surprisals["mean_intensity"] = list(
        map(
            lambda x, y, z: posfinitemean(
                get_values(intensity_dict0 if z == 0 else intensity_dict1, x, y)
            ),
            df_surprisals.word_start,
            df_surprisals.word_stop,
            df_surprisals.channel,
        )
    )

    df_surprisals["max_intensity"] = list(
        map(
            lambda x, y, z: posfinitemax(
                get_values(intensity_dict0 if z == 0 else intensity_dict1, x, y)
            ),
            df_surprisals.word_start,
            df_surprisals.word_stop,
            df_surprisals.channel,
        )
    )

    df_surprisals["min_intensity"] = list(
        map(
            lambda x, y, z: posfinitemin(
                get_values(intensity_dict0 if z == 0 else intensity_dict1, x, y)
            ),
            df_surprisals.word_start,
            df_surprisals.word_stop,
            df_surprisals.channel,
        )
    )

    return df_surprisals


def align_turn_times(lookup_df: pd.DataFrame, turn_df: pd.DataFrame) -> pd.DataFrame:

    i, j = 0, 0
    durations, word_starts, word_stops = [], [], []
    pre_pauses, post_pauses = [], []
    channels = []

    # monotonically align turn_df and lookup_df, allowing for
    # backchannel words in lookup_df and extra punctuation in turn_df
    last_i = 0
    while j < len(turn_df):

        if i >= len(lookup_df):
            print(f"i={i}, j={j}")
            print("lookup_df:", lookup_df.reset_index())
            print("turn_df:", turn_df.reset_index()[["word", "turn_start", "turn_stop"]])

        if not i < len(lookup_df):
            return None

        lookup_w = lookup_df.word.iloc[i]
        turn_w = str(turn_df.word.iloc[j])

        # if the corresponding entries contain the same word
        # need to account for the possible extra punctuation character in turn_w vs. lookup_w
        # ex: "$15," --> "$15"
        # however the lookup may contain punctuation in certain edge cases, and it doesn't necessarily
        # ex: "£500.." --> "£500."

        # if lookup_w in [turn_w, turn_w.rstrip(string.punctuation)]:
        if lookup_w == turn_w or (
            lookup_w == turn_w[:-1] and turn_w[-1] in string.punctuation
        ):
            durations.append(lookup_df.duration.iloc[i])
            word_starts.append(lookup_df.start_time.iloc[i])
            word_stops.append(lookup_df.end_time.iloc[i])
            pre_pauses.append(lookup_df.pre_word_pause.iloc[i])
            post_pauses.append(lookup_df.post_word_pause.iloc[i])
            channels.append(lookup_df.channel.iloc[i])

            i, j = i + 1, j + 1
            last_i = i

        # otherwise keep incrementing the lookup_df counter until a match is found or assertion fails
        else:
            i += 1

    assert len(durations) == len(turn_df.word)
    assert len(word_starts) == len(turn_df.word)
    assert len(word_stops) == len(turn_df.word)
    assert len(post_pauses) == len(turn_df.word)
    assert len(channels) == len(turn_df.word)

    turn_df["duration"] = durations
    turn_df["word_start"] = word_starts
    turn_df["word_stop"] = word_stops
    turn_df["pre_word_pause"] = post_pauses
    turn_df["post_word_pause"] = post_pauses
    turn_df["channel"] = channels
    return turn_df


def get_word_durations(
    df_surprisals: pd.DataFrame, time_word_lookup: pd.DataFrame
) -> pd.DataFrame:

    df_surprisals["uniq_word_id"] = list(range(len(df_surprisals)))
    results = []
    for grp, turn in df_surprisals.groupby(["turn_start", "speaker"]):
        lookup_chunk = time_word_lookup[
            (time_word_lookup.start_time >= turn.turn_start.head(1).item())
            & (time_word_lookup.end_time <= turn.turn_stop.head(1).item())
            & (time_word_lookup.speaker == turn.speaker.head(1).item())
        ].copy()

        result = align_turn_times(lookup_chunk, turn)
        if result is not None:
            results.append(result)
    return pd.concat(results)


def get_word_av_features(
    df_surprisals: pd.DataFrame, df_av: pd.DataFrame
) -> pd.DataFrame:

    print("get_word_av_features")

    gazes = []
    gazes_other = []
    intensities = []
    for i, row in df_surprisals.iterrows():
        if np.isnan(row.word_start) or np.isnan(row.word_stop):
            gazes.append(np.nan)
            intensities.append(np.nan)
            continue

        # gaze of speaker
        av_chunk = df_av[
            (df_av.timedelta >= math.floor(row.word_start))
            & (df_av.timedelta <= math.ceil(row.word_stop))
            & (df_av.user_id == row.speaker)
        ]
        gazes.append(av_chunk.gaze_on.mean())
        intensities.append(av_chunk.intensity.mean())

        # gaze of listener
        av_chunk = df_av[
            (df_av.timedelta >= math.floor(row.word_start))
            & (df_av.timedelta <= math.ceil(row.word_stop))
            & (df_av.user_id != row.speaker)
        ]
        gazes_other.append(av_chunk.gaze_on.mean())

    df_surprisals["gaze_on"] = gazes
    df_surprisals["gaze_on_other"] = gazes_other
    df_surprisals["intensity"] = intensities
    return df_surprisals


def get_word_backchannels(
    df_surprisals: pd.DataFrame, df_turns: pd.DataFrame, df_audiophile: pd.DataFrame
) -> pd.DataFrame:

    print("get_word_backchannels")

    backchannels = []
    tree = IntervalTree()
    df_bc = df_turns[df_turns.backchannel_count > 0]

    # for each turn (t_start, t_end) in backbiter that contains a backchannel,
    # find the turns in audiophile made by the backchannel_speaker during (t_start, t_end),
    # add these turns as intervals to the interval tree
    # these are the backchannels

    for bc_start, bc_stop, bc_speaker, bc_count in zip(
        df_bc.backchannel_start,
        df_bc.backchannel_stop,
        df_bc.backchannel_speaker,
        df_bc.backchannel_count,
    ):
        # tree.add(Interval(bc_start, bc_stop))
        df_sub = df_audiophile[
            (df_audiophile.start >= bc_start)
            & (df_audiophile.stop <= bc_stop)
            & (df_audiophile.speaker == bc_speaker)
        ]

        assert len(df_sub) == bc_count

        for a, b, utt in zip(df_sub.start, df_sub.stop, df_sub.utterance):
            tree.add(Interval(a, b, utt))

    df_surprisals["backchannel_overlap"] = list(
        map(
            lambda x, y: (
                tree.overlaps(x, y) if not (np.isnan(x) or np.isnan(y)) else np.nan
            ),
            df_surprisals.word_start,
            df_surprisals.word_stop,
        )
    )
    df_surprisals.backchannel_overlap = df_surprisals.backchannel_overlap.astype(int)

    df_surprisals["backchannel_utterance"] = list(
        map(
            lambda x, y, z: (
                list(tree[x:y])[0].data
                if z != 0 and not (np.isnan(x) or np.isnan(y))
                else np.nan
            ),
            df_surprisals.word_start,
            df_surprisals.word_stop,
            df_surprisals.backchannel_overlap,
        )
    )

    df_surprisals["backchannel_offset"] = list(
        map(
            lambda x, y, z: (
                list(tree[x:y])[0].begin
                - x  # backchannel start time - word start time = offset
                if z != 0 and not (np.isnan(x) or np.isnan(y))
                else np.nan
            ),
            df_surprisals.word_start,
            df_surprisals.word_stop,
            df_surprisals.backchannel_overlap,
        )
    )

    return df_surprisals


def add_word_features(args: argparse.Namespace) -> None:
    # read surprisals, AV features, and turns (containing backchannels)
    df_surprisals = pd.read_csv(args.surprisals_csv)

    if "transcript_name" in df_surprisals.columns:
        df_surprisals = df_surprisals[df_surprisals.transcript_name == args.convo_id]

    df_av = pd.read_csv(args.av_features_csv)
    df_av.timedelta = df_av.timedelta.apply(lambda x: pd.to_timedelta(x).seconds)
    df_turns = pd.read_csv(args.turns_csv)
    df_audiophile = pd.read_csv(args.audiophile_csv)

    # build a lookup of words and timestamps
    time_word_lookup = build_time_word_lookup(args.transcript_json, args.metadata_json)

    # add word start, end, and durations to the word-by-word surprisals df
    df_surprisals = get_word_durations(df_surprisals, time_word_lookup)

    # add raw sound information (max/min/mean pitch and intensity)
    df_surprisals = get_sound_features(df_surprisals, args.sound_file)

    # add candor gaze (speaker and listener) and intenstiy to the df
    df_surprisals = get_word_av_features(df_surprisals, df_av)

    # add backchannel information to the df
    df_surprisals = get_word_backchannels(df_surprisals, df_turns, df_audiophile)

    # save to file
    df_surprisals["transcript_name"] = args.convo_id
    df_surprisals.to_csv(args.surprisals_features_csv, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript_json")
    parser.add_argument("--metadata_json")
    parser.add_argument("--surprisals_csv")
    parser.add_argument("--turns_csv")
    parser.add_argument("--audiophile_csv")
    parser.add_argument("--av_features_csv")
    parser.add_argument("--surprisals_features_csv")
    parser.add_argument("--convo_id")
    parser.add_argument("--sound_file")
    args = parser.parse_args()
    pd.set_option("display.max_rows", None)
    add_word_features(args)


if __name__ == "__main__":
    main()
