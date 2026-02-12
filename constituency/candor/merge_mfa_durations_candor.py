import glob
import pandas as pd
import string


def clean(s: str) -> str:
    return str(s).lower().strip(string.punctuation)


if __name__ == "__main__":

    ds = []
    for filename in glob.glob("../data/candor/mfa/post_alignment/*/*/*.csv"):
        turn_id = int(filename.split("_")[-1].replace(".csv", ""))
        conversation_id = filename.split("/")[-3]
        d = pd.read_csv(filename)
        d = d[d["Type"] == "words"]
        d["word_pos"] = list(range(len(d)))
        d["word_count"] = len(d)
        d["turn_id"] = turn_id
        d["transcript_name"] = conversation_id
        ds.append(d)

    df_mfa = pd.concat(ds)
    df_mfa = df_mfa.rename(
        columns={
            "Begin": "start_time_within_turn",
            "End": "end_time_within_turn",
            "Speaker": "speaker",
        }
    )
    df_mfa = df_mfa.sort_values(
        by=["transcript_name", "turn_id", "word_pos"], kind="stable"
    )
    df_mfa["duration"] = (
        df_mfa["end_time_within_turn"] - df_mfa["start_time_within_turn"]
    )

    for i in range(0, 5):

        df = pd.read_csv(
            f"../output/candor/merged/gpt2/{i}/backbiter/candor_merged_data_gpt2_{i}_backbiter_confirm.csv",
        )
        df = df.rename(
            columns={
                "pre_word_pause": "pre_word_pause_candor",
                "post_word_pause": "post_word_pause_candor",
                "start_time": "start_time_candor",
                "end_time": "end_time_candor",
            }
        )
        df["word_pos"] = (
            df.groupby(["transcript_name", "turn_id"]).cumcount().reset_index(drop=True)
        )
        df["word_count"] = (
            df.groupby(["transcript_name", "turn_id"])["word"]
            .transform("size")
            .reset_index(drop=True)
        )
        print(f"Original df length: {len(df)}")

        df_mrg = df.merge(
            df_mfa,
            on=["transcript_name", "turn_id", "word_pos"],
            suffixes=["_candor", ""],
            how="left",
        )

        df_mrg["start_time"] = df_mrg["start_time_within_turn"] + df_mrg.groupby(
            ["transcript_name", "turn_id"]
        )["start_time_candor"].transform("min")
        df_mrg["end_time"] = df_mrg["end_time_within_turn"] + df_mrg.groupby(
            ["transcript_name", "turn_id"]
        )["start_time_candor"].transform("min")

        df_mrg["pre_word_pause"] = (
            df_mrg["start_time"]
            - df_mrg.groupby("transcript_name")["end_time"].shift(1)
        ).copy()
        df_mrg.loc[
            df_mrg.groupby("transcript_name").cumcount() == 0, "pre_word_pause"
        ] = 0.0

        df_mrg["post_word_pause"] = (
            df_mrg.groupby("transcript_name")["start_time"].shift(-1)
            - df_mrg["end_time"]
        ).copy()
        df_mrg["post_word_pause"] = df_mrg["post_word_pause"].fillna(0.0)

        df_mrg[(df_mrg["word_count_candor"] != df_mrg["word_count"])].to_csv(
            f"../output/candor/merged/gpt2/{i}/backbiter/candor_merged_data_gpt2_{i}_backbiter_confirm_mfa_errors.csv",
            index=False,
        )
        df_mrg = df_mrg[df_mrg["word_count"] == df_mrg["word_count_candor"]]

        df_mrg["discrepancy"] = (df_mrg["duration"] - df_mrg["duration_candor"]).abs()
        print(f"Merged df length: {len(df_mrg)}")
        df_mrg.to_csv(
            f"../output/candor/merged/gpt2/{i}/backbiter/candor_merged_data_gpt2_{i}_backbiter_confirm_mfa.csv",
            index=False,
        )

        print(
            f"Context Length {i}: {df_mrg.duration.isna().sum()} NA values in MFA duration"
        )
        print(df_mrg["discrepancy"].describe())
