import json
import os
import csv
from copy import deepcopy
from datetime import datetime
from pydub import AudioSegment
import argparse
import pandas as pd
from glob import glob
from tqdm import tqdm


def main(sample_file):

    conversation_id = sample_file.split("/")[-3]

    df = pd.read_csv(sample_file)

    audio = AudioSegment.from_file(
        f"/home/jm3743/data/candor_full_media/{conversation_id}/processed/{conversation_id}.mp3"
    )

    with open(
        f"/home/jm3743/data/candor_full_media/{conversation_id}/processed/channel_map.json"
    ) as ch_map_file:
        ch_map = json.load(ch_map_file)
        ch_map_inv = {v: k for k, v in ch_map.items()}

    # Split stereo to left and right channels
    channels = audio.split_to_mono()
    left_channel = channels[0]
    right_channel = channels[1]

    for i, turn in df.iterrows():

        folder = f"{args.output_dir}/{conversation_id}/{turn.speaker}"
        os.makedirs(folder, exist_ok=True)
        with open(f"{folder}/{conversation_id}_{turn.turn_id}.txt", "w") as f:
            f.write(turn["utterance"])

        channel = (
            left_channel
            if ch_map_inv.get(turn["speaker"]) == "L"
            else right_channel
        )
        channel[turn["start"] * 1000 : turn["stop"] * 1000].export(
            f"{folder}/{conversation_id}_{turn.turn_id}.wav",
            format="wav",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript_pattern",
                        default="/home/jm3743/data/candor_full_media/*/transcription/transcript_audiophile.csv")
    parser.add_argument(
        "--output_dir",
        default="/home/jm3743/data/candor/mfa/pre_alignment",
        help="directory to output data prepped for MFA",
    )
    args = parser.parse_args()

    csv_list = glob(args.transcript_pattern)

    for sample_file in tqdm(csv_list):
        main(sample_file)
