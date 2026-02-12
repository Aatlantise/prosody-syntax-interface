import json
import os
import csv
from copy import deepcopy
from datetime import datetime
from pydub import AudioSegment
import argparse
import pandas as pd
from glob import glob


candor_confirm_convos = [
    "d17b1fe2-009e-4c93-81c0-2db63c3f0287",
    "2699024b-93f4-48b8-ac08-3012274a47fa",
    "075926c6-9b88-42d2-8d03-f5fab051bfbb",
    "f366ef89-3945-415f-a670-4c7245bc7515",
    "5c54c47c-9c0a-4a9b-b82d-a5b353dc50f9",
    "52661f6d-1cce-49eb-bdeb-5ed40f19a273",
    "4a44dec7-65ac-4689-b957-6ba9e7c05b36",
    "2f72249c-39e1-4aab-a942-200e3ae2424c",
    "f1f6de10-a5a9-42a2-a9e5-dd969e37171b",
    "24562220-f4ce-4d8a-8862-e2d33769083e",
    "ddfa817a-e6cd-49e4-8861-f079c3ab26ac",
    "f30fb5e1-9bf8-462f-8c8b-fbfd849f10da",
    "b4e8fd0e-e2f6-44fe-919a-f4626908e4c4",
    "1f8de023-817f-4a7b-aefd-bcb1184a1d9a",
    "c1ab4807-f720-49ec-8c98-cf477661dee5",
    "8afc47b0-f2d9-4f4f-8b51-5c2342ac9c0a",
    "0b019c01-a6b7-4753-afa3-f7bf964932c9",
    "0a6ce575-4650-4942-97c6-f220de641a0c",
    "2174f1c1-9ab7-454b-ba48-a1ff70784398",
    "49694675-cacf-452d-a940-3c93987126ef",
    "4da5f1a9-e9ae-4837-802a-27c8b6675b4c",
    "a7f88eac-652d-4b0e-b86b-4e0666542e3e",
    "c52cd295-df74-4e99-b31f-34ae824a1418",
    "07c818b1-0fe0-4c8c-972b-184da443fa5e",
    "b9291f3e-8395-4803-a723-105db19cae02",
    "43f39618-a519-4bb6-b734-f834d0c641a2",
    "ea7de670-b567-41f3-91fd-46447acc4f2b",
    "d565e7d1-2faa-49e9-bee6-a1b44d9e4712",
    "e5e34d88-8e4c-49ce-b463-ef1932d89597",
    "89d49611-6307-425e-b47f-69d9b317573d",
    "49f6239a-252b-463c-a3b3-2667aeb24286",
    "949ad667-0831-443f-987c-78e61775fa00",
    "c135c056-e51a-40f5-ab96-cc9f217f6389",
    "5ca9b7cd-9c68-4803-8d13-f3e6c7d24f43",
    "01b9ad1a-cf5c-466b-b227-dfe3a36d1359",
    "f91c6e72-a099-44c6-bc79-5d6c01b1fa27",
    "34a130d2-e750-44d9-923c-97ccdc55f60d",
    "e42f6986-08cd-4ec2-9529-ddb47bdb674b",
    "8b0ecdda-eb47-420e-8a79-a41e64a40b09",
    "719582b9-9fbd-43bc-8891-76674a4e603e",
    "33f1c21c-b07a-485e-ba8e-43217e25ae00",
    "48ced667-0e23-4a32-b880-99c2504af593",
    "fae54970-a3a8-4386-8364-7d6ee91a6f4f",
    "7008bc9f-6fc4-4027-a2f2-22e91a520b9e",
    "20c24d28-cf75-43fc-890f-2dc6e33279d9",
    "7cfa2470-f77f-42a6-98b1-9e2e4ef61cdb",
    "22938143-89e2-445c-b3cc-d3503abeda83",
    "777e03b5-425e-43eb-af4f-d8dbb313c319",
    "725b5586-09b0-4701-a975-03bbd6b88edf",
    "c5e66cf8-159b-43d6-8cfd-0c2baed92c69",
]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript_pattern")
    parser.add_argument(
        "--output_dir",
        default="../data/candor/mfa/pre_alignment",
        help="directory to output data prepped for MFA",
    )
    args = parser.parse_args()

    csv_files = glob(args.transcript_pattern)

    for sample_file in csv_files:
        conversation_id = sample_file.split("/")[-3]

        if not conversation_id in candor_confirm_convos:
            continue

        print(f"Current file {sample_file}")
        file_list = []

        df = pd.read_csv(sample_file)

        audio = AudioSegment.from_file(
            f"../data/candor_om2/{conversation_id}/processed/{conversation_id}.mp3"
        )

        with open(
            f"../data/candor_om2/{conversation_id}/processed/channel_map.json"
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
    main()
