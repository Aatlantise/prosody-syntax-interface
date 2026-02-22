import json
import os
import argparse
import pandas as pd
import numpy as np
import scipy.io.wavfile as wavfile
from pydub import AudioSegment
from glob import glob
from tqdm.contrib.concurrent import process_map
from functools import partial


def process_single_file(sample_file, output_dir):
    try:
        conversation_id = sample_file.split("/")[-3]
        df = pd.read_csv(sample_file)

        # Pre-create all speaker directories at once to avoid OS overhead in the loop
        unique_speakers = df['speaker'].unique()
        for speaker in unique_speakers:
            os.makedirs(f"{output_dir}/{conversation_id}/{speaker}", exist_ok=True)

        # Load channel map
        with open(
                f"/home/scratch/jm3743/candor_full_media/{conversation_id}/processed/channel_map.json") as ch_map_file:
            ch_map = json.load(ch_map_file)
            ch_map_inv = {v: k for k, v in ch_map.items()}

        # 1. Load the full MP3 once
        audio = AudioSegment.from_file(
            f"/home/scratch/jm3743/candor_full_media/{conversation_id}/processed/{conversation_id}.mp3"
        )
        sample_rate = audio.frame_rate

        # 2. Split to mono and extract raw NumPy arrays (Lightning fast!)
        channels = audio.split_to_mono()
        left_arr = np.array(channels[0].get_array_of_samples())
        right_arr = np.array(channels[1].get_array_of_samples())

        # 3. Iterate using itertuples() (much faster than iterrows())
        for turn in df.itertuples():
            folder = f"{output_dir}/{conversation_id}/{turn.speaker}"
            base_filename = f"{folder}/{conversation_id}_{turn.turn_id}"

            # Write text file
            with open(f"{base_filename}.txt", "w") as f:
                f.write(str(turn.utterance))

            # Determine which array to slice
            channel_arr = left_arr if ch_map_inv.get(turn.speaker) == "L" else right_arr

            # Calculate array indices based on the sample rate
            start_idx = int(turn.start * sample_rate)
            stop_idx = int(turn.stop * sample_rate)

            # Slice the numpy array and write raw WAV directly
            audio_slice = channel_arr[start_idx:stop_idx]
            wavfile.write(f"{base_filename}.wav", sample_rate, audio_slice)

        return {"file": sample_file, "status": "success"}

    except Exception as e:
        return {"file": sample_file, "status": "failed", "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transcript_pattern",
        default="/home/scratch/jm3743/candor_full_media/*/transcription/transcript_audiophile.csv"
    )
    parser.add_argument(
        "--output_dir",
        default="/home/scratch/jm3743/candor/mfa/pre_alignment",
        help="directory to output data prepped for MFA",
    )
    args = parser.parse_args()

    csv_list = glob(args.transcript_pattern)

    # Grab Slurm cores (defaulting to 4)
    slurm_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))

    # Use functools.partial to safely pass the output_dir argument to the workers
    worker_func = partial(process_single_file, output_dir=args.output_dir)

    print(f"Starting extraction on {len(csv_list)} files using {slurm_cores} cores...")

    results = process_map(
        worker_func,
        csv_list,
        max_workers=slurm_cores,
        chunksize=2  # Small chunksize since loading MP3s uses decent RAM
    )

    # Simple error reporting
    failures = [r for r in results if r["status"] == "failed"]
    if failures:
        print(f"\n[WARNING] {len(failures)} files failed. Check your data.")
        for f in failures[:5]:  # Print first 5 errors
            print(f" - {f['file']}: {f['error']}")


if __name__ == "__main__":
    main()