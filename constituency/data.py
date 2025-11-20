import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader
import torch
import os
from tqdm import tqdm


def load_data(filepath):
    df = pd.read_csv(filepath, sep='\t', names=["start", "end", "token"], keep_default_na=False)
    return df

def extract_examples_from_sent(df):
    current_context = []
    open_tags = []

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
        max_duration = max(max_duration, duration)
        max_pause = max(max_pause, pause)
        examples.append({
            "text": current_context[i]["token"],
            "duration": duration,
            "pause": pause
        })
    return examples, max_duration, max_pause

def get_libritts_data():
    def get_data(split):
        max_duration = 0.
        max_pause = 0.
        examples = []
        split_path = f"/home/jm3743/data/LibriTTSLabel/lab/word/{split}"
        for book in tqdm(os.listdir(split_path)):
            book_path = f"{split_path}/{book}"
            for chapter in os.listdir(book_path):
                chapter_path = f"{book_path}/{chapter}"
                for sentence in os.listdir(chapter_path):
                    sent_path = f"{chapter_path}/{sentence}"
                    df = load_data(sent_path)
                    ex, local_max_dur, local_max_pau = extract_examples_from_sent(df)
                    examples.append(ex)
                    max_duration = max(max_duration, local_max_dur)
                    max_pause = max(max_pause, local_max_pau)

        print(f"Maximum duration: {max_duration}")
        print(f"Maximum pause: {max_pause}")
        return examples

    train = get_data("train-clean-100")
    dev = get_data("dev-clean")
    test = get_data("test-clean")

    return train, dev, test

class PhraseBoundaryDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        return {
            "text": item["text"],
            # normalize to prevent tensor of zeros
            "duration": item["duration"],
            "pause": item["pause"]
        }

def collate_fn(batch):
    texts = [x["text"] for x in batch]
    durations = torch.stack([x["duration"] for x in batch])
    pauses = torch.stack([x["pause"] for x in batch])
    return texts, durations, pauses

def _test():
    filepath = "sample.tsv"
    df = load_data(filepath)
    examples, _, _ = extract_examples_from_sent(df)
    print(json.dumps(examples, indent=4))
    ex_ds = PhraseBoundaryDataset(examples)
    print(ex_ds)

def main():
    train_examples, val_examples, test_examples = get_libritts_data()
    _ = PhraseBoundaryDataset(train_examples)
    _ = PhraseBoundaryDataset(val_examples)
    _ = PhraseBoundaryDataset(test_examples)

if __name__ == "__main__":
    main()
