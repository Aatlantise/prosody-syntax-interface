import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader
import torch
import os
from tqdm import tqdm


def load_data(filepath):
    df = pd.read_csv(filepath, sep='\t', names=["start", "end", "token"], keep_default_na=False, header=0)
    return df

def extract_examples(df):
    current_context = []
    open_tags = []

    for i, row in df.iterrows():
        start, end, token = row["start"], row["end"], row["token"]

        if token.startswith('<') and token.endswith('>'):
            if '/' in token:
                tag = token.strip('</>')
                if open_tags and open_tags[-1] == tag:
                    # End tag: mark the previous real token as ending this phrase
                    for j in range(len(current_context) - 1, -1, -1):
                        if current_context[j]["token"]:
                            current_context[j]["label"].append(f"E-{tag}")
                            break
                    open_tags.pop()
            else:
                tag = token.strip('<>')
                open_tags.append(tag)
        elif not token: # do not consider pause at this time
            continue
        else:
            current_context.append({
                "start": float(start),
                "end": float(end),
                "token": token,
                "label": []  # May include "E-NP" or "E-VP"
            })

    # Flatten into list of examples
    examples = []
    for i in range(len(current_context)):
        text_context = " ".join(tok["token"] for tok in current_context[:i+1])
        duration = current_context[i]["end"] - current_context[i]["start"]
        label = 1 if "E-NP" in current_context[i]["label"] or "E-VP" in current_context[i]["label"] else 0
        examples.append({
            "text": text_context,
            "duration": duration,
            "label": label
        })
    return examples

def get_libritts_data():
    def get_data(split):
        examples = []
        split_path = f"/home/jm3743/data/LibriTTSLabelNPVP/lab/word/{split}/"
        for book in tqdm(os.listdir(split_path)):
            book_path = f"{split_path}/{book}/"
            for chapter in os.listdir(book_path):
                chapter_path = f"{book_path}/{chapter}/"
                for sentence in os.listdir(chapter_path):
                    sent_path = f"{chapter_path}/{sentence}"
                    df = load_data(sent_path)
                    examples += extract_examples(df)
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
            "duration": torch.tensor([item["duration"]], dtype=torch.float),
            "label": torch.tensor(item["label"], dtype=torch.float)
        }

def collate_fn(batch):
    texts = [x["text"] for x in batch]
    durations = torch.stack([x["duration"] for x in batch])
    labels = torch.stack([x["label"] for x in batch])
    return texts, durations, labels

if __name__ == "__main__":
    filepath = "sample.tsv"
    df = load_data(filepath)
    examples = extract_examples(df)
    print(json.dumps(examples, indent=4))