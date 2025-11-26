import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    T5Config, Seq2SeqTrainer, Seq2SeqTrainingArguments,
    PreTrainedTokenizerBase
)

from constituency.util import TokenizerBuilder
from constituency.model import DualEncoderT5  # your cleaned-up DualEncoder class

# -------------------------
# Tokenizer
# -------------------------
def get_tokenizer():
    builder = TokenizerBuilder("t5-base")
    tokenizer = builder.build_tokenizer()
    return tokenizer

# -------------------------
# Load JSONL with "text", "pause", "duration", "parse"
# -------------------------
def load_jsonl_text_prosody_parse(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "text" not in obj or "pause" not in obj or "duration" not in obj or "parse" not in obj:
                continue
            items.append({
                "text": obj["text"],
                "pause": obj["pause"],
                "duration": obj["duration"],
                "parse": obj["parse"],
            })
    return items

# -------------------------
# Preprocess: tokenize text + parse, keep prosody raw
# -------------------------
def preprocess(tokenizer: PreTrainedTokenizerBase, examples, max_source_length, max_target_length):
    inputs = examples["text"]
    targets = examples["parse"]

    model_inputs = tokenizer(
        inputs,
        truncation=True,
        max_length=max_source_length,
        padding="max_length"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            truncation=True,
            max_length=max_target_length,
            padding="max_length"
        )

    # Replace pad token id in labels with -100 for loss masking
    label_pad_token_id = -100
    labels_ids = [
        [(x if x != tokenizer.pad_token_id else label_pad_token_id) for x in seq]
        for seq in labels["input_ids"]
    ]

    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "pause": [[min(int(k * 100),255) for k in j] for j in examples["pause"]],
        "duration": [[min(int(k * 100),255) for k in j] for j in examples["duration"]],
        "labels": labels_ids,
    }

# -------------------------
# Collator
# -------------------------
class DualEncoderCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, device=None):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):

        # === TEXT SIDE ===
        input_ids = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
        attention_mask = [torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        # === PROSODY SIDE (repeat word-level features over subwords) ===
        prosody_list = []
        prosody_mask_list = []

        for ex in batch:
            # word-level features
            dur = ex["duration"]     # list of floats, len = num_words
            pau = ex["pause"]        # list of floats, len = num_words
            word_ids = ex["word_ids"]  # subword → word mapping

            # expand to subword-level
            dur_sub = []
            pau_sub = []
            mask_sub = []

            for w in word_ids:
                if w is None:
                    # special tokens
                    dur_sub.append(0.0)
                    pau_sub.append(0.0)
                    mask_sub.append(0)
                else:
                    dur_sub.append(float(dur[w]))
                    pau_sub.append(float(pau[w]))
                    mask_sub.append(1)

            prosody_list.append(torch.tensor(list(zip(dur_sub, pau_sub)), dtype=torch.float))
            prosody_mask_list.append(torch.tensor(mask_sub, dtype=torch.long))

        # pad to batch dimension
        prosody = pad_sequence(prosody_list, batch_first=True, padding_value=0.0)
        prosody_mask = pad_sequence(prosody_mask_list, batch_first=True, padding_value=0)

        # === LABELS ===
        labels = [torch.tensor(ex["labels"], dtype=torch.long) for ex in batch]
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prosody": prosody,             # shape: (B, T_subwords, 2) [duration, pause]
            # todo: pass one prosody feature at a time
            "prosody_mask": prosody_mask,   # shape: (B, T_subwords)
            "labels": labels,
        }

# -------------------------
# Main Experiment 4
# -------------------------
def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    items = load_jsonl_text_prosody_parse(args.data)
    print(f"Loaded {len(items)} examples.")

    ds = Dataset.from_list(items)
    ds = ds.train_test_split(test_size=args.validation_split, seed=42)
    train_ds = ds["train"]
    eval_ds = ds["test"]

    print("Loading tokenizer...")
    tokenizer = get_tokenizer()

    print("Preprocessing...")
    preprocess_fn = lambda ex: preprocess(tokenizer, ex, args.max_source_length, args.max_target_length)
    tokenized_train = train_ds.map(preprocess_fn, batched=True, remove_columns=train_ds.column_names)
    tokenized_eval = eval_ds.map(preprocess_fn, batched=True, remove_columns=eval_ds.column_names)

    print("Initializing DualEncoder model...")
    config = T5Config.from_pretrained(args.model_name)
    model = DualEncoderT5(config)
    model.to(args.device)

    # Ensure embeddings fit tokenizer
    model.resize_token_embeddings(len(tokenizer))

    collator = DualEncoderCollator(tokenizer, device=args.device)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(outdir / "model"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        weight_decay=0.01,
        fp16=args.fp16,
        remove_unused_columns=False,
        eval_strategy="steps" if args.eval_steps > 0 else "no",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
        report_to=["tensorboard"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
    )

    print("Training...")
    trainer.train()
    trainer.save_model(str(outdir / "model_final"))

    print("Evaluating...")
    eval_res = trainer.evaluate()
    print("Eval loss (nats/token):", eval_res["eval_loss"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/home/jm3743/prosody-syntax-interface/data/constituency_corpus.json")
    parser.add_argument("--outdir", type=str, default="outputs/wp_with_duration")
    parser.add_argument("--model_name", type=str, default="t5-base")
    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--validation_split", type=float, default=0.02)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
