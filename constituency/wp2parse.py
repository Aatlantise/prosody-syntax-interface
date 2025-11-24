#!/usr/bin/env python3
"""
Experiment 4: Predict parse S from words W and prosody P to estimate H(S | W, P).

Usage (example):
python exp4_w_p2parse.py --data data/constituency_parse.json \
    --outdir outputs/exp4 --model_name_or_path t5-base --from_scratch
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import (
    T5Config,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

# Import your tokenizer builder and ProsodyT5 wrapper (we define ProsodyT5 below too)
from constituency.util import TokenizerBuilder


# -------------------------
# Model wrapper: ProsodyT5
# -------------------------
from transformers import T5ForConditionalGeneration


class ProsodyT5(T5ForConditionalGeneration):
    """
    T5 model whose encoder inputs are token embeddings + prosody embeddings.
    Prosody is a real-valued (pause, duration) vector per input token.
    """

    def __init__(self, config, prosody_dim: int = 2):
        super().__init__(config)
        self.prosody_dim = prosody_dim
        # small projection from prosody features -> d_model
        self.prosody_proj = torch.nn.Linear(self.prosody_dim, config.d_model)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pause=None,        # torch.FloatTensor [B, T]
        duration=None,     # torch.FloatTensor [B, T]
        labels=None,
        **kwargs
    ):
        # Build inputs_embeds for encoder: token_emb + prosody_emb
        if input_ids is None and "inputs_embeds" in kwargs:
            # user provided inputs_embeds directly
            inputs_embeds = kwargs.pop("inputs_embeds")
        else:
            # standard token embeddings
            inputs_embeds = self.encoder.embed_tokens(input_ids)

        # build prosody tensor if provided
        if (pause is not None) and (duration is not None):
            # ensure float tensors on same device
            pause = pause.to(inputs_embeds.device)
            duration = duration.to(inputs_embeds.device)
            pros = torch.stack([pause, duration], dim=-1)  # [B, T, 2]
            prosody_emb = self.prosody_proj(pros)          # [B, T, d_model]
            inputs_embeds = inputs_embeds + prosody_emb

        # labels = getattr(kwargs, 'labels', None)

        # Call parent forward using inputs_embeds (and not input_ids)
        return super().forward(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )


# -------------------------
# Utilities: load data
# -------------------------
def load_jsonl_w_p(path: str):
    """
    Expects JSONL with fields: 'text' (source string), 'pause' (list floats),
    'duration' (list floats), and 'parse' (target parse string).
    Returns list[dict].
    """
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text") or obj.get("sentence") or obj.get("source")
            parse = obj.get("parse") or obj.get("target")
            pause = obj.get("pause")
            duration = obj.get("duration")
            if text is None or parse is None or pause is None or duration is None:
                continue
            items.append({
                "text": text,
                "parse": parse,
                "pause": pause,
                "duration": duration,
            })
    return items


# -------------------------
# Preprocess: tokenize and attach prosody (simple alignment)
# -------------------------
def preprocess(tokenizer, examples, max_source_length: int, max_target_length: int):
    """
    Tokenizes sources (text) with tokenizer; tokenizes targets (parse) as labels.
    Aligns prosody sequences by truncating / padding to the tokenized source length.
    NOTE: This uses a *simple* alignment: prosody vectors are assumed to be
    one-per-source-token. If your tokenizer does subword-splitting, you'll
    need a more careful alignment (not implemented here).
    """
    texts: List[str] = examples["text"]
    parses: List[str] = examples["parse"]
    pauses: List[List[float]] = examples["pause"]
    durations: List[List[float]] = examples["duration"]

    # encode sources
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_source_length,
        padding="max_length",
    )

    # prepare prosody: pad/truncate each prosody to match length of input_ids
    proc_pros = []
    for i, input_ids in enumerate(enc["input_ids"]):
        seq_len = len(input_ids)
        p = pauses[i]
        d = durations[i]
        # zip into pairs and truncate/pad
        pairs = list(zip(p, d))
        if len(pairs) >= seq_len:
            pairs = pairs[:seq_len]
        else:
            pad_amt = seq_len - len(pairs)
            pairs = pairs + [(0.0, 0.0)] * pad_amt
        proc_pros.append(pairs)

    # tokenize targets (parse) as labels
    with tokenizer.as_target_tokenizer():
        lab = tokenizer(
            parses,
            truncation=True,
            max_length=max_target_length,
            padding="max_length"
        )

    # mask labels pad token -> -100
    labels = []
    for seq in lab["input_ids"]:
        labels.append([ (x if x != tokenizer.pad_token_id else -100) for x in seq ])

    # return everything; HF Dataset.map expects lists per key
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "pause": proc_pros,
        "duration": [[pair[1] for pair in p] for p in proc_pros],  # list of durations aligned
        "pause_only": [[pair[0] for pair in p] for p in proc_pros],
        "labels": labels,
    }


# -------------------------
# Collator for Trainer
# -------------------------
class WPCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # batch is a list of examples (dicts with keys as returned by preprocess)
        input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)
        attention_mask = torch.tensor([ex["attention_mask"] for ex in batch], dtype=torch.long)
        # prosody lists -> tensors
        pause = torch.tensor([ex["pause"] for ex in batch], dtype=torch.float)       # shape [B, T, 2] or we stored separately
        # our preprocess stored pause,duration separately; reconstruct both
        # we used 'pause' as pairs; but also produced pause_only and duration separately; handle both possibilities:
        if isinstance(batch[0]["pause"][0], (list, tuple)):
            # pause is list of pairs
            pros = torch.tensor([ex["pause"] for ex in batch], dtype=torch.float)  # [B, T, 2]
            pause_tensor = pros[..., 0]
            duration_tensor = pros[..., 1]
        else:
            pause_tensor = torch.tensor([ex["pause"] for ex in batch], dtype=torch.float)
            duration_tensor = torch.tensor([ex["duration"] for ex in batch], dtype=torch.float)

        labels = torch.tensor([ex["labels"] for ex in batch], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pause": pause_tensor,
            "duration": duration_tensor,
            "labels": labels,
        }


# -------------------------
# Compute per-sequence NLLs
# -------------------------
def compute_sequence_nlls(model, dataset, tokenizer, device="cuda", batch_size=8):
    model.eval()
    model.to(device)

    nlls = []
    tok_counts = []

    def collate_fn(ex_list):
        input_ids = torch.tensor([ex["input_ids"] for ex in ex_list], dtype=torch.long)
        attention_mask = torch.tensor([ex["attention_mask"] for ex in ex_list], dtype=torch.long)
        # prosody arrays
        pause = torch.tensor([ex["pause"] for ex in ex_list], dtype=torch.float)
        if isinstance(ex_list[0]["pause"][0], (list, tuple)):
            pause = torch.tensor([ [p for p in ex["pause"]] for ex in ex_list], dtype=torch.float)[...,0]
            duration = torch.tensor([ [p for p in ex["pause"]] for ex in ex_list], dtype=torch.float)[...,1]
        else:
            duration = torch.tensor([ex["duration"] for ex in ex_list], dtype=torch.float)

        labels = torch.tensor([ex["labels"] for ex in ex_list], dtype=torch.long)

        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "pause": pause.to(device),
            "duration": duration.to(device),
            "labels": labels.to(device),
        }

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    with torch.no_grad():
        for batch in tqdm(loader, desc="NLL eval"):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pause=batch["pause"],
                duration=batch["duration"],
                labels=batch["labels"],
            )
            logits = outputs.logits  # (B, T, V)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            labels = batch["labels"]
            mask = (labels != -100)
            labels_exp = labels.unsqueeze(-1).clamp(min=0)
            gold_log_probs = torch.gather(log_probs, dim=-1, index=labels_exp).squeeze(-1)
            gold_log_probs_masked = gold_log_probs * mask
            seq_nll = -gold_log_probs_masked.sum(dim=1).cpu().numpy()
            seq_len = mask.sum(dim=1).cpu().numpy()
            nlls.extend(seq_nll.tolist())
            tok_counts.extend(seq_len.tolist())

    return nlls, tok_counts


# -------------------------
# Main
# -------------------------
def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading JSONL data...", args.data)
    items = load_jsonl_w_p(args.data)
    if len(items) == 0:
        raise SystemExit("No examples found in " + args.data)
    print(f"Loaded {len(items)} examples")

    # create HF dataset
    ds = Dataset.from_list(items)
    if args.validation_split > 0.0:
        ds = ds.train_test_split(test_size=args.validation_split, seed=42)
        train_ds = ds["train"]
        eval_ds = ds["test"]
    else:
        train_ds = ds
        eval_ds = None

    # tokenizer (reuse experiment 2 tokenizer)
    print("Building tokenizer...")
    builder = TokenizerBuilder("t5-base")
    tokenizer = builder.build_tokenizer()

    # ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    print("Preprocessing (tokenize targets, align prosody to source length)...")
    preprocess_fn = lambda examples: preprocess(tokenizer, examples, args.max_source_length, args.max_target_length)
    tokenized_train = train_ds.map(preprocess_fn, batched=True, remove_columns=train_ds.column_names)
    tokenized_eval = eval_ds.map(preprocess_fn, batched=True, remove_columns=eval_ds.column_names) if eval_ds else None

    print("Train/Val sizes:", len(tokenized_train), len(tokenized_eval) if tokenized_eval else 0)

    # Create model: from_scratch or from_pretrained
    if args.from_scratch:
        print("Initializing model from scratch using config:", args.model_name_or_path)
        config = T5Config.from_pretrained(args.model_name_or_path)
        config.vocab_size = len(tokenizer)
        model = ProsodyT5(config)
    else:
        print("Loading pretrained model and adding prosody head:", args.model_name_or_path)
        model = ProsodyT5.from_pretrained(args.model_name_or_path)

    # resize embeddings to tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # collator
    collator = WPCollator(tokenizer)

    # training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(outdir / "model"),
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        predict_with_generate=False,
        eval_strategy="steps" if args.eval_steps > 0 else "no",
        eval_steps=args.eval_steps if args.eval_steps > 0 else None,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        fp16=args.fp16,
        remove_unused_columns=False,
        report_to=["tensorboard"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=collator,
    )

    # Train
    print("Starting training...")
    trainer.train()
    trainer.save_model(str(outdir / "model_final"))

    # Evaluate: trainer.evaluate gives avg nats/token
    if tokenized_eval:
        print("Running trainer.evaluate() ...")
        res = trainer.evaluate(eval_dataset=tokenized_eval)
        print("Eval loss (nats/token):", res.get("eval_loss"))
        print("Eval loss (bits/token):", (res.get("eval_loss") / math.log(2)) if res.get("eval_loss") else None)

    # Compute per-sequence NLLs
    if tokenized_eval:
        print("Computing per-sequence NLLs ...")
        nlls, tok_counts = compute_sequence_nlls(model, tokenized_eval, tokenizer, device=args.device, batch_size=args.nll_batch_size)
        out_nll = outdir / "eval_sequence_nlls.jsonl"
        with open(out_nll, "w", encoding="utf-8") as fout:
            for nll, tok in zip(nlls, tok_counts):
                fout.write(json.dumps({
                    "nll_nats": float(nll),
                    "tokens": int(tok),
                    "bits_per_sequence": float(nll / math.log(2)),
                    "bits_per_token": float((nll / tok) / math.log(2)) if tok > 0 else None
                }) + "\n")
        print("Saved per-sequence NLLs to", out_nll)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/home/jm3743/prosody-syntax-interface/data/constituency_corpus.json")
    parser.add_argument("--outdir", type=str, default="outputs/wp2parse")
    parser.add_argument("--model_name_or_path", type=str, default="t5-base")
    parser.add_argument("--from_scratch", action="store_true",
                        help="If set, initialize model weights randomly from the config (no pretrained weights).")
    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--nll_batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--validation_split", type=float, default=0.02)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
