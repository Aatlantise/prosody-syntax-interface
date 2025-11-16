"""
train_text2parse_seq2seq.py

- Fine-tune a pretrained encoder-decoder (T5/BART) to map sentence text -> linearized constituency parse.
- Save model & tokenizer.
- Report eval_loss (avg negative log-likelihood per token, in nats) and bits/token.
- Produce per-sequence negative log-likelihoods (nats) for downstream MI estimation.

Usage example:
python train_text2parse_seq2seq.py --data data/constituency_corpus.json \
    --outdir outputs/text2parse_t5 --model_name_or_path t5-base \
    --epochs 3 --per_device_train_batch_size 8 --fp16
"""

import argparse
import json
import math
import os
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
)
import numpy as np
import torch
from tqdm import tqdm
from constituency.util import TokenizerBuilder

def get_tokenizer():
    builder = TokenizerBuilder("t5-base")
    tokenizer = builder.build_tokenizer()
    return tokenizer

# -------------------------
# Utility: load data
# -------------------------
def load_jsonl_pairs(path):
    """
    Expects JSONL with at least 'text' (source words) and 'parse' (target linearized parse).
    Falls back to plain TSV/text if necessary.
    """
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        first = f.readline()
        f.seek(0)
        if first.strip().startswith("{"):
            for line in f:
                obj = json.loads(line)
                src = obj.get("text") or obj.get("sentence") or obj.get("source")
                tgt = obj.get("parse") or obj.get("target") or obj.get("parse_text")
                if src and tgt:
                    pairs.append({"text": src, "parse": tgt})
        else:
            # try TSV: src \t tgt per line, else treat each line as src and tgt same (unlikely)
            for line in f:
                if "\t" in line:
                    src, tgt = line.rstrip("\n").split("\t", 1)
                    if src and tgt:
                        pairs.append({"text": src, "parse": tgt})
                else:
                    # skip
                    continue
    return pairs

# -------------------------
# Tokenization map
# -------------------------
def preprocess(tokenizer, examples, max_source_length, max_target_length):
    # encoder inputs
    inputs = examples['text']
    targets = examples['parse']
    model_inputs = tokenizer(inputs, truncation=True, max_length=max_source_length, padding="max_length")

    # Tokenize targets with tokenizer as labels (for T5/BART this is fine)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, truncation=True, max_length=max_target_length, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    # convert pad token id to -100 so that loss ignores padding tokens
    label_pad_token_id = -100
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else label_pad_token_id) for l in seq] for seq in model_inputs["labels"]
    ]
    return model_inputs

# -------------------------
# Compute per-sequence NLLs
# -------------------------
def compute_sequence_nlls(model, dataset, tokenizer, device="cuda", batch_size=8):
    """
    Compute per-sequence negative log-likelihood (total nats) and token counts.
    Returns lists: nlls (nats), token_counts (tokens used in loss).
    Works by performing a forward pass with labels and summing log-probabilities for non-ignored tokens.
    """
    model.eval()
    model.to(device)
    nlls = []
    token_counts = []

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: {
        "input_ids": torch.tensor([ex["input_ids"] for ex in x], dtype=torch.long),
        "attention_mask": torch.tensor([ex["attention_mask"] for ex in x], dtype=torch.long),
        "labels": torch.tensor([ex["labels"] for ex in x], dtype=torch.long),
    })

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing NLLs"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # outputs.logits : (batch, tgt_seq_len, vocab)
            # HF returns outputs.loss (avg over non-ignored tokens)
            logits = outputs.logits  # shape (B, T, V)
            # shift logits & labels for seq2seq models if needed: HF models already align logits to labels provided (labels are decoder side)
            # We'll compute tokenwise log-probs:
            vocab_size = logits.size(-1)
            # compute log softmax
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # natural log (nats)

            # labels shape: (B, T) with -100 for padding
            # convert -100 -> some index (we'll mask)
            label_mask = (labels != -100)
            # gather log-probabilities of the gold tokens
            # To gather: expand labels to (B, T, 1)
            labels_exp = labels.unsqueeze(-1).clamp(min=0)  # replace -100 with 0 but mask will ignore
            gold_token_log_probs = torch.gather(log_probs, dim=-1, index=labels_exp).squeeze(-1)  # (B, T)
            # set masked positions to 0 for summation
            gold_token_log_probs_masked = gold_token_log_probs * label_mask

            # sum negative log-probs per sequence (nats)
            seq_nll = -gold_token_log_probs_masked.sum(dim=1).cpu().numpy()  # shape (B,)
            seq_tok_counts = label_mask.sum(dim=1).cpu().numpy()  # shape (B,)

            nlls.extend(seq_nll.tolist())
            token_counts.extend(seq_tok_counts.tolist())

    return nlls, token_counts

# -------------------------
# Main
# -------------------------
def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    pairs = load_jsonl_pairs(args.data)
    if len(pairs) == 0:
        raise SystemExit("No data loaded from %s" % args.data)
    print(f"Loaded {len(pairs)} examples.")

    # Convert to HF Dataset
    ds = Dataset.from_list(pairs)
    # Optionally split into train/val if you don't already have splits
    if args.validation_split > 0.0:
        ds = ds.train_test_split(test_size=args.validation_split, seed=42)
        train_ds = ds["train"]
        eval_ds = ds["test"]
    else:
        train_ds = ds
        eval_ds = None

    print("Loading tokenizer & model:", args.model_name_or_path)
    tokenizer = get_tokenizer()
    # Add POS tags and brackets as special tokens


    # Ensure pad token exists (T5 uses pad token by default)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # Preprocess datasets (tokenize)
    print("Tokenizing...")
    preprocess_fn = lambda examples: preprocess(tokenizer, examples, args.max_source_length, args.max_target_length)
    tokenized_train = train_ds.map(preprocess_fn, batched=True, remove_columns=train_ds.column_names)
    tokenized_eval = eval_ds.map(preprocess_fn, batched=True, remove_columns=eval_ds.column_names) if eval_ds is not None else None

    print(f"Train set has {len(tokenized_train)} sentences"
          f" with an average of {sum([len([i for i in k if i != -100]) for k in tokenized_train['labels']]) / len(tokenized_train)} tokens.")
    print(f"Eval set has {len(tokenized_eval)} sentences"
          f" with an average of {sum([len([i for i in k if i != -100]) for k in tokenized_eval['labels']]) / len(tokenized_eval)} tokens.")

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    # resize token embeddings if we added pad token
    model.resize_token_embeddings(len(tokenizer))

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=None)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(outdir / "model"),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=False,  # we use labels for loss
        eval_strategy="steps" if args.eval_steps > 0 else "no",
        eval_steps=args.eval_steps if args.eval_steps > 0 else None,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        fp16=args.fp16,
        warmup_steps=args.warmup_steps,
        remove_unused_columns=False,
        push_to_hub=False,
        metric_for_best_model="eval_loss",  # or any other metric you're computing
        greater_is_better=False,
        report_to=["tensorboard"]
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    trainer.train()
    trainer.save_model(str(outdir / "model_final"))
    tokenizer.save_pretrained(str(outdir / "tokenizer_final"))

    # Evaluation: average token-level cross-entropy from Trainer.evaluate()
    if tokenized_eval is not None:
        print("Running trainer.evaluate() to get avg eval loss (nats/token)...")
        eval_res = trainer.evaluate(eval_dataset=tokenized_eval)
        avg_nats_per_token = eval_res.get("eval_loss")
        avg_bits_per_token = (avg_nats_per_token / math.log(2)) if avg_nats_per_token is not None else None
        print(f"Eval loss (nats per token): {avg_nats_per_token}")
        print(f"Eval (bits per token): {avg_bits_per_token}")
    else:
        print("No evaluation dataset provided (validation_split=0). Skipping trainer.evaluate().")

    # Compute per-sequence NLLs on evaluation set if present (use smaller batch for safety)
    if tokenized_eval is not None:
        print("Computing per-sequence NLLs...")
        nlls, tok_counts = compute_sequence_nlls(model, tokenized_eval, tokenizer, device=args.device, batch_size=args.nll_batch_size)
        # Save per-example results aligned with eval dataset original metadata if available
        out_nll_path = outdir / "eval_sequence_nlls.jsonl"
        with open(out_nll_path, "w", encoding="utf-8") as fout:
            for i, (nll, n_tok) in enumerate(zip(nlls, tok_counts)):
                # convert to bits per token and bits per sequence
                bits_per_token = (nll / n_tok) / math.log(2) if n_tok > 0 else None
                bits_per_sequence = nll / math.log(2)
                fout.write(json.dumps({
                    "index": i,
                    "nll_nats": float(nll),
                    "tokens": int(n_tok),
                    "bits_per_token": bits_per_token,
                    "bits_per_sequence": bits_per_sequence,
                    # include source/target if you want:
                    "source": tokenized_eval[i]["input_ids"][:32],  # token ids snippet for reference
                    "labels_masked_len": int(n_tok),
                }) + "\n")
        print("Saved per-sequence NLLs to", str(out_nll_path))

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/home/jm3743/prosody-syntax-interface/data/constituency_corpus.json", help="Path to JSONL file with 'text' and 'parse' fields.")
    parser.add_argument("--outdir", type=str, default="outputs/text2parse", help="Output dir.")
    parser.add_argument("--model_name_or_path", type=str, default="t5-base", help="Pretrained seq2seq model (t5-base, facebook/bart-base, etc).")
    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--validation_split", type=float, default=0.02)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--nll_batch_size", type=int, default=8, help="Batch size for computing per-sequence NLLs")
    parser.add_argument("--device", type=str, default="cuda", help="Device for NLL computation")
    args = parser.parse_args()
    main(args)
