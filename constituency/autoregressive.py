"""
autoregressive.py

- Trains a Byte-Level BPE tokenizer on linearized constituency parses.
- Builds a GPT-2 style causal LM from scratch (random init).
- Fine-tunes / trains on the parse corpus.
- Evaluates average cross-entropy (bits per token equivalent can be derived).

Usage:
python autoregressive.py --parses data/parses.txt --outdir outputs/parse_gpt2 --epochs 3
"""

import argparse
import json
import os
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)
from datasets import load_dataset, Dataset
import math
from constituency.util import TokenizerBuilder

def get_tokenizer():
    builder = TokenizerBuilder("gpt2")
    tokenizer = builder.build_tokenizer()
    return tokenizer

def prepare_dataset(parses_path, hf_tokenizer, block_size=512):
    data = []
    try:
        with open(parses_path, 'r', encoding='utf-8') as f:
            # peek first line to decide format
            first = f.readline().strip()
            f.seek(0)
            if first.startswith("{"):
                # JSONL
                for line in f:
                    obj = json.loads(line)
                    parse = obj.get("parse") or obj.get("text") or None
                    if parse:
                        data.append({"text": parse})
            else:
                # plain text, one parse per line
                for line in f:
                    t = line.strip()
                    if t:
                        data.append({"text": t})
    except FileNotFoundError:
        raise

    ds = Dataset.from_list(data)

    def tokenize_fn(examples):
        # keep special tokens and do not add BOS/EOS automatically (we want model to predict)
        # return hf_tokenizer(examples["text"], truncation=True, max_length=block_size, return_attention_mask=False)
        return hf_tokenizer.batch_encode(examples["text"])

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    # group into blocks (not necessary if gen sequences are already sentence-per-example; keep one example per sequence)
    # For sentence-level modeling, do not concatenate; treat each parse as a sequence.
    return tokenized

def build_model(tokenizer, n_layer=12, n_head=12, n_embd=768, max_position_embeddings=1024):
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=max_position_embeddings,
        n_ctx=max_position_embeddings,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    model = GPT2LMHeadModel(config)
    return model

def compute_cross_entropy_per_sequence(model, tokenized_dataset, tokenizer, device='cpu'):
    """
    Compute average token cross-entropy over dataset.
    Returns avg_nats_per_token and avg_bits_per_token (bits = nats / ln(2)).
    Also returns per-sequence cross-entropies as list.
    """
    import torch
    model.eval()
    model.to(device)
    seq_xent = []
    # tokenized_dataset items expected to include 'input_ids'
    for example in tokenized_dataset:
        input_ids = torch.tensor(example["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            # outputs.loss is average negative log-likelihood per token (in nats if model uses natural log)
            loss = outputs.loss.item()  # this is in natural log units
            # but HF returns average negative log-likelihood per token (nats)
            seq_xent.append(loss * input_ids.size(1))  # sequence total nats
    total_tokens = sum(len(x["input_ids"]) for x in tokenized_dataset)
    total_nats = sum(seq_xent)
    avg_nats_per_token = total_nats / total_tokens
    avg_bits_per_token = avg_nats_per_token / math.log(2)
    return avg_nats_per_token, avg_bits_per_token

def load_data_and_model():
    # 1) Build or load tokenizer
    hf_tokenizer = get_tokenizer()

    # 2) Prepare dataset
    print("Preparing dataset...")
    tokenized = prepare_dataset(args.parses, hf_tokenizer, block_size=args.block_size)

    # The HF Trainer expects 'input_ids' (which we have).
    # Optionally split into train/valid if you have no explicit split.
    split = tokenized.train_test_split(test_size=args.validation_split, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"Train set has {len(train_ds)} sentences with "
          f"{sum([len(i) for i in train_ds['input_ids']]) / len(train_ds['input_ids'])} tokens per sentence")
    print(f"Eval set has {len(eval_ds)} sentences with "
          f"{sum([len(i) for i in eval_ds['input_ids']]) / len(eval_ds['input_ids'])} tokens per sentence")

    return hf_tokenizer, train_ds, eval_ds

def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hf_tokenizer, train_ds, eval_ds = load_data_and_model()

    # 3) Build model (random init)
    print("Building model...")
    model = build_model(hf_tokenizer, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, max_position_embeddings=args.block_size)
    model.resize_token_embeddings(len(hf_tokenizer))  # ensure embedding matrix matches tokenizer

    # 4) Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=hf_tokenizer, mlm=False)

    # 5) Training arguments
    training_args = TrainingArguments(
        output_dir=str(outdir / "model"),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_steps=args.eval_steps if args.eval_steps > 0 else None,
        eval_strategy="steps" if args.eval_steps > 0 else "no",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=500,
        fp16=args.fp16,
        metric_for_best_model="eval_loss",  # or any other metric you're computing
        greater_is_better=False,
        push_to_hub=False,
        report_to=["tensorboard"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds if args.eval_steps > 0 else None,
    )

    # 6) Train
    trainer.train()
    trainer.save_model(str(outdir / "model_final"))
    hf_tokenizer.save_pretrained(str(outdir / "tokenizer_final"))

    # 7) Evaluate average cross-entropy per token (approx H(S))
    print("Computing evaluation cross-entropy on evaluation set...")
    device = "cuda" if trainer.args.fp16 and torch.cuda.is_available() else "cpu"
    # we will compute sequence-level loss via Trainer.predict to use batching efficiently
    # Simpler: run trainer.evaluate to get avg loss (which is mean per-token nll)
    eval_result = trainer.evaluate(eval_dataset=eval_ds, metric_key_prefix="eval")
    # HF returns 'eval_loss' as average negative log-likelihood per token (nats)
    avg_nats_per_token = eval_result.get("eval_loss")
    avg_bits_per_token = avg_nats_per_token / math.log(2) if avg_nats_per_token is not None else None
    print("Eval loss (nats per token):", avg_nats_per_token)
    print("Eval (bits per token):", avg_bits_per_token)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parses", type=str, default="/home/jm3743/prosody-syntax-interface/data/constituency_corpus.json",
                        help="Path to parses file (JSONL with 'parse' or plain text lines).")
    parser.add_argument("--outdir", type=str, default="outputs/parse_lm", help="Output directory.")
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=8)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--validation_split", type=float, default=0.02)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    main(args)
