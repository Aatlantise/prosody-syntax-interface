import argparse
import json
import math
from pathlib import Path
from datasets import Dataset

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoTokenizer,
    T5Config,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from constituency.util import TokenizerBuilder
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config


class ProsodyT5(T5ForConditionalGeneration):
    """
    T5 model whose encoder only receives prosody embeddings:
    per-token pause/duration → linear layer → prosody embedding
    """

    def __init__(self, config: T5Config):
        super().__init__(config)

        # Size of prosody vector per token: pause + duration = 2
        self.prosody_dim = 2

        # Projection layer from real prosody → hidden size
        self.prosody_proj = nn.Linear(self.prosody_dim, config.d_model)

    def encode_with_prosody(
        self,
        attention_mask,
        pause,
        duration,
        **kwargs
    ):
        """
        Build encoder inputs of the form:
        embeddings = token_emb + pos_enc + prosody_emb
        """

        # 1. Build prosody feature tensor
        # pause/duration have shape [batch, seq]
        prosody_feats = torch.stack([pause, duration], dim=-1)  # -> [B, T, 2]

        # 2. Map to embedding space
        prosody_emb = self.prosody_proj(prosody_feats)          # [B, T, d_model]

        # 4. Pass to encoder
        return self.encoder(
            inputs_embeds=prosody_emb,
            attention_mask=attention_mask,
            **kwargs
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pause=None,
        duration=None,
        labels=None,
        **kwargs
    ):
        """
        Override forward to use prosody-aware encoding.
        """
        # For training and generation, the decoder stays unchanged.
        # Only the encoder has prosody embeddings added.

        # Build encoder_hidden_states using prosody
        encoder_outputs = self.encode_with_prosody(
            attention_mask=attention_mask,
            pause=pause,
            duration=duration,
        )

        # Extract hidden states

        # Now call the rest of the T5 forward using these encoder states
        return super().forward(
            input_ids=None,     # don't let parent recompute input_embeds
            attention_mask=attention_mask,
            encoder_outputs=(encoder_outputs,),
            labels=labels
        )



# -------------------------------------------------------
# Tokenizer (reuse from Experiment 2)
# -------------------------------------------------------
def get_tokenizer():
    builder = TokenizerBuilder("t5-base")
    tokenizer = builder.build_tokenizer()
    return tokenizer


# -------------------------------------------------------
# Load JSONL with "pause", "duration", and "parse"
# -------------------------------------------------------
def load_jsonl_prosody_parse(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "pause" not in obj or "duration" not in obj or "parse" not in obj:
                continue
            items.append({
                "pause": obj["pause"],
                "duration": obj["duration"],
                "parse": obj["parse"],
            })
    return items


# -------------------------------------------------------
# Preprocess: tokenize parse target, keep prosody raw
# -------------------------------------------------------
def preprocess(tokenizer, examples, max_target_length):
    targets = examples["parse"]

    # Tokenize the parse labels
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
        "pause": examples["pause"],
        "duration": examples["duration"],
        "labels": labels_ids,
    }

def get_data_properties(ex_list):
    pause = [torch.tensor(ex["pause"], dtype=torch.float) for ex in ex_list]
    duration = [torch.tensor(ex["duration"], dtype=torch.float) for ex in ex_list]

    pause = pad_sequence(pause, batch_first=True, padding_value=0.0)
    duration = pad_sequence(duration, batch_first=True, padding_value=0.0)
    attention_mask = (pause != 0.0).long()

    labels = torch.tensor([ex["labels"] for ex in ex_list], dtype=torch.long)
    return pause, duration, attention_mask, labels

# -------------------------------------------------------
# Custom collator (pad pause/duration + sequence2sequence items)
# -------------------------------------------------------
class ProsodyCollator:
    def __init__(self, tokenizer, device=None):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
        # pad 1D pause/duration sequences
        pause, duration, attention_mask, labels = get_data_properties(batch)

        return {
            "pause": pause,
            "duration": duration,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# -------------------------------------------------------
# NLL computation (same as Experiment 2, reuse)
# -------------------------------------------------------
def compute_sequence_nlls(model, dataset, tokenizer, device="cuda", batch_size=8):
    model.eval()
    model.to(device)

    nlls = []
    tok_counts = []

    def collate_fn(ex_list):
        pause, duration, attention_mask, labels = get_data_properties(ex_list)

        return {
            "pause": pause.to(device),
            "duration": duration.to(device),
            "attention_mask": attention_mask.to(device),
            "labels": labels.to(device),
        }

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    with torch.no_grad():
        for batch in loader:
            outputs = model(
                pause=batch["pause"],
                duration=batch["duration"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )

            logits = outputs.logits  # [B, T, V]
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


# -------------------------------------------------------
# Main Experiment 3 entrypoint
# -------------------------------------------------------
def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    items = load_jsonl_prosody_parse(args.data)
    print(f"Loaded {len(items)} examples.")

    ds = Dataset.from_list(items)

    # Split
    ds = ds.train_test_split(test_size=args.validation_split, seed=42)
    train_ds = ds["train"]
    eval_ds = ds["test"]


    print("Loading tokenizer...")
    tokenizer = get_tokenizer()

    print("Preprocessing...")
    preprocess_fn = lambda ex: preprocess(tokenizer, ex, args.max_target_length)
    tokenized_train = train_ds.map(preprocess_fn, batched=True, remove_columns=train_ds.column_names)
    tokenized_eval = eval_ds.map(preprocess_fn, batched=True, remove_columns=eval_ds.column_names) if eval_ds else None

    print(f"Eval set has {len(tokenized_eval)} sentences"
          f" with an average of {sum([len([i for i in k if i != -100]) for k in tokenized_eval['labels']]) / len(tokenized_eval)} tokens.")

    print("Initializing ProsodyT5 model...")
    config = T5Config.from_pretrained(args.model_name)
    model = ProsodyT5(config)

    # Ensure output embeddings fit tokenizer
    model.resize_token_embeddings(len(tokenizer))

    collator = ProsodyCollator(tokenizer)

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

    if tokenized_eval:
        print("Running trainer.evaluate()...")
        eval_res = trainer.evaluate()
        print("Eval loss (nats/token):", eval_res["eval_loss"])

        print("Computing sequence NLLs...")
        nlls, tok_counts = compute_sequence_nlls(model, tokenized_eval, tokenizer)
        out_nll = outdir / "eval_sequence_nlls.jsonl"
        with open(out_nll, "w") as f:
            for nll, t in zip(nlls, tok_counts):
                bits = nll / math.log(2)
                f.write(json.dumps({
                    "nll_nats": float(nll),
                    "tokens": int(t),
                    "bits_per_sequence": bits,
                    "bits_per_token": (nll/t)/math.log(2) if t>0 else None
                }) + "\n")

        print("Saved to", out_nll)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/home/jm3743/prosody-syntax-interface/data/constituency_corpus.json")
    parser.add_argument("--outdir", type=str, default="outputs/prosody2parse")
    parser.add_argument("--model_name", type=str, default="t5-base")
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=float, default=30)
    parser.add_argument("--validation_split", type=float, default=0.02)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    main(args)
