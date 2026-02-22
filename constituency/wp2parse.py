import argparse
from pathlib import Path
from datasets import Dataset
from transformers import (
    T5Config, Seq2SeqTrainer, Seq2SeqTrainingArguments,
    PreTrainedTokenizerBase, EarlyStoppingCallback, T5ForConditionalGeneration
)
from constituency.util import TokenizerBuilder, load_jsonl_data, preprocess
from constituency.model import DualEncoderT5, DualEncoderCollator
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader
import pandas as pd


def get_tokenizer():
    builder = TokenizerBuilder("t5-base")
    tokenizer = builder.build_tokenizer()
    return tokenizer


def compute_sequence_surprisals(model, tokenizer, dataset, collator, batch_size, device):
    """
    Computes the total surprisal (sum of NLL) for each sequence in the dataset.
    Returns a list of float values.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    surprisals = []

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    with torch.no_grad():
        for batch in loader:
            # Move batch to device

            input_ids = batch["input_ids"].to(device) if batch["input_ids"] is not None else None
            labels = batch["labels"].to(device)

            # Forward pass
            # Note: T5ForConditionalGeneration automatically shifts labels for decoding
            # but we need to check if your DualEncoderT5 does the same.
            # Assuming it inherits or behaves like T5:
            outputs = model(input_ids=input_ids, labels=labels)
            logits = outputs.logits

            # T5 Standard Loss Calculation Logic (Simulated with reduction='none')
            # Reshape logits to [Batch * Seq_Len, Vocab] and labels to [Batch * Seq_Len]
            # But to keep per-sequence sum, we work with [Batch, Seq_Len, Vocab]

            # Check dimensions match
            # Logits: [B, Seq_Len, Vocab], Labels: [B, Seq_Len]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # NOTE: T5 usually handles the shifting internally in the forward pass
            # and returns loss. However, since we need unreduced loss,
            # and 'outputs.loss' is already a mean scalar, we must recalculate
            # using outputs.logits.

            # Standard T5 behavior: it doesn't shift inside forward() for loss calculation
            # if we look at the source, but it depends on your DualEncoderT5 implementation.
            # SAFE BET: Calculate CrossEntropy on the full logits/labels provided.
            # If your model shifts internally, use logits/labels as is.
            # If it's standard T5, the labels are already aligned with logits.

            B, L, V = logits.shape

            # Flatten batch for CrossEntropy, then reshape back
            flat_logits = logits.view(-1, V)
            flat_labels = labels.view(-1)

            per_token_loss = loss_fct(flat_logits, flat_labels)

            # Reshape back to [Batch, Seq_Len]
            per_token_loss = per_token_loss.view(B, L)

            # Sum loss per sequence (masking is handled by ignore_index=-100 in loss_fct)
            # This gives total surprisal (nats) for the sequence
            seq_surprisal = per_token_loss.sum(dim=1)

            surprisals.extend(seq_surprisal.cpu().tolist())

    return surprisals

def single_run(args, tokenizer, tokenized_train, tokenized_eval):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Initializing DualEncoder model...")
    # Load base model + config first
    base = T5ForConditionalGeneration.from_pretrained(args.model_name)

    # Resize embeddings to new tokenizer
    base.resize_token_embeddings(len(tokenizer))

    # Update config so new model matches
    config = base.config
    config.vocab_size = len(tokenizer)

    # Now build your custom model
    model = DualEncoderT5(config)
    model.tokenizer = tokenizer

    # Load pretrained weights to only the encoder and shared (to be used as encoder embedding layer)
    print("Loading  weights...")
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    print("Re-initializing decoder layer, to be trained from scratch")
    model.decoder.init_weights()

    model.to(args.device)

    collator = DualEncoderCollator(tokenizer,
                                   device=args.device,
                                   return_text=args.use_text,
                                   return_pause=args.use_pause,
                                   return_duration=args.use_duration,
                                   return_zeros=args.use_zeros)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(outdir / "model"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=1,
        weight_decay=0.01,
        fp16=args.fp16,
        remove_unused_columns=False,
        eval_strategy="steps" if args.eval_steps > 0 else "no",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
        report_to=["tensorboard"],
        load_best_model_at_end = True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    trainer.model.floating_point_ops = lambda _: 0  # allow input_ids = None

    print("Training...")
    trainer.train()
    trainer.save_model(str(outdir / "model_final"))

    print("Evaluating (Mean Loss)...")
    eval_res = trainer.evaluate()  # Keep this for quick sanity check

    print("Calculating per-sequence surprisals...")
    # Use the helper function here
    # Ensure you pass the model from the trainer (it might be wrapped in DDP/DataParallel)
    model_to_use = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model

    seq_surprisals = compute_sequence_surprisals(
        model=model_to_use,
        tokenizer=tokenizer,
        dataset=tokenized_eval,
        collator=collator,
        batch_size=args.batch_size,
        device=args.device
    )

    return eval_res, seq_surprisals

def main(args):
    k = 10
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    print("Loading data...")
    items = load_jsonl_data(args.data, debug=args.debug)
    print(f"Loaded {len(items)} examples.")

    # Convert the list of items to a Hugging Face Dataset
    ds_full = Dataset.from_list(items)
    full_indices = list(range(len(ds_full)))  # Get the indices of the full dataset

    # --- Cross-Validation Loop ---
    # This loop will run K times, once for each fold

    all_results = []
    all_fold_level_metrics = []
    for fold, (train_index, eval_index) in enumerate(kf.split(full_indices)):
        print(f"\n--- Starting Fold {fold + 1}/{k} ---")

        # 1. Create Train and Evaluation Datasets for the current fold
        # Use the indices generated by KFold to select the data slices
        train_ds = ds_full.select(train_index)
        eval_ds = ds_full.select(eval_index)

        print(f"Fold {fold + 1} sizes: Train={len(train_ds)}, Eval={len(eval_ds)}")

        # 2. Preprocessing (as you had it)
        print("Loading tokenizer...")
        # NOTE: You might move get_tokenizer() outside the loop if it's slow,
        # but I'm keeping the original structure for clarity.
        tokenizer = get_tokenizer()

        print("Preprocessing...")
        preprocess_fn = lambda ex: preprocess(tokenizer, ex, args.max_source_length, args.max_target_length)

        # Map the preprocessing function onto the current fold's datasets
        tokenized_train = train_ds.map(preprocess_fn, batched=True, remove_columns=train_ds.column_names)
        tokenized_eval = eval_ds.map(preprocess_fn, batched=True, remove_columns=eval_ds.column_names)

        per_seq_len = sum([len([t for t in ex if t != -100]) for ex in tokenized_eval['labels']]) / len(tokenized_eval['labels'])
        print(f"Token length per parse sequence: {per_seq_len:.4f}")

        # 3. Model Training and Evaluation (Your next steps)
        # --- YOUR TRAINING/EVALUATION CODE GOES HERE ---
        eval_metrics, surprisals = single_run(args, tokenizer, tokenized_train, tokenized_eval)

        # 4. Store Results
        # We zip the specific indices used in this evaluation fold with their scores
        # 'eval_index' is the numpy array of original indices from the full dataset
        for orig_idx, score in zip(eval_index, surprisals):
            all_results.append({
                "original_index": int(orig_idx),
                "fold": fold + 1,
                "surprisal": score,
                # Optional: Add metadata from items if useful
                # "length": len(items[orig_idx]['text'].split())
            })

        fold_eval_loss = eval_metrics['eval_loss']
        print(f"Fold {fold + 1} Evaluation per token: {fold_eval_loss:.4f}")
        print(f"Fold {fold + 1} Entropy per sequence: {fold_eval_loss * per_seq_len:.4f}")
        all_fold_level_metrics.append(fold_eval_loss * per_seq_len)

    # --- Save Final Results ---
    print("Saving detailed results...")
    df = pd.DataFrame(all_results)

    # Sort by original index to look neat (optional)
    df = df.sort_values(by=["original_index", "fold"])

    # Save to CSV
    save_path = Path(args.outdir) / "cross_validation_results.csv"
    df.to_csv(save_path, index=False)
    print(f"Saved per-sequence surprisals to {save_path}")

    # Calculate aggregate stats from the dataframe
    mean_surprisal = df["surprisal"].mean()
    print(f"Global Mean Surprisal: {mean_surprisal:.4f}")

    # --- Final Results ---
    import numpy as np
    final_mean = np.mean(all_fold_level_metrics)
    final_std = np.std(all_fold_level_metrics)

    print("\n--- Cross-Validation Results ---")
    print(f"Individual Fold Metrics: {all_fold_level_metrics}")
    print(f"Mean Metric (Entropy/Loss): {final_mean:.4f}")
    print(f"Standard Deviation: {final_std:.4f}")
    print(f"Variance: {final_std**2:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/home/jm3743/prosody-syntax-interface/data/constituency_corpus_reldur.json")
    parser.add_argument("--model_name", type=str, default="t5-base")
    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--validation_split", type=float, default=0.1)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_pause", action="store_true", default=False)
    parser.add_argument("--use_duration", action="store_true", default=False)
    parser.add_argument("--use_zeros", action="store_true", default=False)
    parser.add_argument("--use_text", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    feats = []
    if args.use_zeros:
        feats.append("zero")
    elif args.use_pause:
        feats.append("pause")
    elif args.use_duration:
        feats.append("duration")
    if args.use_text:
        feats.append("text")
    if args.debug:
        feats.append("debug")
    args.outdir = f"/home/jm3743/prosody-syntax-interface/outputs/{'_'.join(feats)}"

    main(args)
