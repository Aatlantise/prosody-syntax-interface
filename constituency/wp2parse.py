import argparse
from pathlib import Path
from datasets import Dataset
from transformers import (
    T5Config, Seq2SeqTrainer, Seq2SeqTrainingArguments,
    PreTrainedTokenizerBase, EarlyStoppingCallback, T5ForConditionalGeneration
)
from constituency.util import TokenizerBuilder, load_jsonl_data, preprocess
from constituency.model import DualEncoderT5, DualEncoderCollator


def get_tokenizer():
    builder = TokenizerBuilder("t5-base")
    tokenizer = builder.build_tokenizer()
    return tokenizer


def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    items = load_jsonl_data(args.data, debug=args.debug)
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

    # Load base model + config first
    base = T5ForConditionalGeneration.from_pretrained(args.model_name)

    # Resize embeddings to new tokenizer
    base.resize_token_embeddings(len(tokenizer))

    # Update config so new model matches
    config = base.config
    config.vocab_size = len(tokenizer)

    # Now build your custom model
    model = DualEncoderT5(config)

    # Load pretrained weights
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.word_encoder.embed_tokens = model.shared

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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    trainer.model.floating_point_ops = lambda _: 0 # allow input_ids = None

    print("Training...")
    trainer.train()
    trainer.save_model(str(outdir / "model_final"))

    print("Evaluating...")
    eval_res = trainer.evaluate()
    print("Eval loss (nats/token):", eval_res["eval_loss"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/home/jm3743/prosody-syntax-interface/data/constituency_corpus.json")
    parser.add_argument("--model_name", type=str, default="t5-base")
    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=10)
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
