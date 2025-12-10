from transformers import AutoTokenizer, AutoModel, AutoConfig
from model import DualEncoderT5, DualEncoderCollator
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from constituency.util import load_jsonl_data, preprocess
from datasets import Dataset
from transformers import (
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
)

def load_model(checkpoint_path, model_class=DualEncoderT5, device="cuda"):
    """
    Loads the model and tokenizer from a checkpoint.
    model_class should be your custom class, e.g. ProsodyT5ForConditionalGeneration.
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = model_class.from_pretrained(checkpoint_path, ignore_mismatched_sizes=False)
    model = model.to(device)
    model.tokenizer = tokenizer
    model.eval()
    return tokenizer, model


def infer_example(
        model,
        tokenizer,
        text=None,
        prosody_feats=None,  # FloatTensor[1, T_p, F] or numpy array
        prosody_mask=None,  # BoolTensor[1, T_p]
        parse=None,
        max_length=128,
        device="cuda"
):
    """
    Runs a single inference pass and returns:
    - output_text: decoded sequence
    - token_logprobs: log p(token | history)
    - token_entropies: per-token entropies
    - mean_entropy: average entropy over generated tokens

    For simplicity, this function always runs generate() then runs a
    forced-decoding pass to compute token-wise cross-entropy.
    """

    # -------------------------
    # 1. Tokenize input text
    # -------------------------
    if text is not None:
        enc = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
    else:
        input_ids = torch.tensor([[tokenizer.pad_token_id]], device=device)
        attention_mask = torch.tensor([[1]], device=device)

    labels = None
    if parse is not None:
        parse_enc = tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True
        ).to(device)
        labels = parse_enc["input_ids"]
    else:
        raise ValueError("Please pass parse to calculate entropy.")

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    # -------------------------
    # 2. Move prosody to device
    # -------------------------
    if prosody_feats is not None:
        if isinstance(prosody_feats, np.ndarray):
            prosody_feats = torch.tensor(prosody_feats, dtype=torch.float32)
        prosody_feats = prosody_feats.to(device)

    if prosody_mask is not None:
        prosody_mask = prosody_mask.to(device)

    if not hasattr(model.word_encoder, "embed_tokens") or model.word_encoder.embed_tokens is None:
        model.word_encoder.embed_tokens = model.shared

    with torch.no_grad():
        out = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prosody_feats=prosody_feats,
            prosody_mask=prosody_mask,
            decoder_input_ids=None,
            labels=labels,
            return_dict=True
        )
        logits = out.logits  # shape (1, T, V)

    # Get output text
    token_ids = torch.argmax(logits, dim=-1)
    output_text = tokenizer.batch_decode(token_ids, skip_special_tokens=True)

    # Shift labels so that token t is predicted from token t-1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Mask out positions where label == -100
    active_positions = shift_labels != -100
    vocab = shift_logits.size(-1)

    # Compute log-probs
    logprobs = F.log_softmax(shift_logits, dim=-1)

    # Gather log p( correct_token )
    token_logprobs = logprobs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Mask
    token_logprobs = token_logprobs[active_positions]

    # per-token cross entropy in nats
    token_entropies = -token_logprobs
    mean_entropy = token_entropies.mean().item()

    return {
        "output_text": output_text,
        "token_entropies": token_entropies.cpu(),
        "mean_entropy": mean_entropy,
        "token_logprobs": token_logprobs.cpu()
    }

def run_inference_example(model_path, model_class, parse, text=None, prosody=None):
    tokenizer, model = load_model(model_path, model_class)

    # prosody: e.g. numpy array [T, 1]
    if prosody is not None:
        prosody_feats = torch.tensor(prosody).unsqueeze(0)  # (1, T, F)
        prosody_mask = torch.ones(prosody_feats.shape[:2], dtype=torch.bool)
    else:
        prosody_feats = None
        prosody_mask = None

    result = infer_example(
        model=model,
        tokenizer=tokenizer,
        text=text,
        prosody_feats=prosody_feats,
        prosody_mask=prosody_mask,
        parse=parse,
    )

    print("Input text:", text)
    print("Generated parse:", result["output_text"])
    print("Mean entropy (nats/token):", result["mean_entropy"])
    print("Token entropies:", result["token_entropies"])
    return result

def eval_model():
    items = load_jsonl_data(debug=True)
    print(f"Loaded {len(items)} examples.")

    ds = Dataset.from_list(items)
    ds = ds.train_test_split(test_size=1, seed=42)
    eval_ds = ds["test"]

    checkpoint_path = "/home/jm3743/prosody-syntax-interface/outputs/text/model/checkpoint-2460"
    tokenizer, model = load_model(checkpoint_path)
    model.word_encoder.embed_tokens = model.shared

    print("Preprocessing...")
    preprocess_fn = lambda ex: preprocess(tokenizer, ex, 256, 256)
    tokenized_eval = eval_ds.map(preprocess_fn, batched=True, remove_columns=eval_ds.column_names)

    collator = DualEncoderCollator(tokenizer,
                                   device="cuda",
                                   return_text=True,
                                   return_pause=False,
                                   return_duration=False,
                                   return_zeros=False)

    training_args = Seq2SeqTrainingArguments(
        per_device_eval_batch_size=16,
        save_total_limit=3,
        weight_decay=0.01,
        remove_unused_columns=False,
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
        eval_dataset=tokenized_eval,
    )
    trainer.model.floating_point_ops = lambda _: 0 # allow input_ids = None

    # print("Training...")
    # trainer.train()
    # trainer.save_model(str(outdir / "model_final"))

    print("Evaluating...")
    eval_res = trainer.evaluate()
    print("Eval loss (nats/token):", eval_res["eval_loss"])


def main():
    result = run_inference_example(
        model_path="/home/jm3743/prosody-syntax-interface/outputs/text/model/checkpoint-2460",
        model_class=DualEncoderT5,
        text="She spoils the look of the room.",
        parse="(ROOT (S (NP (PRP She)) (VP (VBZ spoils) (NP (NP (DT the) (NN look)) (PP (IN of) (NP (DT the) (NN room))))) (. .)))",
        # prosody=None,
        prosody=[0.00, 0.03, 0.07, 0.40, 0.02, 0.05]  # example pause duration
    )

if __name__ == "__main__":
    # eval_model()
    main()