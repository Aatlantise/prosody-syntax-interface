""" 
getting surprisal values for words in convo-ados dataset
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import warnings
from typing import Iterable
from dataclasses import dataclass
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from glob import glob

CANDOR_CLIFFHANGER_COL_MAPPING = {
    "speaker": "speaker",
    "start": "start",
    "stop": "stop",
    "utterance": "text",
}

CANDOR_BACKBITER_COL_MAPPING = {
    "speaker": "speaker",
    "start": "start",
    "stop": "stop",
    "utterance": "text",
}


@dataclass
class LMResult:
    words: Iterable[str]
    surprisals: Iterable[float]


def maybe_fix_model_name(model_name: str):
    if model_name == "gpt2":
        return "gpt2-small"
    else:
        return model_name

def read_candor_dataset(inputfile, turn_strategy):
    assert turn_strategy in ["audiophile", "cliffhanger", "backbiter"], print(turn_strategy)
    if turn_strategy == "cliffhanger":
        df = pd.read_csv(inputfile)
        df = df.rename(columns=CANDOR_CLIFFHANGER_COL_MAPPING)
    if turn_strategy == "backbiter":
        df = pd.read_csv(inputfile)
        df = df.rename(columns=CANDOR_BACKBITER_COL_MAPPING)
    if turn_strategy == "audiophile":
        df = pd.read_csv(inputfile)
        df = df.rename(columns=CANDOR_CLIFFHANGER_COL_MAPPING)
    return df


def load_lm(model_name: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT-2 doesn't have a pad token by default, which is required for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tell tokenizer to truncate from the left (removing oldest context first)
    tokenizer.truncation_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model

@torch.no_grad()
def _token_surprisals(text: str, tokenizer, model, device="cuda"):
    enc = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True
    )

    # add BOS token to input_ids
    input_ids = torch.cat([torch.tensor([[tokenizer.bos_token_id]]), enc["input_ids"]], dim=-1).to(device)

    # offset stays as is; nothing was added here
    offsets = enc["offset_mapping"][0].tolist()

    outputs = model(input_ids)
    logits = outputs.logits[:, :-1, :].contiguous()
    targets = input_ids[:, 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_loss = loss_fct(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )

    token_surprisal = token_loss.view(-1)       # (T-1,)

    return {
        "token_ids": input_ids[0, 1:].tolist(),
        "token_strs": tokenizer.convert_ids_to_tokens(input_ids[0, 1:]),
        "offsets": offsets,                  # align with targets
        "surprisals": token_surprisal             # torch tensor
    }


@torch.no_grad()
def batched_token_surprisals(texts: list[str], tokenizer, model, max_tokens: int, device="cuda"):
    """Processes a batch of sentences through the LM simultaneously."""

    # Tokenize the entire batch at once with padding and left-truncation
    enc = tokenizer(
        texts,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=max_tokens - 1  # Leave 1 space for the BOS token
    )

    batch_size = len(texts)

    # Create and prepend BOS tokens to input_ids
    bos_tokens = torch.full((batch_size, 1), tokenizer.bos_token_id, dtype=torch.long)
    input_ids = torch.cat([bos_tokens, enc["input_ids"]], dim=1).to(device)

    # Prepend 1s to attention mask for the new BOS tokens
    bos_mask = torch.ones((batch_size, 1), dtype=torch.long)
    attention_mask = torch.cat([bos_mask, enc["attention_mask"]], dim=1).to(device)

    # Run the model on the full batched tensor
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].contiguous()
    targets = input_ids[:, 1:].contiguous()

    # Calculate loss without reducing so we keep per-token surprisals
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_loss = loss_fct(
        logits.view(-1, logits.size(-1)),
        targets.view(-1)
    ).view(batch_size, -1)  # Reshape back to (batch_size, seq_len)

    # Reconstruct the per-sequence data, ignoring padding tokens
    batch_results = []
    for i in range(batch_size):
        # Calculate how many valid (non-padding) tokens this sequence has
        valid_len = enc["attention_mask"][i].sum().item()

        seq_input_ids = targets[i, :valid_len].tolist()
        seq_offsets = enc["offset_mapping"][i, :valid_len].tolist()
        seq_surprisals = token_loss[i, :valid_len].tolist()

        batch_results.append({
            "token_ids": seq_input_ids,
            "token_strs": tokenizer.convert_ids_to_tokens(seq_input_ids),
            "offsets": seq_offsets,
            "surprisals": seq_surprisals
        })

    return batch_results

def word_surprisals(text, token_data):
    words = []
    current_word = ""
    current_surprisal = 0.0
    results = []

    for tok, (start, end), s in zip(
        token_data["token_strs"],
        token_data["offsets"],
        token_data["surprisals"]
    ):

        token_text = text[start:end]

        if start == end:  # special tokens
            continue

        if start == 0 or tok[0] == "Ġ": # gpt-2 style
            if current_word:
                results.append((current_word, current_surprisal))
            current_word = token_text
            current_surprisal = float(s)
        else:
            current_word += token_text
            current_surprisal += float(s)

    if current_word:
        results.append((current_word, current_surprisal))

    return results

def compute_word_surprisals(text, tokenizer, model, device="cuda"):
    tok_data = token_surprisals(text, tokenizer, model, device)
    return word_surprisals(text, tok_data)


def _compute(sentences: Iterable[str], tokenizer, model, b_size: int = 2, ) -> LMResult:

    n_splits = len(sentences) // b_size
    words, surprisals = [], []
    for b_sentences in np.array_split(sentences, n_splits):

        text = "\n".join(b_sentences)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows = []
            for sent_id, text in enumerate(b_sentences):
                ws = compute_word_surprisals(text, tokenizer, model)
                for w_i, (word, surprisal) in enumerate(ws):
                    rows.append({
                        "text_id": sent_id,
                        "word_id": w_i,
                        "word": word,
                        "surprisal": surprisal
                    })

            df = pd.DataFrame(rows)
            words.extend(df.groupby("text_id")["word"].apply(list).tolist())
            surprisals.extend(df.groupby("text_id")["surprisal"].apply(list).tolist())

    return LMResult(words, surprisals)


def compute(sentences: list[str], tokenizer, model, max_tokens: int, b_size: int = 16, device="cuda") -> LMResult:
    words, surprisals = [], []

    # Process in safe, vectorized batches
    for i in range(0, len(sentences), b_size):
        b_sentences = sentences[i:i + b_size]

        # Get batched tensor results
        batch_tok_data = batched_token_surprisals(b_sentences, tokenizer, model, max_tokens, device)

        # Reconstruct string words for each sentence in the batch
        for text, tok_data in zip(b_sentences, batch_tok_data):
            ws = word_surprisals(text, tok_data)
            words.append([w for w, s in ws])
            surprisals.append([s for w, s in ws])

    return LMResult(words, surprisals)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv")
    parser.add_argument("--out_csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lm_model", type=str, default="gpt2")
    parser.add_argument("--turn_strategy", type=str, default="audiophile")
    parser.add_argument("--convo_id")
    parser.add_argument(
        "--context_len",
        type=int,
        help="number of previous turns to prepend to current turn as context. 0 -> use current turn alone.",
        default=1,
    )
    args = parser.parse_args()

    max_tokens = AutoConfig.from_pretrained(args.lm_model).max_position_embeddings
    tokenizer, model = load_lm(args.lm_model)

    paths = glob("/home/scratch/jm3743/candor_full_media/*/transcription/transcript_audiophile.csv")
    for in_csv in tqdm(paths):
        args.in_csv = in_csv
        args.convo_id = in_csv.split("/")[-3]
        args.out_csv = f"/home/scratch/jm3743/candor_full_media/{args.convo_id}/surprisal.csv"

        # read data and prepare it
        df = read_candor_dataset(args.in_csv, args.turn_strategy)

        df["turn_id"] = list(range(len(df)))
        df["window_wc"] = df.text.astype(str).apply(lambda x: len(x.split()))

        # prepend context
        df["context"] = ""
        for i in range(1, args.context_len + 1):
            df["context"] = (
                df["text"].shift(i).fillna("") + " " + df["context"]
            ).str.strip()
        df["text_w_context"] = (df["context"].fillna("") + " " + df["text"]).str.strip()
        df["context_wc"] = df.context.astype(str).apply(lambda x: len(x.split()))

        # truncate
        df["text_w_context_enc"] = df.text_w_context.apply(lambda x: tokenizer.encode(x))
        df["text_w_context_enc"] = df.text_w_context_enc.apply(
            lambda x: x[-(max_tokens - 2) :]
        )
        df["text_w_context"] = df.text_w_context_enc.apply(lambda x: tokenizer.decode(x))

        sentences = df.text_w_context.to_list()

        print(f"Computing surprisals for {len(sentences)} sentences.")

        lm_result = compute(sentences, tokenizer, model)
        df["word"] = lm_result.words
        df["surprisal"] = lm_result.surprisals
        assert (df.surprisal.apply(len) == df.word.apply(len)).all()

        # truncate to only contain the last window_wc items of each of the variables
        df["word"] = list(map(lambda x, y: x[-y:], df.word, df.window_wc))
        df["surprisal"] = list(map(lambda x, y: x[-y:], df.surprisal, df.window_wc))

        # explode dataframe so there's one row for each word
        df = df.explode(["word", "surprisal"])
        df = df.rename(columns={"start": "turn_start", "stop": "turn_stop"})

        df.to_csv(
            args.out_csv,
            index=False,
            columns=[
                "turn_id",
                "turn_start",
                "turn_stop",
                "speaker",
                "window_wc",
                "word",
                "surprisal",
                "context_wc",
            ],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lm_model", type=str, default="gpt2")
    parser.add_argument("--turn_strategy", type=str, default="audiophile")
    parser.add_argument(
        "--context_len",
        type=int,
        help="number of previous turns to prepend to current turn as context. 0 -> use current turn alone.",
        default=1,
    )
    args = parser.parse_args()

    max_tokens = AutoConfig.from_pretrained(args.lm_model).max_position_embeddings
    tokenizer, model = load_lm(args.lm_model)

    paths = glob("/home/scratch/jm3743/candor_full_media/*/transcription/transcript_audiophile.csv")

    for in_csv in tqdm(paths, desc="Processing Conversations"):
        convo_id = in_csv.split("/")[-3]
        out_csv = f"/home/scratch/jm3743/candor_full_media/{convo_id}/surprisal.csv"

        df = read_candor_dataset(in_csv, args.turn_strategy)
        df["turn_id"] = range(len(df))

        # Count words based on spaces to match the slicing logic later
        df["window_wc"] = df.text.astype(str).apply(lambda x: len(x.split()))

        # Prepend context efficiently
        df["context"] = ""
        for i in range(1, args.context_len + 1):
            df["context"] = (df["text"].shift(i).fillna("") + " " + df["context"]).str.strip()

        df["text_w_context"] = (df["context"].fillna("") + " " + df["text"]).str.strip()
        df["context_wc"] = df.context.astype(str).apply(lambda x: len(x.split()))

        sentences = df.text_w_context.tolist()

        # Compute surprisals using the fast batched tensor logic
        lm_result = compute(sentences, tokenizer, model, max_tokens, b_size=args.batch_size)

        df["word"] = lm_result.words
        df["surprisal"] = lm_result.surprisals

        # Slicing the lists to isolate only the words in the current turn
        # (Assuming the number of grouped tokens matches the string split logic)
        df["word"] = [w[-wc:] if wc > 0 else [] for w, wc in zip(df.word, df.window_wc)]
        df["surprisal"] = [s[-wc:] if wc > 0 else [] for s, wc in zip(df.surprisal, df.window_wc)]

        # Explode into word-level rows
        df = df.explode(["word", "surprisal"])
        df = df.rename(columns={"start": "turn_start", "stop": "turn_stop"})

        df.to_csv(
            out_csv,
            index=False,
            columns=[
                "turn_id",
                "turn_start",
                "turn_stop",
                "speaker",
                "window_wc",
                "word",
                "surprisal",
                "context_wc",
            ],
        )


if __name__ == "__main__":
    main()