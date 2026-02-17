""" 
getting surprisal values for words in convo-ados dataset
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import warnings
from transformers import AutoConfig, AutoTokenizer
from typing import Iterable
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model

@torch.no_grad()
def token_surprisals(text: str, tokenizer, model, device="cuda"):
    enc = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True
    )

    input_ids = enc["input_ids"].to(device)
    offsets = enc["offset_mapping"][0].tolist()

    outputs = model(input_ids)
    logits = outputs.logits[:, :-1, :]          # predict next token
    targets = input_ids[:, 1:]                  # shifted targets

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_loss = loss_fct(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )

    token_surprisal = token_loss.view(-1)       # (T-1,)

    return {
        "token_ids": input_ids[0, 1:].tolist(),
        "token_strs": tokenizer.convert_ids_to_tokens(input_ids[0, 1:]),
        "offsets": offsets[1:],                  # align with targets
        "surprisals": token_surprisal             # torch tensor
    }

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
        if start == end:  # special tokens
            continue

        token_text = text[start:end]

        if start == 0 or text[start - 1] == " ":
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


def compute(sentences: Iterable[str], model_name: str, b_size: int = 2) -> LMResult:
    tokenizer, model = load_lm(model_name)

    n_splits = len(sentences) // b_size
    words, surprisals = [], []
    for b_sentences in tqdm(np.array_split(sentences, n_splits)):

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv")
    parser.add_argument("--out_csv")
    parser.add_argument("--batch_size", type=int, default=8)
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

    args.in_csv = "/home/jm3743/data/candor_full_media/002d68da-7738-4177-89d9-d72ae803e0e4/transcription/transcript_audiophile.csv"
    args.out_csv = "test.csv"
    args.convo_id = "002d68da-7738-4177-89d9-d72ae803e0e4"

    # model setup
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model)

    # read data and prepare it
    df = read_candor_dataset(args.in_csv, args.turn_strategy)

    df["turn_id"] = list(range(len(df)))
    df["window_wc"] = df.text.astype(str).apply(lambda x: len(x.split()))

    max_tokens = AutoConfig.from_pretrained(args.lm_model).max_position_embeddings

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

    lm_result = compute(sentences, args.lm_model)
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


if __name__ == "__main__":
    main()
