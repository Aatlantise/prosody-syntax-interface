from transformers import AutoTokenizer, AutoModel, AutoConfig
from constituency.model import DualEncoderT5, DualEncoderCollator
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from constituency.util import load_jsonl_data, preprocess
from datasets import Dataset
from transformers import (
    Seq2SeqTrainer, Seq2SeqTrainingArguments, GPT2LMHeadModel
)
from constituency.wp2parse import get_tokenizer
from torch.utils.data import DataLoader
import re
import math
import argparse

def load_model(checkpoint_path, model_class=DualEncoderT5, device="cuda"):
    """
    Loads the model and tokenizer from a checkpoint.
    model_class should be your custom class, e.g. ProsodyT5ForConditionalGeneration.
    """
    tokenizer = get_tokenizer()
    model = model_class.from_pretrained(checkpoint_path, ignore_mismatched_sizes=False)
    model = model.to(device)
    model.tokenizer = tokenizer
    model.eval()
    return tokenizer, model


def infer_example(
        model,
        tokenizer,
        text=None,
        prosody_feats=None,
        prosody_mask=None,
        parse=None,
        max_length=128,
        device="cuda"
):
    # Place your list at the top of the function
    pos_tags = [
        '$', "''", ',', '-LRB-', '-RRB-', '.', ':', 'ADD', 'ADJP', 'ADVP', 'AFX', 'CC', 'CD', 'CONJP', 'DT', 'EX',
        'FRAG', 'FW', 'HYPH', 'GW', 'IN', 'INTJ', 'JJ', 'JJR', 'JJS', 'LS', 'LST', 'MD', 'NAC', 'NFP', 'NML', 'NN',
        'NNP', 'NNPS',
        'NNS', 'NP', 'PDT', 'POS', 'PP', 'PRN', 'PRP', 'PRP$', 'PRT', 'QP', 'RB', 'RBR', 'RBS', 'ROOT', 'RP', 'RRC',
        'S', 'SBAR', 'SBARQ', 'SINV', 'SQ', 'SYM', 'TO', 'UCP', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'VP',
        'WDT', 'WHADJP', 'WHADVP', 'WHNP', 'WHPP', 'WP', 'WP$', 'WRB', 'X', '``', '(', ')'
    ]

    vocab_to_id = tokenizer.get_vocab()
    vocab_size = len(vocab_to_id)

    # Create a boolean mask of what is considered a legal structural token
    # True = Structural Tag/Bracket, False = Literal Word (e.g., "dog", "cat")
    is_structural_token = torch.zeros(vocab_size, dtype=torch.bool, device=device)

    for tag in pos_tags:
        # Some tokenizers prepend spaces or special boundary symbols (like ' ' or 'Ġ')
        # We find all vocabulary IDs that contain or match your POS tags cleanly
        for vocab_str, token_id in vocab_to_id.items():
            clean_vocab_str = vocab_str.replace(' ', '').replace('Ġ', '').strip()
            if clean_vocab_str == tag:
                is_structural_token[token_id] = True

    # Create the inverse mask: True for words like "dog", "cat", etc.
    is_literal_word = ~is_structural_token

    # -------------------------
    # 1. Tokenize input text (Your original infrastructure)
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
        input_ids = None
        attention_mask = None

    if parse is not None:
        parse_enc = tokenizer(
            parse,
            return_tensors="pt",
            padding=False,
            truncation=True
        ).to(device)
        labels = parse_enc["input_ids"]
    else:
        raise ValueError("Please pass parse to calculate entropy.")

    # -------------------------
    # 2. Move prosody to device (Your original infrastructure)
    # -------------------------
    if prosody_feats is not None:
        if isinstance(prosody_feats, np.ndarray):
            prosody_feats = torch.tensor(prosody_feats, dtype=torch.float32)
        prosody_feats = prosody_feats.to(device)

    if prosody_mask is not None:
        prosody_mask = prosody_mask.to(device)

    # -------------------------
    # 3. Model Forward Execution
    # -------------------------
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

    # Get baseline generated targets
    token_ids = torch.argmax(logits, dim=-1)
    output_text = tokenizer.batch_decode(token_ids, skip_special_tokens=True)

    # Active positional masks filtering out paddings/special label indices
    active_positions = (labels != -100) & (labels != tokenizer.pad_token_id)

    # Clean probability distributions over target vocabulary
    probabilities = F.softmax(logits, dim=-1).squeeze(0)  # Shape: (T, V)
    flat_labels = labels.squeeze(0)  # Shape: (T,)
    flat_active = active_positions.squeeze(0)  # Shape: (T,)

    vocab_to_id = tokenizer.get_vocab()

    # Identify bracket token IDs in your vocabulary
    open_bracket_id = vocab_to_id.get('(', None)
    close_bracket_id = vocab_to_id.get(')', None)

    structural_stack = []
    total_leaked_mass = 0.0
    valid_token_steps = 0

    # Initialize state tracker to observe what token preceded the current prediction step
    prev_label_id = None

    # -------------------------
    # 4. Syntactic State Evaluation Loop
    # -------------------------
    for t in range(len(flat_labels)):
        if not flat_active[t]:
            continue

        current_label_id = flat_labels[t].item()
        next_step_probs = probabilities[t]

        # --- CONSTRAINT A: Unmatched closing brackets ---
        if len(structural_stack) == 0 and close_bracket_id is not None:
            total_leaked_mass += next_step_probs[close_bracket_id].item()
            valid_token_steps += 1

        # --- CONSTRAINT B: Literal words after open parentheses (NEW) ---
        if prev_label_id == open_bracket_id:
            # Sum up all probability mass mistakenly assigned to non-POS/non-structural words
            word_leakage_mass = next_step_probs[is_literal_word].sum().item()
            total_leaked_mass += word_leakage_mass
            valid_token_steps += 1

        # --- Update our state trackers using ground-truth history ---
        if current_label_id == open_bracket_id:
            structural_stack.append('(')
        elif current_label_id == close_bracket_id and len(structural_stack) > 0:
            structural_stack.pop()

        # Move the history slider forward
        prev_label_id = current_label_id

    # Compute final percentage metric
    average_leakage = (total_leaked_mass / valid_token_steps) * 100 if valid_token_steps > 0 else 0.0

    # Compute standard cross-entropy statistics (Your original logic)
    logprobs = F.log_softmax(logits, dim=-1)
    token_logprobs = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs[active_positions]
    token_entropies = -token_logprobs
    mean_entropy = token_entropies.mean().item()

    return {
        "output_text": output_text,
        "token_entropies": token_entropies.cpu(),
        "mean_entropy": mean_entropy,
        "token_logprobs": token_logprobs.cpu(),
        "leakage_pct": average_leakage  # <-- New metric returned here
    }

def run_inference_example(model_path, model_class, parse, text=None, prosody=None):
    tokenizer, model = load_model(model_path, model_class=model_class)

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
    print("Leakage pct:", result["leakage_pct"])
    return result

def eval_model(checkpoint_path, model_class):
    items = load_jsonl_data(debug=True)
    print(f"Loaded {len(items)} examples.")

    eval_ds = Dataset.from_list(items)

    tokenizer, model = load_model(checkpoint_path, model_class=model_class)

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
        per_device_train_batch_size=16,
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

    print("Evaluating...")
    eval_res = trainer.evaluate()
    print("Eval loss (nats/token):", eval_res["eval_loss"])


def run_corpus_leakage_analysis(mode, debug=False):
    """
    Runs an explicit batch-wise syntactic leakage analysis over the entire dataset.
    Returns a pandas DataFrame containing per-sentence structural features and leakage metrics.
    """
    prefix = "/home/jm3743/prosody-syntax-interface/"
    checkpoint_path = f"{prefix}outputs/{mode}/model_final"
    model_class = DualEncoderT5
    jsonl_path = f"{prefix}data/constituency_corpus_reldur.json" if 'libri' in mode\
        else f"{prefix}data/candor_corpus.json"

    args = argparse.Namespace()
    args.data = jsonl_path
    args.dyck = 'dyck' in mode
    args.nopunct = 'nopunct' in mode
    args.debug = False

    # 1. Load your raw dataset items to retain clean text and parse trees
    items = load_jsonl_data(args)
    print(f"Loaded {len(items)} sentences for syntax tracking.")

    max_sample_size = 1000
    if len(items) > max_sample_size:
        import random
        # Using a fixed seed guarantees every model evaluates the EXACT same sentences
        random.seed(42)
        items = random.sample(items, max_sample_size)

    # 2. Re-use your native setup pipeline
    tokenizer, model = load_model(checkpoint_path, model_class=model_class)
    device = "cuda"

    # Pre-compile the global structural/POS vocabulary masks
    pos_tags = [
        '$', "''", ',', '-LRB-', '-RRB-', '.', ':', 'ADD', 'ADJP', 'ADVP', 'AFX', 'CC', 'CD', 'CONJP', 'DT', 'EX',
        'FRAG', 'FW', 'HYPH', 'GW', 'IN', 'INTJ', 'JJ', 'JJR', 'JJS', 'LS', 'LST', 'MD', 'NAC', 'NFP', 'NML', 'NN',
        'NNP', 'NNPS',
        'NNS', 'NP', 'PDT', 'POS', 'PP', 'PRN', 'PRP', 'PRP$', 'PRT', 'QP', 'RB', 'RBR', 'RBS', 'ROOT', 'RP', 'RRC',
        'S', 'SBAR', 'SBARQ', 'SINV', 'SQ', 'SYM', 'TO', 'UCP', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'VP',
        'WDT', 'WHADJP', 'WHADVP', 'WHNP', 'WHPP', 'WP', 'WP$', 'WRB', 'X', '``', '(', ')'
    ]
    vocab_to_id = tokenizer.get_vocab()
    vocab_size = len(vocab_to_id)
    is_structural_token = torch.zeros(vocab_size, dtype=torch.bool, device=device)

    for tag in pos_tags:
        for vocab_str, token_id in vocab_to_id.items():
            clean_str = vocab_str.replace(' ', '').replace('Ġ', '').strip()
            if clean_str == tag:
                is_structural_token[token_id] = True
    is_literal_word = ~is_structural_token

    open_bracket_id = vocab_to_id.get('(', None)
    close_bracket_id = vocab_to_id.get(')', None)

    # 3. Use your custom data collator to handle feature configurations uniformly
    # Adjust flags to match your running condition (e.g., return_duration=True for duration models)
    is_pause_model = 'pause' in checkpoint_path
    is_duration_model = 'duration' in checkpoint_path

    collator = DualEncoderCollator(
        tokenizer,
        device=device,
        return_text=not (is_pause_model or is_duration_model),
        return_pause=is_pause_model,
        return_duration=is_duration_model,
        return_zeros=False
    )

    # Transform dataset using your preprocessing function
    eval_ds = Dataset.from_list(items)
    preprocess_fn = lambda ex: preprocess(tokenizer, ex, 256, 256)
    tokenized_eval = eval_ds.map(preprocess_fn, batched=True, remove_columns=eval_ds.column_names)

    # Build a standard PyTorch DataLoader using your collator
    eval_loader = DataLoader(tokenized_eval, batch_size=1, shuffle=False, collate_fn=collator)

    corpus_analysis_data = []
    print("Evaluating vocabulary distribution spaces step-by-step...")

    # 4. Process instance-by-instance to maintain exact feature mappings
    with torch.no_grad():
        for idx, batch in enumerate(eval_loader):

            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            raw_item = items[idx]

            # Extract independent geometric sentence properties
            clean_text = re.sub(r'[",?.!:]', '', raw_item['text'])
            word_count = len(clean_text.split())

            current_depth, parse_depth = 0, 0
            for char in raw_item['parse']:
                if char == '(':
                    current_depth += 1
                    if current_depth > parse_depth:
                        parse_depth = current_depth
                elif char == ')':
                    current_depth -= 1

            # Execute forward model pass to collect decoder logits
            outputs = model.forward(
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                prosody_feats=batch.get("prosody_feats"),
                prosody_mask=batch.get("prosody_mask"),
                decoder_input_ids=None,
                labels=batch.get("labels"),
                return_dict=True
            )

            logits = outputs.logits.squeeze(0)  # Shape: (T, V)
            labels = batch["labels"].squeeze(0)  # Shape: (T,)

            # Filter out ignored positions (-100 and pad tokens)
            active_positions = (labels != -100) & (labels != tokenizer.pad_token_id)
            probabilities = F.softmax(logits, dim=-1)  # Shape: (T, V)

            structural_stack = []
            total_leaked_mass = 0.0
            valid_token_steps = 0
            prev_label_id = None

            # Sentence-level sequential verification loop
            for t in range(len(labels)):
                if not active_positions[t]:
                    continue

                current_label_id = labels[t].item()
                next_step_probs = probabilities[t]

                # Check bracket constraints
                # if len(structural_stack) == 0 and close_bracket_id is not None:
                #     total_leaked_mass += next_step_probs[close_bracket_id].item()
                #     valid_token_steps += 1

                # Check lexical POS tags placement rules
                if prev_label_id == open_bracket_id:
                    total_leaked_mass += next_step_probs[is_literal_word].sum().item()
                    valid_token_steps += 1

                # Advance context state machine
                if current_label_id == open_bracket_id:
                    structural_stack.append('(')
                elif current_label_id == close_bracket_id and len(structural_stack) > 0:
                    structural_stack.pop()

                prev_label_id = current_label_id

            # Calculate precise sequence percentage score
            sentence_leakage = (total_leaked_mass / valid_token_steps) * 100 if valid_token_steps > 0 else 0.0

            corpus_analysis_data.append({
                'original_index': idx,
                'text': raw_item['text'],
                'word_count': word_count,
                'parse_depth': parse_depth,
                'syntax_leakage_pct': sentence_leakage
            })

    # 5. Pack everything into a clean DataFrame for downstream analytics
    leakage_scores = [row['syntax_leakage_pct'] for row in corpus_analysis_data]
    n_samples = len(leakage_scores)

    if n_samples > 0:
        # Calculate Mean
        mean_leakage = sum(leakage_scores) / n_samples

        # Calculate Standard Deviation (Sample Standard Deviation)
        variance = sum((x - mean_leakage) ** 2 for x in leakage_scores) / max(1, n_samples - 1)
        std_deviation = math.sqrt(variance)

        # Calculate Maximum
        max_leakage = max(leakage_scores)
    else:
        mean_leakage, std_deviation, max_leakage = 0.0, 0.0, 0.0

    # Print high-level diagnostic overview (100% independent of Pandas)
    print("\n=======================================================")
    print(" CORPUS-WIDE SYNTAX LEAKAGE SUMMARY")
    print("=======================================================")
    print(f"Evaluated Sample Size:         {n_samples} sentences")
    print(f"Mean In-Session Leakage Rate:  {mean_leakage:.4f}%")
    print(f"Standard Deviation of Leakage: {std_deviation:.4f}%")
    print(f"Maximum Sentence Leakage:      {max_leakage:.4f}%")
    print("=======================================================\n")

    # If you need to return the records to another function, just return the raw dictionary list
    return corpus_analysis_data

    return df_results


def main(mode):
    print(f"Inferring on {mode}...")
    model_path = f"/home/jm3743/prosody-syntax-interface/outputs/{mode}/model_final"
    model_class = DualEncoderT5
    if 'candor' in mode:
        if 'text' in mode:
            text = "She spoils the look of the room."
        else:
            text = None
        parse = "(ROOT (S (NP PRP) (VP VBZ (NP (NP DT NN) (PP IN (NP DT NN)))) .))"
        pause = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02]
        duration = [0.11, 0.6, 0.1, 0.22, 0.14, 0.1, 0.5]
    elif 'libri' in mode:
        if 'text' in mode:
            text = "So, how did you end up in Kentucky then?"
        else:
            text = None
        parse = "(ROOT (SBARQ RB (WHADVP WRB) (SQ VBD (NP PRP) (VP VB (PRT RP) (PP IN (NP NNP)) (ADVP RB))) .))"
        pause = [0.0100000000001045, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4900000000000091]
        duration = [0.3699999999998908, 0.2999999999999545, 0.1599999999999681, 0.1100000000000136, 0.1499999999999772,
                    0.1900000000000545, 0.2699999999999818, 0.14333333333331666, 0.32000000000005]
    else:
        raise ValueError

    if 'pause' in mode:
        prosody = pause
    elif 'duration' in mode:
        prosody = duration
    else:
        prosody = None
    result = run_inference_example(
        model_path=model_path,
        model_class=model_class,
        text=text,
        parse=parse,
        prosody=prosody
    )

if __name__ == "__main__":
    # run_corpus_leakage_analysis("libri_nopunct")
    # run_corpus_leakage_analysis("libri_duration_nopunct")
    # run_corpus_leakage_analysis("libri_pause_nopunct")
    # run_corpus_leakage_analysis("libri_text_nopunct")
    # run_corpus_leakage_analysis("libri_text_duration_nopunct")
    # run_corpus_leakage_analysis("libri_text_pause_nopunct")

    run_corpus_leakage_analysis("candor_nopunct")
    run_corpus_leakage_analysis("candor_duration_nopunct")
    run_corpus_leakage_analysis("candor_pause_nopunct")

