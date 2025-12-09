from transformers import AutoTokenizer, AutoModel, AutoConfig
from model import DualEncoderT5
import torch
import torch.nn.functional as F
import numpy as np

def load_model(checkpoint_path, model_class=DualEncoderT5, device="cuda"):
    """
    Loads the model and tokenizer from a checkpoint.
    model_class should be your custom class, e.g. ProsodyT5ForConditionalGeneration.
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = model_class.from_pretrained(checkpoint_path, ignore_mismatched_sizes=False)
    model = model.to(device)
    model.eval()
    return tokenizer, model


def infer_example(
        model,
        tokenizer,
        text=None,
        prosody_feats=None,  # FloatTensor[1, T_p, F] or numpy array
        prosody_mask=None,  # BoolTensor[1, T_p]
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
            padding=False,
            truncation=True
        ).to(device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
    else:
        input_ids = torch.tensor([[tokenizer.pad_token_id]], device=device)
        attention_mask = torch.tensor([[1]], device=device)

    # -------------------------
    # 2. Move prosody to device
    # -------------------------
    if prosody_feats is not None:
        if isinstance(prosody_feats, np.ndarray):
            prosody_feats = torch.tensor(prosody_feats, dtype=torch.float32)
        prosody_feats = prosody_feats.to(device)

    if prosody_mask is not None:
        prosody_mask = prosody_mask.to(device)

    # -------------------------
    # 3. Generate output
    # -------------------------
    with torch.no_grad():
        gen_out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prosody_feats=prosody_feats,
            prosody_mask=prosody_mask,
            max_length=max_length,
            num_beams=1,
            use_cache=False
        )

    output_text = tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0]

    # -------------------------
    # 4. Compute per-token cross-entropy
    # -------------------------
    # Prepare decoder labels identical to gen_out (teacher forcing)
    labels = gen_out.clone()

    if not hasattr(model.word_encoder, "embed_tokens") or model.word_encoder.embed_tokens is None:
        model.word_encoder.embed_tokens = model.shared

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prosody_feats=prosody_feats,
            prosody_mask=prosody_mask,
            decoder_input_ids=labels,
            labels=labels,
            return_dict=True
        )
        logits = out["logits"]  # shape (1, T, V)

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

def run_inference_example(model_path, model_class, text, prosody=None):
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
        prosody_mask=prosody_mask
    )

    print("Input text:", text)
    print("Generated parse:", result["output_text"])
    print("Mean entropy (nats/token):", result["mean_entropy"])
    print("Token entropies:", result["token_entropies"])
    return result

if __name__ == "__main__":
    result = run_inference_example(
        model_path="/home/jm3743/prosody-syntax-interface/outputs/pause_debug/model_final",
        model_class=DualEncoderT5,
        # text="The dog walked to the park",
        text=None,
        prosody=[0.00, 0.03, 0.07, 0.40, 0.02, 0.05]  # example pause durations
    )
