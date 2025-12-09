from transformers import T5Config, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack
from torch.nn.utils.rnn import pad_sequence
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class ProsodyEncoder(nn.Module):
    """
    Prosody encoder that implements:
      - centisecond binning of scalar prosody values (0..255; 255 = >=2.55s)
      - sinusoidal lookup for positions (base_pos) and for magnitude bins (base_mag)
      - concatenation of pos_encoding||mag_encoding -> vector dim = prosody_dim (D)
      - small TransformerEncoder stack over these vectors (no projection before stack)
      - optional up-projection to target_dim for cross-attention fusion

    Inputs:
      prosody_vals: FloatTensor[B, T]  (raw continuous measurement per token; e.g., pause length seconds)
      prosody_mask: Bool/LongTensor[B, T]  (1 for valid positions, 0 for padding) or None -> all valid

    Outputs:
      prosody_states: FloatTensor[B, T, out_dim]  (out_dim == prosody_dim if up_proj=False else target_dim)
      prosody_mask: same mask as input (Bool Tensor)
    """
    def __init__(
        self,
        prosody_dim: int = 16,          # D total (must be >=2)
        max_len: int = 256,             # maximum sequence length expected (positions)
        base_pos: float = 10000.0,      # frequency base for positional sinusoids
        base_mag: float = 20000.0,      # frequency base for magnitude sinusoids (different from pos)
        prosody_bins: int = 256,        # number of centisecond bins (0..255)
        transformer_heads: int = 2,
        transformer_layers: int = 2,
        transformer_ff: Optional[int] = None,
        dropout: float = 0.1,
        target_dim: Optional[int] = None,   # if provided, up-project to this dim for fusion
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        assert prosody_dim >= 2, "prosody_dim must be >= 2"
        # we'll split prosody_dim into pos_dim and mag_dim
        pos_dim = prosody_dim // 2
        mag_dim = prosody_dim - pos_dim

        self.prosody_dim = prosody_dim
        self.pos_dim = pos_dim
        self.mag_dim = mag_dim
        self.max_len = max_len
        self.prosody_bins = prosody_bins
        self.base_pos = base_pos
        self.base_mag = base_mag
        self.target_dim = target_dim

        self.device = device
        self.dtype = dtype

        # Precompute sinusoidal tables (positions and magnitude bins)
        # shape (1, max_len, pos_dim) and (1, prosody_bins, mag_dim)
        pos_table = self._build_sinusoidal_table(max_len, pos_dim, base_pos)
        mag_table = self._build_sinusoidal_table(prosody_bins, mag_dim, base_mag)

        # register as buffers so they move with .to(device)
        self.register_buffer("pos_table", pos_table)   # (1, max_len, pos_dim)
        self.register_buffer("mag_table", mag_table)   # (1, prosody_bins, mag_dim)

        # Small Transformer encoder on top of the concatenated vectors
        d_model = prosody_dim
        if transformer_ff is None:
            transformer_ff = max(4 * d_model, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff,
            dropout=dropout,
            batch_first=True,
            activation="relu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Optional up-projection to match e.g. T5 d_model for cross-attention
        if target_dim is not None:
            self.up_proj = nn.Linear(d_model, target_dim)
        else:
            self.up_proj = None

        # small dropout after up-projection if used
        self.out_dropout = nn.Dropout(dropout)

    @staticmethod
    def _build_sinusoidal_table(length: int, dim: int, base: float) -> torch.Tensor:
        """
        Build (1, length, dim) sinusoidal table using base (like 10000).
        Uses standard formulation but with base argument to separate pos/mag frequencies.
        """
        pe = torch.zeros(length, dim)
        position = torch.arange(0, length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(base) / dim))
        if dim % 2 == 1:
            # handle odd dims by computing for floor(dim/2) pairs and leaving last column zero
            even_dim = dim - 1
            div_term = torch.exp(torch.arange(0, even_dim, 2).float() * (-math.log(base) / dim))
            pe[:, 0:even_dim:2] = torch.sin(position * div_term)
            pe[:, 1:even_dim:2] = torch.cos(position * div_term)
            # last column remains zero (or could copy last cos)
        else:
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, length, dim)

    @staticmethod
    def bin_centiseconds(values_seconds: torch.Tensor) -> torch.LongTensor:
        """
        Bin durations/pause values into centisecond bins 0..255.
        values_seconds: Tensor[B, T] float seconds
        Returns LongTensor[B, T] with integers in 0..255
        """
        # convert seconds -> centiseconds integer
        # floor to nearest centisecond
        cs = (values_seconds * 100.0).floor().long()  # centiseconds
        cs = torch.clamp(cs, min=0, max=255)
        return cs

    def forward(
        self,
        prosody_vals: torch.Tensor,        # shape (B, T) float seconds (can be already binned if you prefer)
        prosody_mask: Optional[torch.Tensor] = None,  # shape (B, T) 1/0 (1=valid)
        already_binned: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          prosody_states: (B, T, out_dim) where out_dim == prosody_dim or target_dim if up_proj set
          prosody_mask: BoolTensor (B, T) where True=valid
        """
        # Device and dtype forwarding
        device = prosody_vals.device
        dtype = prosody_vals.dtype

        # 1) get bin indices
        if already_binned:
            bin_idx = prosody_vals.long().clamp(0, self.prosody_bins - 1)  # ensure long
        else:
            bin_idx = self.bin_centiseconds(prosody_vals)  # (B, T) long in 0..255

        B, T = bin_idx.shape

        # 2) build magnitude and positional encodings from lookup tables
        # mag: use bin index to lookup the mag_table
        # mag_table: (1, prosody_bins, mag_dim)
        mag_table = self.mag_table.to(device=device, dtype=dtype)
        # gather: expand bin_idx to (B, T, mag_dim) via indexing
        mag_enc = mag_table[0].index_select(0, bin_idx.view(-1)).view(B, T, self.mag_dim)  # (B,T,mag_dim)

        # pos: take first T entries from pos_table
        pos_table = self.pos_table.to(device=device, dtype=dtype)
        if T > self.max_len:
            raise ValueError(f"Sequence length {T} > max_len {self.max_len}; increase max_len.")
        pos_enc = pos_table[:, :T, :].expand(B, -1, -1)  # (B, T, pos_dim)

        # 3) concatenate pos||mag -> (B, T, prosody_dim)
        x = torch.cat([pos_enc, mag_enc], dim=-1)

        # 4) build src_key_padding_mask expected by TransformerEncoder
        # TransformerEncoder expects src_key_padding_mask bool with True for positions to be masked (i.e., padding)
        if prosody_mask is None:
            src_key_padding_mask = None
            prosody_mask_bool = torch.ones(B, T, dtype=torch.bool, device=device)
        else:
            # user may pass 1/0 longs or bools; we want bool where True = valid
            prosody_mask_bool = (prosody_mask != 0).to(torch.bool).to(device)
            # invert for src_key_padding_mask: True means position is masked
            src_key_padding_mask = ~prosody_mask_bool  # True=pad, False=keep

        # If a batch row has all positions masked, provide a dummy time-step to avoid panic in Transformer
        if src_key_padding_mask is not None:
            all_masked = src_key_padding_mask.all(dim=1)  # (B,)
            if all_masked.any():
                # for those examples, set first position as valid with an arbitrary encoding (pos0 + mag0)
                idxs = torch.nonzero(all_masked, as_tuple=False).squeeze(1)
                x[idxs, 0, :] = 0.0  # safe zero vector (pos0/mag0 combination)
                src_key_padding_mask[idxs, 0] = False
                prosody_mask_bool[idxs, 0] = True

        # 5) run small transformer
        # TransformerEncoder expects src_key_padding_mask with True=to-ignore
        pros_out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (B, T, prosody_dim)

        # 6) optional up-projection
        if self.up_proj is not None:
            pros_out = self.out_dropout(self.up_proj(pros_out))  # map to target_dim

        # return pros_out and boolean mask of valid steps
        return pros_out, prosody_mask_bool



class DualEncoderT5(T5ForConditionalGeneration):
    """
    Dual-encoder T5:
      - word encoder: pretrained or from-config (uses `shared` embedding for words)
      - prosody encoder: small-vocab T5Stack with its own Embedding
      - cross-attention fusion: word queries prosody (nn.MultiheadAttention)
      - decoder: either random (from-config) or pretrained

    Forward signature compatible with HF Trainer (returns dict with loss & logits if labels provided).
    """

    def __init__(
        self,
        config: T5Config,
        prosody_vocab_size: int = 256,
        use_pretrained_encoder: bool = True,
        pretrained_name: str = "t5-base",
        decoder_from_scratch: bool = True,
    ):
        super().__init__(config)

        self.config = config


        # word embedding and encoder
        if use_pretrained_encoder:
            pre = T5ForConditionalGeneration.from_pretrained(pretrained_name)
            # shared embedding (word vocabulary, tied with decoder in default T5)
            self.shared = pre.shared
            # pretrained encoder (T5Stack)
            self.word_encoder = pre.encoder
            # optionally reuse decoder later if requested
            pretrained_decoder = pre.decoder
            pretrained_lm_head = pre.lm_head
        else:
            # create new shared embedding (random init)
            self.shared = nn.Embedding(config.vocab_size, config.d_model)
            self.word_encoder = T5Stack(config, embed_tokens=self.shared)
            pretrained_decoder = None
            pretrained_lm_head = None

        # store dims
        self.prosody_feature_dim = 1   # e.g., 1 (pause) or 2 (pause+dur)
        self.prosody_latent_dim = 8     # e.g., 8 or 16
        prosody_n_heads = 2
        prosody_n_layers = 2

        self.prosody_encoder = ProsodyEncoder(
            prosody_dim=16,  # whatever D you decide (8, 16, etc.)
            max_len=prosody_vocab_size,  # same as before
            base_pos=10000.0,
            base_mag=20000.0,
            prosody_bins=256,  # centisecond bins 0–255
            transformer_heads=prosody_n_heads,
            transformer_layers=prosody_n_layers,
            target_dim=config.d_model,  # IMPORTANT: match T5 dimension for cross-attention
            dropout=float(getattr(config, "dropout_rate", 0.1)),
        )


        # ---------------------
        # 4) Up-projection from prosody latent -> T5 d_model for cross-attention
        # ---------------------
        self.prosody_up = nn.Linear(self.prosody_latent_dim, config.d_model)

        # --- CROSS-ATTENTION (word queries prosody) ---
        # Use PyTorch MultiheadAttention for reliability and efficiency (batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.num_heads,
            batch_first=True
        )

        # small layernorm + dropout around fusion (optional)
        self.fusion_layer_norm = nn.LayerNorm(config.d_model)
        self.fusion_dropout = nn.Dropout(config.dropout_rate)

        # --- DECODER: either random (from-config) or reuse pretrained decoder ---
        if decoder_from_scratch:
            # create a decoder embedding separate from `shared` so decoder is independent
            # If you prefer tied embeddings between encoder and decoder, pass shared to T5Stack
            self.decoder_embed = nn.Embedding(config.vocab_size, config.d_model)
            self.decoder = T5Stack(config, embed_tokens=self.decoder_embed)
            # LM head tied to decoder embedding (weight tying)
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            # tie weights
            self.lm_head.weight = self.decoder_embed.weight
            # initialize decoder params with T5 init style if needed (left to HF internals in T5Stack)
        else:
            # reuse pretrained decoder and lm_head if pretrained encoder was loaded
            if use_pretrained_encoder:
                self.decoder = pretrained_decoder
                self.lm_head = pretrained_lm_head
            else:
                # no pretrained available → build random decoder but tie to shared embedding if desired
                self.decoder = T5Stack(config, embed_tokens=self.shared)
                self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
                self.lm_head.weight = self.shared.weight


    # ---------------------------
    # Utility: compute key padding mask for nn.MultiheadAttention
    # nn.MultiheadAttention with batch_first=True expects key_padding_mask shape (B, S)
    # with True in positions that should be masked (i.e., padding positions).
    # ---------------------------
    @staticmethod
    def _make_key_padding_mask(attn_mask):
        # attention_mask is 1 for valid tokens, 0 for padding (HF convention)
        # key_padding_mask expects True where positions are to be ignored
        if attn_mask is None:
            return None
        return ~(attn_mask.bool())  # True for pads

    def forward(
        self,
        input_ids=None,            # word token ids (B, Tw)
        attention_mask=None,       # mask for words (B, Tw)
        prosody_feats=None,        # new: (B, Tp, F) float tensor, F = prosody_feature_dim
        prosody_mask=None,         # mask for prosody (B, Tp) or same as attention_mask
        decoder_input_ids=None,    # optional decoder input ids for teacher forcing
        labels=None,               # optional labels (B, T_y) with -100 for pad
        return_dict: bool = True,
        **kwargs
    ):
        """
        - Assumes prosody_ids are aligned to input_ids shape (same sequence length).
        - If prosody_mask is None, uses attention_mask.
        - decoder_input_ids: if None and labels provided, we will shift labels inside if using lm_head
        """
        # --- Encode words ---
        # word_encoder accepts input_ids & attention_mask directly and returns BaseModelOutput
        word_states = None
        if input_ids is not None:
            word_enc_out = self.word_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, use_cache=False)
            # T5Stack returns .last_hidden_state or .last_hidden_state is accessible as .last_hidden_state
            # but to be robust, read .last_hidden_state or .last_hidden_state alias:
            try:
                word_states = word_enc_out.last_hidden_state
            except AttributeError:
                # fallback for different HF versions
                word_states = word_enc_out[0]

        # --- Encode prosody ---
        # prosody_vals: FloatTensor[B, Tp]
        # prosody_mask: BoolTensor[B, Tp]
        prosody_states, prosody_mask_bool = None, None
        if prosody_feats is not None:
            # Now prosody_encoder handles:
            #   - centisecond binning
            #   - sinusoidal pos + mag encoding
            #   - mini-transformer
            #   - up-projection to config.d_model
            prosody_states, prosody_mask_bool = self.prosody_encoder(
                prosody_feats,
                prosody_mask=prosody_mask,
                already_binned=False  # or True if your collator already bins to 0–255
            )


        # --- Cross-attention: word queries prosody ---
        if input_ids is not None and prosody_feats is not None:

            # Compute padding mask once
            # prosody_mask_bool: True = real, False = pad
            # key_padding_mask: True = ignore
            key_padding_mask = ~prosody_mask_bool  # (B, Tp)

            # Q = words, K,V = prosody
            cross_out, _ = self.cross_attn(
                query=word_states,  # (B, Tw, d)
                key=prosody_states,  # (B, Tp, d)
                value=prosody_states,  # (B, Tp, d)
                key_padding_mask=key_padding_mask,
            )

            fused = word_states + self.fusion_dropout(cross_out)
            fused = self.fusion_layer_norm(fused)

        elif input_ids is not None:
            fused = word_states

        elif prosody_feats is not None:
            fused = prosody_states

        # --- Decoder ---
        # Prepare decoder inputs. If decoder_input_ids not provided but labels exist, shift labels.
        if decoder_input_ids is None and labels is not None:
            # classic shift-right for T5: decoder_input_ids = shift_right(labels)
            # We can use HF T5 shift method if present; otherwise implement a simple shift_right
            decoder_input_ids = self._shift_right_t5(labels)

        # T5Stack decoder expects input_ids or inputs_embeds and encoder_hidden_states
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=fused,
            encoder_attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )

        # logits from LM head (on top of decoder last hidden state)
        dec_hidden = decoder_outputs.last_hidden_state
        logits = self.lm_head(dec_hidden)

        loss = None
        if labels is not None:
            logits_ = logits.view(-1, logits.size(-1))
            labels_ = labels.view(-1)

            mask = labels_ != -100
            masked_logits = logits_[mask]  # (N_valid, V)
            masked_labels = labels_[mask]  # (N_valid,)

            loss = nn.CrossEntropyLoss()(masked_logits, masked_labels)

        if return_dict:
            return {"loss": loss, "logits": logits, "encoder_last_hidden_state": fused}
        else:
            return loss, logits, fused


    def _shift_right_t5(self, labels):
        """
        Shift the labels to the right, and prepend decoder start token id.
        This is a simple implementation that uses config.decoder_start_token_id if present,
        otherwise uses eos_token_id as start.
        """
        if labels is None:
            return None
        decoder_start_token_id = getattr(self.config, "decoder_start_token_id", None)
        if decoder_start_token_id is None:
            decoder_start_token_id = getattr(self.config, "eos_token_id", 1)

        # labels: (B, T)
        shifted = labels.new_zeros(labels.size())
        shifted[:, 0] = decoder_start_token_id
        shifted[:, 1:] = labels[:, :-1]
        # replace -100 (ignore) with pad_token_id if present
        pad_token_id = getattr(self.config, "pad_token_id", 0)
        shifted = shifted.masked_fill(shifted == -100, pad_token_id)
        return shifted


class DualEncoderCollator:
    def __init__(self, tokenizer, device=None, return_text=False, return_pause=False, return_duration=False, return_zeros=False):
        self.tokenizer = tokenizer
        self.device = device
        self.return_text = return_text
        self.return_duration = return_duration
        self.return_pause = return_pause
        self.return_zeros = return_zeros

    def __call__(self, batch):

        # === TEXT SIDE ===
        input_ids = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
        attention_mask = [torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        duration_list = [torch.tensor(ex["duration"], dtype=torch.float) for ex in batch]
        pause_list = [torch.tensor(ex["pause"], dtype=torch.float) for ex in batch]


        # pad to batch dimension
        duration = pad_sequence(duration_list, batch_first=True, padding_value=0.0)
        pause = pad_sequence(pause_list, batch_first=True, padding_value=0.0)

        # mask is same for input_id, duration, and pause
        prosody_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        # === LABELS ===
        labels = [torch.tensor(ex["labels"], dtype=torch.long) for ex in batch]
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        if not self.return_text:
            input_ids = None
        else:
            pass

        assert not (self.return_duration and self.return_pause)
        if self.return_duration:
            prosody = duration
        elif self.return_pause:
            prosody = pause
        else:
            prosody = None

        if self.return_zeros:
            prosody = torch.zeros(attention_mask.shape, dtype=torch.float)
            prosody_mask = torch.ones(attention_mask.shape, dtype=torch.float)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prosody_feats": prosody,
            "prosody_mask": prosody_mask,   # shape: (B, T_subwords)
            "labels": labels,
        }
