import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack
from torch.nn.utils.rnn import pad_sequence


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

        # ---------------------
        # 1) Project continuous prosody -> small latent (d_p)
        # ---------------------
        # simple 2-layer MLP with activation + optional layernorm
        self.prosody_proj_in = nn.Sequential(
            nn.Linear(self.prosody_feature_dim, self.prosody_latent_dim),
            nn.ReLU(),
            nn.LayerNorm(self.prosody_latent_dim),
            nn.Linear(self.prosody_latent_dim, self.prosody_latent_dim),
            nn.ReLU()
        )

        # ---------------------
        # 2) Learned positional embeddings for prosody latent sequence
        #    (we add these to the prosody projection so positions are distinguished)
        # ---------------------
        self.prosody_max_len = prosody_vocab_size
        self.prosody_pos_emb = nn.Embedding(self.prosody_max_len, self.prosody_latent_dim)

        # ---------------------
        # 3) Mini Transformer encoder for prosody (nn.TransformerEncoder)
        #    small stack so it is lightweight
        # ---------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.prosody_latent_dim,
            nhead=prosody_n_heads,
            dim_feedforward=max(4 * self.prosody_latent_dim, 64),
            dropout=float(getattr(config, "dropout_rate", 0.1)),
            batch_first=True,       # important: we use batch_first
            activation="relu"
        )
        self.prosody_transformer = nn.TransformerEncoder(encoder_layer, num_layers=prosody_n_layers)

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
        if prosody_feats is not None:
            # prosody_feats expected shape (B, T_p, F)
            # If prosody_mask is None, try to use attention_mask (after possible expansion)
            if prosody_mask is None:
                # If prosody sequence was expanded to match input_ids length, use attention_mask
                prosody_mask = attention_mask

            # 2.1 Project to small latent dim
            # flatten to (B*T_p, F) if needed is avoided since Sequential handles (B,T,F)->(B,T,d)
            x = self.prosody_proj_in(prosody_feats.unsqueeze(2))  # → (B, T_p, d_p)

            # 2.2 Add learned positional embeddings (clip positions if needed)
            batch_size, seq_len, _ = x.shape
            device = x.device
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)  # (B, T_p)
            pos_emb = self.prosody_pos_emb(pos_ids)  # (B, T_p, d_p)
            x = x + pos_emb

            # 2.3 Build src_key_padding_mask for TransformerEncoder: mask True where padding
            # HF convention: prosody_mask has 1 for valid tokens, 0 for padding
            if prosody_mask is not None:
                # Transformer expects key_padding_mask with True for positions to be masked
                src_key_padding_mask = (prosody_mask == 0)  # shape (B, T_p)
            else:
                src_key_padding_mask = None

            # 2.4 Run mini-transformer (batch_first=True)
            # nn.TransformerEncoder with batch_first expects (B, T, d)
            pros_out = self.prosody_transformer(x, src_key_padding_mask=src_key_padding_mask)  # (B, T_p, d_p)

            # 2.5 Upsample to d_model to match T5 hidden size
            prosody_states = self.prosody_up(pros_out)  # (B, T_p, d_model)

        # --- Cross-attention: word queries prosody ---
        if input_ids is not None and prosody_feats is not None:
            # key_padding_mask expects True where positions are *to be ignored* (padding positions).
            key_padding_mask = self._make_key_padding_mask(prosody_mask)  # (B, Tp)
            # MultiheadAttention (batch_first=True): query=(B, Tw, d), key=(B, Tp, d)
            # Ethan: text should bq Q, V; prosody should be K, since prosody is the key that controls access to information in text
            cross_out, _ = self.cross_attn(query=word_states, key=prosody_states, value=word_states, key_padding_mask=key_padding_mask)

            # Fusion: add cross-attn output to word states (residual) -> normalize + dropout
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
            # compute loss (shifted labels are typical for seq2seq)
            # align logits and labels: logits: (B, T, V), labels: (B, T) expected to be decoder-target tokens (not shifted)
            # If we used shift_right earlier, labels are raw target ids and logits are aligned to predict next tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

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

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prosody_feats": prosody,
            "prosody_mask": prosody_mask,   # shape: (B, T_subwords)
            "labels": labels,
        }