import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model
from data import load_data, extract_examples_from_sent


class GPT2Classifier(nn.Module):
    def __init__(self, gpt2_name='gpt2', dropout_rate=0.1):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gpt2 = GPT2Model.from_pretrained(gpt2_name, pad_token_id=self.tokenizer.pad_token_id)
        self.gpt2_hidden_dim = self.gpt2.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.gpt2_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def _get_hidden(self, texts):
        # texts: list of strings (context up to current token)

        encoded = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoded['input_ids'].to(self.gpt2.device)
        attention_mask = encoded['attention_mask'].to(self.gpt2.device)

        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)

        return attention_mask, last_hidden

    def forward(self, texts, durations=None):
        attention_mask, last_hidden = self._get_hidden(texts)
        # Find the final non-padding token index for each sequence
        last_indices = attention_mask.sum(dim=1) - 1  # (batch,)
        final_reps = last_hidden[torch.arange(last_hidden.size(0)), last_indices]  # (batch, hidden)
        dropped_reps = self.dropout(final_reps)
        logits = self.classifier(dropped_reps).squeeze(1)  # (batch,)

        return logits

class GPT2WithDurationClassifier(GPT2Classifier):
    def __init__(self, gpt2_name='gpt2', duration_emb_dim=16, dropout_rate=0.1):
        super().__init__(gpt2_name=gpt2_name, dropout_rate=dropout_rate)

        self.duration_proj = nn.Sequential(
            nn.Linear(1, duration_emb_dim),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.gpt2_hidden_dim + duration_emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # 3 classes
        )

    def forward(self, texts, durations=None):
        attention_mask, last_hidden = self._get_hidden(texts)

        # Find the final non-padding token index for each sequence
        last_indices = attention_mask.sum(dim=1) - 1  # (batch,)
        final_reps = last_hidden[torch.arange(last_hidden.size(0)), last_indices]  # (batch, hidden)

        dropped_reps = self.dropout(final_reps)
        duration_emb = self.duration_proj(dropped_reps)  # (batch, dur_dim)

        joint = torch.cat([final_reps, duration_emb], dim=1)  # (batch, hidden + dur_dim)
        logits = self.classifier(joint).squeeze(1)  # (batch,)

        return logits

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2WithDurationClassifier().to(device)

    # Example input
    sentence = ["art", "with", "the", "recklessness"]
    d = [2.27, 1.5, 1.33, 0.35]

    batch_texts = [" ".join(sentence[:i + 1]) for i in range(len(sentence))]
    batch_durations = torch.tensor(d).unsqueeze(1).to(device)  # shape: (len, 1)

    logits = model(batch_texts, batch_durations)
    preds = torch.argmax(logits, dim=1).cpu()
    print(preds)
