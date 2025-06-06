import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model
from data import load_data, extract_examples

class GPT2WithDurationClassifier(nn.Module):
    def __init__(self, gpt2_name='gpt2', duration_emb_dim=16):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gpt2 = GPT2Model.from_pretrained(gpt2_name, pad_token_id=self.tokenizer.pad_token_id)
        self.gpt2_hidden_dim = self.gpt2.config.hidden_size

        self.duration_proj = nn.Sequential(
            nn.Linear(1, duration_emb_dim),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.gpt2_hidden_dim + duration_emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Binary classification
        )

    def forward(self, texts, durations):
        # texts: list of strings (context up to current token)
        # durations: tensor of shape (batch_size, 1)
        encoded = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoded['input_ids'].to(self.gpt2.device)
        attention_mask = encoded['attention_mask'].to(self.gpt2.device)

        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)

        # Find the final non-padding token index for each sequence
        last_indices = attention_mask.sum(dim=1) - 1  # (batch,)
        final_reps = last_hidden[torch.arange(last_hidden.size(0)), last_indices]  # (batch, hidden)

        duration_emb = self.duration_proj(durations)  # (batch, dur_dim)

        joint = torch.cat([final_reps, duration_emb], dim=1)  # (batch, hidden + dur_dim)
        logits = self.classifier(joint).squeeze(1)  # (batch,)

        return logits

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2WithDurationClassifier().to(device)

    # Example input
    sentence = ["art", "with", "the", "recklessness"]
    durations = [0.35, 0.25, 0.22, 0.75]

    batch_texts = [" ".join(sentence[:i + 1]) for i in range(len(sentence))]
    batch_durations = torch.tensor(durations).unsqueeze(1).to(device)  # shape: (len, 1)

    logits = model(batch_texts, batch_durations)
    probs = torch.sigmoid(logits)  # Convert to probabilities
    print(probs)