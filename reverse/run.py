import os
import pickle
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.functional import binary_cross_entropy_with_logits

from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from data import PhraseBoundaryDataset, collate_fn, get_libritts_data, load_data, extract_examples
from model import GPT2WithDurationClassifier


def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for texts, durations, labels in tqdm(dataloader):
        durations, labels = durations.to(device), labels.to(device)

        logits = model(texts, durations)
        loss = binary_cross_entropy_with_logits(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for texts, durations, labels in dataloader:
            durations = durations.to(device)
            labels = labels.to(device)

            logits = model(texts, durations)
            loss = binary_cross_entropy_with_logits(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().cpu()
            labels = labels.long()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())

    avg_loss = total_loss / total_samples
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="micro")

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():
    train_examples, val_examples, test_examples = get_libritts_data()

    # Assume examples is a list of {"text", "duration", "label"}
    train_dataset = PhraseBoundaryDataset(train_examples)
    val_dataset = PhraseBoundaryDataset(val_examples)
    test_dataset = PhraseBoundaryDataset(test_examples)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2WithDurationClassifier().to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(20):
        train_loss = train(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")

    test_metrics = evaluate(model, test_loader, device)
    print(f"  Test Loss: {test_metrics['loss']:.4f} | Val Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1']:.4f}")

def test():
    filepath = "sample.tsv"
    df = load_data(filepath)
    examples = extract_examples(df)

    dataset = PhraseBoundaryDataset(examples)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2WithDurationClassifier().to(device)

    test_metrics = evaluate(model, loader, device)
    print(
        f"  Initial Loss: {test_metrics['loss']:.4f} | Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1']:.4f}")

if __name__ == "__main__":
    main()
