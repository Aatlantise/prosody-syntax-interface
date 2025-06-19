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
from model import GPT2WithDurationClassifier, GPT2Classifier


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

    with torch.no_grad():
        for texts, durations, labels in tqdm(dataloader):
            durations = durations.to(device)
            labels = labels.to(device)

            logits = model(texts, durations)
            loss = binary_cross_entropy_with_logits(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().cpu()
            labels = labels.long()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="micro")

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main(args):
    train_examples, val_examples, test_examples = get_libritts_data()

    # Assume examples is a list of {"text", "duration", "label"}
    train_dataset = PhraseBoundaryDataset(train_examples)
    val_dataset = PhraseBoundaryDataset(val_examples)
    test_dataset = PhraseBoundaryDataset(test_examples)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_duration_info:
        model = GPT2WithDurationClassifier().to(device)
    else:
        model = GPT2Classifier().to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    min_loss = 99.99
    no_improvement_epoch = 0

    for epoch in range(10):
        train_loss = train(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")
        if min_loss > val_metrics["loss"]:
            min_loss = val_metrics["loss"]
        else:
            no_improvement_epoch += 1
            if no_improvement_epoch > 2:
                break

    test_metrics = evaluate(model, test_loader, device)
    print(f"  Test Loss: {test_metrics['loss']:.4f} | Val Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1']:.4f}")

def _test(args):
    filepath = "sample.tsv"
    df = load_data(filepath)
    examples = extract_examples(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_duration_info:
        model = GPT2WithDurationClassifier().to(device)
    else:
        model = GPT2Classifier().to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    l, j, = [int(len(examples) * 0.8), int(len(examples) * 0.9)]
    train_ex, val_ex, test_ex = examples[:l], examples[l:j], examples[j:]

    train_loader = DataLoader(PhraseBoundaryDataset(train_ex), batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(PhraseBoundaryDataset(val_ex), batch_size=8, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(PhraseBoundaryDataset(test_ex), batch_size=8, shuffle=False, collate_fn=collate_fn)

    train_loss = train(model, train_loader, optimizer, device)
    val_metrics = evaluate(model, val_loader, device)

    print(f"Epoch 0")
    print(f"  Train Loss: {train_loss:.4f}")
    print(
        f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")
    test_metrics = evaluate(model, test_loader, device)
    print(
        f"  Initial Loss: {test_metrics['loss']:.4f} | Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_duration_info", default=False, action="store_true")
    args = parser.parse_args()

    main(args)
    # _test(args)