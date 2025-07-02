import os
import pickle
import argparse
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.functional import cross_entropy

from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from data import PhraseBoundaryDataset, collate_fn, get_libritts_data, load_data, extract_examples_from_sent
from model import GPT2WithProsodyClassifier, GPT2Classifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args, model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    batch_loss = 0.0

    batches = tqdm(dataloader)
    for texts, durations, pauses, labels in batches:
        if args.debug:
            batches.set_description(desc=f"Train loss: {batch_loss}")
        durations, pauses, labels = durations.to(device), pauses.to(device), labels.to(device).long()

        prosody = torch.tensor([]).to(device)
        if args.use_duration_info:
            prosody = torch.cat([prosody, durations], dim=1)
        if args.use_pause_info:
            prosody = torch.cat([prosody, pauses], dim=1)
        prosody = prosody if list(prosody.shape)[0] else None


        logits = model(texts, prosody)
        loss = cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += loss.item()
        assert batch_loss > 0, print(labels)

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for texts, durations, labels in tqdm(dataloader):
            durations = durations.to(device)
            labels = labels.to(device).long()

            logits = model(texts, durations)
            loss = cross_entropy(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu()
            labels = labels.long()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

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

    # Assume examples is a list of {"text", "duration", "pause", "label"}
    train_dataset = PhraseBoundaryDataset(train_examples)
    val_dataset = PhraseBoundaryDataset(val_examples)
    test_dataset = PhraseBoundaryDataset(test_examples)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_duration_info or args.use_pause_info:
        num_prosodic_feats = int(args.use_duration_info) + int(args.use_pause_info)
        model = GPT2WithProsodyClassifier(num_prosodic_feats, prosody_emb_dim=args.prosody_emb_size).to(device)
    else:
        model = GPT2Classifier().to(device)

    optimizer = AdamW(model.parameters(),
                      lr=5e-5,
                      betas=(0.9, 0.95),
                      weight_decay=1e-5
                      )

    min_loss = 99.99
    no_improvement_epoch = 0

    max_epoch = 30

    for epoch in range(max_epoch):
        train_loss = train(args, model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")

        test_metrics = evaluate(model, test_loader, device)
        print(
            f"  Test Loss: {test_metrics['loss']:.4f} | Test Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1']:.4f}")

        if min_loss > val_metrics["loss"]:
            min_loss = val_metrics["loss"]
        else:
            no_improvement_epoch += 1
            if no_improvement_epoch > 2:
                break


def _test(args):
    filepath = "sample.tsv"
    df = load_data(filepath)
    examples, _, _ = extract_examples_from_sent(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_duration_info or args.use_pause_info:
        num_prosodic_feats = int(args.use_duration_info) + int(args.use_pause_info)
        model = GPT2WithProsodyClassifier(num_prosodic_feats, prosody_emb_dim=args.prosody_emb_size).to(device)
    else:
        model = GPT2Classifier().to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    l, j, = [int(len(examples) * 0.8), int(len(examples) * 0.9)]
    train_ex, val_ex, test_ex = examples[:l], examples[l:j], examples[j:]

    train_loader = DataLoader(PhraseBoundaryDataset(train_ex), batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(PhraseBoundaryDataset(val_ex), batch_size=8, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(PhraseBoundaryDataset(test_ex), batch_size=8, shuffle=False, collate_fn=collate_fn)

    train_loss = train(args, model, train_loader, optimizer, device)
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
    parser.add_argument("--use_pause_info", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--prosody_emb_size", default=16, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)
    # main(args)
    args.use_duration_info = True
    args.use_pause_info = True
    _test(args)
