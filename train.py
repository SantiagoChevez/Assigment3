"""Train classifier using skip-gram document vectors and save model as `skipgram.pth`.

This script expects `vectorized_news_skip-gram_embeddings.csv` with columns:
  - `date`, `symbol`, `news_vector` (JSON list), `impact_score`.

Run:
  python train.py --data vectorized_news_skip-gram_embeddings.csv --epochs 10 --out skipgram.pth
"""
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from model import build_default_model

# Standard (fixed) settings — no CLI args
DATA_PATH = 'datasets/vectorized_news_skip_gram_embeddings.csv'
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
OUT_PATH = 'skipgram.pth'
NO_CUDA = False


class NewsDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def impact_to_label(x):
    try:
        v = float(x)
    except Exception:
        return None
    # handle NaN or infinite values
    if not np.isfinite(v):
        return None
    v = int(round(v))
    v = max(-3, min(3, v))
    return v + 3


def load_vectorized_csv(path):
    df = pd.read_csv(path)
    vecs = []
    labels = []
    for _, r in df.iterrows():
        nv = r.get('news_vector')
        if pd.isna(nv):
            continue
        try:
            arr = np.array(json.loads(nv), dtype=np.float32)
        except Exception:
            try:
                arr = np.array(eval(nv), dtype=np.float32)
            except Exception:
                continue
        lab = impact_to_label(r.get('impact_score'))
        if lab is None:
            continue
        vecs.append(arr)
        labels.append(lab)

    X = np.stack(vecs)
    y = np.array(labels)
    return X, y


def train_skip():
    device = torch.device('cuda' if torch.cuda.is_available() and not NO_CUDA else 'cpu')
    print('Device:', device)

    X, y = load_vectorized_csv(DATA_PATH)
    print('Loaded', X.shape[0], 'examples; vector dim =', X.shape[1])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_ds = NewsDataset(X_train, y_train)
    val_ds = NewsDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = build_default_model(input_size=X.shape[1])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        avg_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{EPOCHS} - loss: {avg_loss:.4f}')

        # validation
        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                preds.extend(batch_preds.tolist())
                trues.extend(yb.cpu().numpy().tolist())
        acc = accuracy_score(trues, preds)
        print(f'Validation accuracy: {acc:.4f}')

    print('Classification report:')
    print(classification_report(trues, preds, digits=4))

    # save model state_dict
    torch.save(model.state_dict(), OUT_PATH)
    print('Saved model to', OUT_PATH)


if __name__ == '__main__':
    train()
