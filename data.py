"""
data.py — Download and prepare Tiny Shakespeare
"""

import os
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader

DATA_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "tinyshakespeare.txt"

def get_text():
    if not os.path.exists(DATA_PATH):
        print("Downloading Tiny Shakespeare...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        print("Done.")
    with open(DATA_PATH, "r") as f:
        return f.read()

class CharDataset(Dataset):
    def __init__(self, data, ctx_len):
        self.data    = data
        self.ctx_len = ctx_len

    def __len__(self):
        return len(self.data) - self.ctx_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.ctx_len]
        y = self.data[idx + self.ctx_len]
        return x, y

def build_loaders(ctx_len=64, batch_size=128, train_split=0.9):
    text  = get_text()
    chars = sorted(set(text))
    vocab_size = len(chars)
    c2i   = {c: i for i, c in enumerate(chars)}
    i2c   = {i: c for c, i in c2i.items()}

    data  = torch.tensor([c2i[c] for c in text], dtype=torch.long)
    split = int(len(data) * train_split)

    train_ds = CharDataset(data[:split],  ctx_len)
    val_ds   = CharDataset(data[split:],  ctx_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    print(f"Vocab size : {vocab_size}")
    print(f"Train tokens: {len(data[:split]):,}  Val tokens: {len(data[split:]):,}")
    return train_loader, val_loader, vocab_size, c2i, i2c