"""
train.py — Training loop for Hierarchical Wave vs Transformer experiment

Usage:
    python train.py --model wave
    python train.py --model transformer
    python train.py --model both
"""

import argparse
import json
import math
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import build_loaders
from wave_model import WaveModel
from transformer_model import TransformerModel

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CFG = {
    "ctx_len"     : 64,
    "batch_size"  : 128,
    "epochs"      : 30,
    "lr"          : 3e-4,
    "weight_decay": 1e-2,
    "dropout"     : 0.1,
    "seed"        : 42,

    # Wave-Transformer hybrid
    "wave": {
        "n_freqs" : 8,
        "d_model" : 128,
        "n_heads" : 4,
        "n_layers": 2,
        "d_ff"    : 256,
    },

    # Transformer
    "transformer": {
        "d_model" : 128,
        "n_heads" : 4,
        "n_layers": 3,
        "d_ff"    : 256,
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# TRAIN ONE EPOCH
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss   = 0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
            loss   = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss   += loss.item() * len(y)
        total_tokens += len(y)

    return total_loss / total_tokens


# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct    = 0
    total      = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
            loss   = criterion(logits, y)

        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(dim=-1) == y).sum().item()
        total      += len(y)

    avg_loss   = total_loss / total
    perplexity = math.exp(min(avg_loss, 20))
    accuracy   = correct / total
    return avg_loss, perplexity, accuracy


# ─────────────────────────────────────────────
# GENERATE SAMPLE
# ─────────────────────────────────────────────
@torch.no_grad()
def generate(model, seed_text, c2i, i2c, n_chars=300, temperature=0.8):
    model.eval()
    ctx_len    = CFG["ctx_len"]
    vocab_size = len(c2i)

    context = [c2i.get(c, 0) for c in seed_text[-ctx_len:]]
    context = ([0] * (ctx_len - len(context)) + context)[-ctx_len:]
    out = seed_text

    for _ in range(n_chars):
        x      = torch.tensor([context], dtype=torch.long, device=DEVICE)
        logits = model(x)[0] / temperature
        probs  = torch.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, 1).item()
        context  = context[1:] + [next_idx]
        out     += i2c[next_idx]

    return out


# ─────────────────────────────────────────────
# FULL TRAINING RUN
# ─────────────────────────────────────────────
def run(model_name, train_loader, val_loader, vocab_size, c2i, i2c):
    set_seed(CFG["seed"])

    if model_name == "wave":
        model = WaveModel(
            vocab_size = vocab_size,
            ctx_len    = CFG["ctx_len"],
            c2i        = c2i,
            n_freqs    = CFG["wave"]["n_freqs"],
            d_model    = CFG["wave"]["d_model"],
            n_heads    = CFG["wave"]["n_heads"],
            n_layers   = CFG["wave"]["n_layers"],
            d_ff       = CFG["wave"]["d_ff"],
            dropout    = CFG["dropout"],
        )
    else:
        model = TransformerModel(
            vocab_size = vocab_size,
            ctx_len    = CFG["ctx_len"],
            d_model    = CFG["transformer"]["d_model"],
            n_heads    = CFG["transformer"]["n_heads"],
            n_layers   = CFG["transformer"]["n_layers"],
            d_ff       = CFG["transformer"]["d_ff"],
            dropout    = CFG["dropout"],
        )

    model     = model.to(DEVICE)
    n_params  = model.count_params()

    print(f"\n{'═'*60}")
    print(f"  Model      : {model_name.upper()}")
    if model_name == "wave":
        print(f"  Levels     : Character → Word → Phrase")
    print(f"  Parameters : {n_params:,}")
    print(f"{'═'*60}")
    print(f"  {'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Perplexity':<12} {'Acc %':<8} {'Time'}")
    print(f"  {'─'*62}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(),
                      lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=CFG["epochs"], eta_min=CFG["lr"] / 10)
    scaler    = torch.amp.GradScaler()

    history     = []
    train_start = time.time()
    best_val_loss = float("inf")

    for epoch in range(1, CFG["epochs"] + 1):
        ep_start = time.time()

        train_loss                = train_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss, perplexity, acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        ep_time = time.time() - ep_start
        print(f"  {epoch:<6} {train_loss:<12.4f} {val_loss:<12.4f} {perplexity:<12.2f} {acc*100:<8.2f} {ep_time:.1f}s")

        history.append({
            "epoch"      : epoch,
            "train_loss" : round(train_loss, 6),
            "val_loss"   : round(val_loss, 6),
            "perplexity" : round(perplexity, 4),
            "val_acc"    : round(acc, 6),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{model_name}_best.pt")

    total_time = time.time() - train_start
    print(f"  {'─'*62}")
    print(f"  Total time: {total_time:.1f}s")

    # Sample
    print(f"\n─── {model_name.upper()} SAMPLE OUTPUT ───")
    sample = generate(model, "HAMLET:\n", c2i, i2c)
    print(sample)

    # Save
    results = {
        "model"            : model_name,
        "parameters"       : n_params,
        "ctx_len"          : CFG["ctx_len"],
        "vocab_size"       : vocab_size,
        "epochs"           : CFG["epochs"],
        "total_time_s"     : round(total_time, 2),
        "best_val_loss"    : round(best_val_loss, 6),
        "final_val_loss"   : history[-1]["val_loss"],
        "final_val_acc"    : history[-1]["val_acc"],
        "final_perplexity" : history[-1]["perplexity"],
        "history"          : history,
        "sample"           : sample,
    }

    out_path = f"{model_name}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        choices=["wave", "transformer", "both"],
                        default="both")
    args = parser.parse_args()

    train_loader, val_loader, vocab_size, c2i, i2c = build_loaders(
        ctx_len    = CFG["ctx_len"],
        batch_size = CFG["batch_size"],
    )

    if args.model == "both":
        run("wave",        train_loader, val_loader, vocab_size, c2i, i2c)
        run("transformer", train_loader, val_loader, vocab_size, c2i, i2c)
    elif args.model == "wave":
        run("wave",        train_loader, val_loader, vocab_size, c2i, i2c)
    else:
        run("transformer", train_loader, val_loader, vocab_size, c2i, i2c)