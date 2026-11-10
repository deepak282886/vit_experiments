"""
compare.py — Plot and compare wave vs transformer results

Usage:
    pip install matplotlib
    python compare.py
"""

import json
import os
import math

def load(path):
    if not os.path.exists(path):
        print(f"Missing: {path} — run train.py first.")
        return None
    with open(path) as f:
        return json.load(f)

wave = load("wave_results.json")
tx   = load("transformer_results.json")

if not wave or not tx:
    exit(1)

# ─────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "═"*62)
print(f"  WAVE MODEL  vs  TRANSFORMER  —  Tiny Shakespeare (PyTorch)")
print("═"*62)

metrics = [
    ("Parameters",      wave["parameters"],           tx["parameters"],           False, "{:,}"),
    ("Best val loss",   wave["best_val_loss"],         tx["best_val_loss"],         False, "{:.4f}"),
    ("Final perplexity",wave["final_perplexity"],      tx["final_perplexity"],      False, "{:.2f}"),
    ("Final val acc",   wave["final_val_acc"]*100,     tx["final_val_acc"]*100,     True,  "{:.2f}%"),
    ("Train time (s)",  wave["total_time_s"],          tx["total_time_s"],          False, "{:.1f}s"),
]

for label, wv, tv, higher_better, fmt in metrics:
    w_str = fmt.format(wv)
    t_str = fmt.format(tv)
    w_win = wv >= tv if higher_better else wv <= tv
    w_mark = " ◄ BETTER" if w_win else ""
    t_mark = " ◄ BETTER" if not w_win else ""
    print(f"  {label:<20} Wave: {w_str:<14}{w_mark:<10}  TX: {t_str:<14}{t_mark}")

print("═"*62)

w_eff = wave["final_val_acc"]  / wave["parameters"] * 1e6
t_eff = tx["final_val_acc"]    / tx["parameters"]   * 1e6
ratio = tx["parameters"] / wave["parameters"]
print(f"  {'Efficiency':<20} Wave: {w_eff:.4f}            TX: {t_eff:.4f}  (acc / M params)")
print(f"  Parameter ratio: Transformer uses {ratio:.2f}x the parameters of Wave")
print("═"*62)

# ─────────────────────────────────────────────
# EPOCH TABLE
# ─────────────────────────────────────────────
print(f"\n  EPOCH-BY-EPOCH  |  Val Loss              |  Perplexity")
print(f"  {'Epoch':<8} {'Wave':<12} {'TX':<12} {'Better':<10}  {'Wave PPL':<12} {'TX PPL':<12}")
print(f"  {'─'*64}")

w_hist = {h["epoch"]: h for h in wave["history"]}
t_hist = {h["epoch"]: h for h in tx["history"]}
epochs = sorted(set(w_hist) & set(t_hist))

wave_wins = 0
for ep in epochs:
    wl  = w_hist[ep]["val_loss"]
    tl  = t_hist[ep]["val_loss"]
    wp  = w_hist[ep]["perplexity"]
    tp  = t_hist[ep]["perplexity"]
    better = "Wave ⟿" if wl < tl else "TX ⊞  "
    if wl < tl:
        wave_wins += 1
    print(f"  {ep:<8} {wl:<12.4f} {tl:<12.4f} {better:<10}  {wp:<12.2f} {tp:<12.2f}")

print(f"\n  Wave won {wave_wins}/{len(epochs)} epochs")

# ─────────────────────────────────────────────
# SAMPLE TEXT
# ─────────────────────────────────────────────
print(f"\n{'─'*62}")
print("  WAVE MODEL SAMPLE:")
print(f"{'─'*62}")
print(wave.get("sample", "N/A")[:400])

print(f"\n{'─'*62}")
print("  TRANSFORMER SAMPLE:")
print(f"{'─'*62}")
print(tx.get("sample", "N/A")[:400])

# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(14, 10), facecolor="#0a0a0f")
    gs  = gridspec.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.4)

    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]

    for ax in axes:
        ax.set_facecolor("#111118")
        ax.tick_params(colors="#6060a0")
        ax.spines[:].set_color("#1e1e2e")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color("#6060a0")

    w_ep  = [h["epoch"]      for h in wave["history"]]
    w_vl  = [h["val_loss"]   for h in wave["history"]]
    w_ppl = [h["perplexity"] for h in wave["history"]]
    w_acc = [h["val_acc"]*100 for h in wave["history"]]
    w_tl  = [h["train_loss"] for h in wave["history"]]

    t_ep  = [h["epoch"]      for h in tx["history"]]
    t_vl  = [h["val_loss"]   for h in tx["history"]]
    t_ppl = [h["perplexity"] for h in tx["history"]]
    t_acc = [h["val_acc"]*100 for h in tx["history"]]
    t_tl  = [h["train_loss"] for h in tx["history"]]

    wave_label = f"Wave ({wave['parameters']:,} params)"
    tx_label   = f"Transformer ({tx['parameters']:,} params)"

    # Val loss
    axes[0].plot(w_ep, w_vl, color="#00e5ff", lw=2, label=wave_label)
    axes[0].plot(t_ep, t_vl, color="#ff6b6b", lw=2, label=tx_label)
    axes[0].set_title("Validation Loss", color="#e0e0f0", pad=8)
    axes[0].set_xlabel("Epoch", color="#6060a0")
    axes[0].legend(facecolor="#111118", labelcolor="#e0e0f0", edgecolor="#1e1e2e", fontsize=8)
    axes[0].grid(color="#1e1e2e", lw=0.5)

    # Perplexity
    axes[1].plot(w_ep, w_ppl, color="#00e5ff", lw=2, label=wave_label)
    axes[1].plot(t_ep, t_ppl, color="#ff6b6b", lw=2, label=tx_label)
    axes[1].set_title("Perplexity (lower = better)", color="#e0e0f0", pad=8)
    axes[1].set_xlabel("Epoch", color="#6060a0")
    axes[1].legend(facecolor="#111118", labelcolor="#e0e0f0", edgecolor="#1e1e2e", fontsize=8)
    axes[1].grid(color="#1e1e2e", lw=0.5)

    # Accuracy
    axes[2].plot(w_ep, w_acc, color="#00e5ff", lw=2, label=wave_label)
    axes[2].plot(t_ep, t_acc, color="#ff6b6b", lw=2, label=tx_label)
    axes[2].set_title("Validation Accuracy %", color="#e0e0f0", pad=8)
    axes[2].set_xlabel("Epoch", color="#6060a0")
    axes[2].legend(facecolor="#111118", labelcolor="#e0e0f0", edgecolor="#1e1e2e", fontsize=8)
    axes[2].grid(color="#1e1e2e", lw=0.5)

    # Train loss
    axes[3].plot(w_ep, w_tl, color="#00e5ff", lw=2, label=wave_label)
    axes[3].plot(t_ep, t_tl, color="#ff6b6b", lw=2, label=tx_label)
    axes[3].set_title("Training Loss", color="#e0e0f0", pad=8)
    axes[3].set_xlabel("Epoch", color="#6060a0")
    axes[3].legend(facecolor="#111118", labelcolor="#e0e0f0", edgecolor="#1e1e2e", fontsize=8)
    axes[3].grid(color="#1e1e2e", lw=0.5)

    fig.suptitle("Wave Model vs Transformer — Tiny Shakespeare (PyTorch + CUDA)",
                 color="#e0e0f0", fontsize=13, y=1.01)

    plt.savefig("comparison_plot.png", dpi=150, bbox_inches="tight", facecolor="#0a0a0f")
    print(f"\n  Plot saved to comparison_plot.png")
    plt.show()

except ImportError:
    print("\n  pip install matplotlib  to generate plots")