"""
ECG VQ-VAE Foundation Model — Stage 1: Tokenizer
=================================================
Dataset : PTB-XL (21,837 ECGs, 12-lead, 500 Hz)
          https://physionet.org/content/ptb-xl/

Setup:
    pip install torch wfdb neurokit2 pandas numpy matplotlib scipy

Download data first:
    mkdir -p data/ptbxl
    cd data/ptbxl
    wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
    # OR use the physionet CLI:
    # pip install physionet-build
    # physionet dl ptb-xl

    Alternatively, just set DATA_DIR to wherever you put the PTB-XL folder.
    The script expects:
        DATA_DIR/
            ptbxl_database.csv
            records500/         ← 500Hz recordings
                00000/
                    00001_hr.hea
                    00001_hr.dat
                    ...

Usage:
    python ecg_vqvae_train.py

Outputs (saved to ./outputs/):
    checkpoints/tokenizer_best.pt   ← best model weights
    ecg_reconstruction.png          ← original vs reconstructed
    ecg_loss_curve.png              ← training curves
    ecg_token_sequences.png         ← token patterns by HR
    codebook_usage.png              ← codebook utilization
"""

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import wfdb

# ──────────────────────────────────────────────────────────────
# CONFIG — change these to match your setup
# ──────────────────────────────────────────────────────────────

DATA_DIR    = "./data/ptbxl/physionet.org/files/ptb-xl/1.0.3"
OUTPUT_DIR  = "./outputs"
LEAD        = 1          # which ECG lead to use (0-11). Lead 1 = Lead II, good for rhythm
FS          = 500        # sampling rate (Hz) — use records500 folder
SEG_LEN     = 512        # samples per segment (~1 second at 500Hz)
SEG_OVERLAP = 0.5        # overlap between consecutive segments
N_PATIENTS  = 2000       # how many records to load (None = all ~21k)
BATCH_SIZE  = 128
N_EPOCHS    = 60
LR          = 3e-4
N_CODEBOOK  = 512        # vocabulary size
EMB_DIM     = 64         # token embedding dimension
N_FILTERS   = 32         # filterbank width
EMA_DECAY   = 0.99       # EMA decay for codebook update (fixes collapse)
SEED        = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ──────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────────────────────

class PTBXLDataset(Dataset):
    """
    Loads PTB-XL ECG records, segments them into fixed windows.
    Each item is a normalized single-lead segment of shape (1, SEG_LEN).
    """
    def __init__(self, records_df, data_dir, lead=1, fs=500,
                 seg_len=512, overlap=0.5):
        self.seg_len = seg_len
        self.step    = int(seg_len * (1 - overlap))
        self.lead    = lead
        self.data_dir = data_dir

        self.segments = []   # (signal_segment,)
        self.patient_ids = []
        self.hr_labels = []

        print(f"Loading {len(records_df)} records...")
        for _, row in records_df.iterrows():
            try:
                # PTB-XL stores path relative to data_dir
                rec_path = os.path.join(data_dir, row['filename_hr'])
                record = wfdb.rdrecord(rec_path)
                signal = record.p_signal[:, lead].astype(np.float32)

                # Segment
                for start in range(0, len(signal) - seg_len, self.step):
                    seg = signal[start:start + seg_len]
                    # Normalize per-segment (removes patient-specific offset/scale)
                    seg = (seg - seg.mean()) / (seg.std() + 1e-8)
                    self.segments.append(seg)
                    self.patient_ids.append(row['patient_id'])
                    # heart_axis from metadata (rough proxy; real HR needs R-peak detection)
                    self.hr_labels.append(row.get('heart_axis', 0))

            except Exception as e:
                continue  # skip corrupted records

        self.segments    = np.array(self.segments, dtype=np.float32)
        self.patient_ids = np.array(self.patient_ids)
        print(f"  Total segments: {len(self.segments):,}")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.segments[idx]).unsqueeze(0)  # (1, seg_len)
        return x, self.patient_ids[idx]


def load_ptbxl(data_dir, n_patients=None, lead=1, fs=500, seg_len=512):
    csv_path = os.path.join(data_dir, "ptbxl_database.csv")
    df = pd.read_csv(csv_path, index_col='ecg_id')

    # Only use 500Hz records
    df = df[df['filename_hr'].notna()]

    if n_patients is not None:
        df = df.sample(n=min(n_patients, len(df)), random_state=SEED)

    # Train/val split by patient (no patient leakage)
    patients = df['patient_id'].unique()
    np.random.shuffle(patients)
    split = int(0.85 * len(patients))
    train_patients = set(patients[:split])
    val_patients   = set(patients[split:])

    train_df = df[df['patient_id'].isin(train_patients)]
    val_df   = df[df['patient_id'].isin(val_patients)]

    print(f"Train patients: {len(train_patients)} | Val patients: {len(val_patients)}")

    train_ds = PTBXLDataset(train_df, data_dir, lead=lead, fs=fs, seg_len=seg_len)
    val_ds   = PTBXLDataset(val_df,   data_dir, lead=lead, fs=fs, seg_len=seg_len)

    return train_ds, val_ds


# ──────────────────────────────────────────────────────────────
# 2. MODEL: LEARNED FILTERBANK + EMA VQ-VAE
# ──────────────────────────────────────────────────────────────

class LearnedFilterbank(nn.Module):
    """
    Core idea: instead of fixed FFT/STFT, learn the optimal
    frequency decomposition directly from ECG data.

    Architecture: stacked 1D convolutions with increasing receptive field.
    - Early filters: fine-grained morphology (QRS complex shape)
    - Later filters: coarser structure (rhythm, respiratory modulation)
    """
    def __init__(self, n_filters=32):
        super().__init__()
        # Layer 1: short filters — capture fine morphology (P,Q,R,S,T waves)
        self.conv1 = nn.Conv1d(1, n_filters, kernel_size=16, padding=8)
        self.bn1   = nn.BatchNorm1d(n_filters)

        # Layer 2: medium filters — capture beat-level structure
        self.conv2 = nn.Conv1d(n_filters, n_filters*2, kernel_size=32, padding=16)
        self.bn2   = nn.BatchNorm1d(n_filters*2)

        # Layer 3: long filters — capture rhythm and inter-beat patterns
        self.conv3 = nn.Conv1d(n_filters*2, n_filters*2, kernel_size=64, padding=32)
        self.bn3   = nn.BatchNorm1d(n_filters*2)

    def forward(self, x):
        # x: (B, 1, T)
        x = F.gelu(self.bn1(self.conv1(x)))   # (B, n_filters, T)
        x = F.gelu(self.bn2(self.conv2(x)))   # (B, n_filters*2, T)
        x = F.gelu(self.bn3(self.conv3(x)))   # (B, n_filters*2, T)
        return x


class EMAVectorQuantizer(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average codebook updates.

    WHY EMA:
    Standard VQ-VAE uses gradient descent on the codebook, which causes
    'codebook collapse' — most tokens go unused because the gradient
    pushes everything toward a few central embeddings.

    EMA directly tracks which encoder outputs get assigned to each token
    and updates the token to be the running average of those outputs.
    This is more stable and results in much higher codebook utilization.

    Reference: van den Oord et al., "Neural Discrete Representation Learning" (2017)
    """
    def __init__(self, n_embeddings=512, embedding_dim=64, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.n_embeddings  = n_embeddings
        self.embedding_dim = embedding_dim
        self.decay         = decay
        self.epsilon       = epsilon

        # Codebook
        embedding = torch.randn(n_embeddings, embedding_dim)
        self.register_buffer('embedding', embedding)
        self.register_buffer('ema_cluster_size', torch.zeros(n_embeddings))
        self.register_buffer('ema_dw', embedding.clone())

    def forward(self, z):
        """
        z: (B, embedding_dim, T) — encoder output
        Returns:
            z_q:       quantized (B, embedding_dim, T)
            indices:   token ids (B*T,)
            loss:      commitment loss only (codebook updated via EMA)
        """
        # Reshape for distance computation
        z = z.permute(0, 2, 1).contiguous()          # (B, T, D)
        z_flat = z.view(-1, self.embedding_dim)        # (B*T, D)

        # Nearest codebook entry
        distances = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.embedding.T
            + self.embedding.pow(2).sum(1)
        )
        indices = distances.argmin(1)                  # (B*T,)

        # One-hot encode assignments
        one_hot = F.one_hot(indices, self.n_embeddings).float()  # (B*T, K)

        # EMA update (only during training)
        if self.training:
            # Count how many vectors were assigned to each code
            cluster_size = one_hot.sum(0)              # (K,)
            self.ema_cluster_size = (
                self.decay * self.ema_cluster_size
                + (1 - self.decay) * cluster_size
            )
            # Sum of encoder outputs assigned to each code
            dw = one_hot.T @ z_flat                   # (K, D)
            self.ema_dw = (
                self.decay * self.ema_dw
                + (1 - self.decay) * dw
            )
            # Laplace smoothing to prevent dead codes
            n = self.ema_cluster_size.sum()
            smoothed = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.n_embeddings * self.epsilon)
                * n
            )
            self.embedding = self.ema_dw / smoothed.unsqueeze(1)

        # Quantize
        z_q = self.embedding[indices]                  # (B*T, D)
        z_q = z_q.view(z.shape)                        # (B, T, D)

        # Straight-through estimator
        z_q_st = z + (z_q - z).detach()

        # Commitment loss: encoder output should stay close to chosen code
        commitment_loss = F.mse_loss(z_q.detach(), z)

        z_q_st = z_q_st.permute(0, 2, 1)              # (B, D, T)
        return z_q_st, indices, commitment_loss


class ECGEncoder(nn.Module):
    def __init__(self, n_filters=32, emb_dim=64):
        super().__init__()
        self.filterbank = LearnedFilterbank(n_filters)
        # Project filterbank output to embedding dimension
        self.proj = nn.Sequential(
            nn.Conv1d(n_filters*2, emb_dim, kernel_size=1),
            nn.BatchNorm1d(emb_dim),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.filterbank(x)
        x = self.proj(x)
        return x   # (B, emb_dim, T)


class ECGDecoder(nn.Module):
    """
    Reconstructs ECG waveform from quantized token embeddings.
    If reconstruction is good, the tokens carry all signal information.
    """
    def __init__(self, emb_dim=64, n_filters=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(emb_dim, n_filters*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_filters*2),
            nn.GELU(),
            nn.Conv1d(n_filters*2, n_filters*2, kernel_size=16, padding=8),
            nn.BatchNorm1d(n_filters*2),
            nn.GELU(),
            nn.Conv1d(n_filters*2, n_filters, kernel_size=32, padding=16),
            nn.BatchNorm1d(n_filters),
            nn.GELU(),
            nn.Conv1d(n_filters, 1, kernel_size=64, padding=32),
        )

    def forward(self, x):
        return self.net(x)


class ECGVQVAE(nn.Module):
    """
    Full VQ-VAE for ECG tokenization.

    Pipeline:
        Raw ECG → Learned Filterbank → EMA VQ → Discrete Tokens
                                               ↓
                                          Decoder → Reconstructed ECG

    The discrete tokens are the 'ECG vocabulary' — feed these to
    a generative transformer in Stage 2.
    """
    def __init__(self, n_codebook=512, emb_dim=64, n_filters=32, ema_decay=0.99):
        super().__init__()
        self.encoder = ECGEncoder(n_filters, emb_dim)
        self.vq      = EMAVectorQuantizer(n_codebook, emb_dim, decay=ema_decay)
        self.decoder = ECGDecoder(emb_dim, n_filters)

    def forward(self, x):
        # x: (B, 1, T)
        z     = self.encoder(x)                        # (B, D, T)
        z_q, indices, commitment_loss = self.vq(z)     # quantize
        recon = self.decoder(z_q)                      # (B, 1, T')
        recon = recon[:, :, :x.shape[2]]               # match input length
        return recon, indices, commitment_loss

    def encode(self, x):
        """Get token sequence from raw ECG"""
        with torch.no_grad():
            z = self.encoder(x)
            _, indices, _ = self.vq(z)
        return indices

    def decode_tokens(self, indices, seq_len):
        """Reconstruct ECG from token indices"""
        with torch.no_grad():
            z_q = self.vq.embedding[indices]
            if z_q.dim() == 2:
                z_q = z_q.unsqueeze(0)
            z_q = z_q.permute(0, 2, 1)
            recon = self.decoder(z_q)[:, :, :seq_len]
        return recon


# ──────────────────────────────────────────────────────────────
# 3. TRAINING
# ──────────────────────────────────────────────────────────────

def train():
    # --- Load data ---
    train_ds, val_ds = load_ptbxl(
        DATA_DIR,
        n_patients=N_PATIENTS,
        lead=LEAD,
        fs=FS,
        seg_len=SEG_LEN,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # --- Model ---
    model = ECGVQVAE(
        n_codebook=N_CODEBOOK,
        emb_dim=EMB_DIM,
        n_filters=N_FILTERS,
        ema_decay=EMA_DECAY,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N_EPOCHS)

    train_losses, val_losses, codebook_usages = [], [], []
    best_val_loss = float('inf')

    print(f"\n{'Epoch':>6} {'Train':>10} {'Val':>10} {'Codebook':>12} {'LR':>10}")
    print("─" * 55)

    for epoch in range(N_EPOCHS):
        # ── Train ──
        model.train()
        epoch_loss = 0
        n_batches  = 0

        for x, _ in train_loader:
            x = x.to(DEVICE)
            recon, indices, commitment_loss = model(x)

            recon_loss = F.mse_loss(recon, x)
            loss = recon_loss + 0.25 * commitment_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        # ── Validate ──
        model.eval()
        val_loss   = 0
        all_indices = []

        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(DEVICE)
                recon, indices, _ = model(x)
                val_loss += F.mse_loss(recon, x).item()
                all_indices.append(indices.cpu())

        val_loss /= len(val_loader)
        all_indices = torch.cat(all_indices)
        n_used = all_indices.unique().numel()
        current_lr = scheduler.get_last_lr()[0]

        train_losses.append(epoch_loss / n_batches)
        val_losses.append(val_loss)
        codebook_usages.append(n_used)

        print(f"{epoch+1:>6} {train_losses[-1]:>10.4f} {val_loss:>10.4f} "
              f"{n_used:>6}/{N_CODEBOOK:<6} {current_lr:>10.2e}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': {
                    'n_codebook': N_CODEBOOK,
                    'emb_dim': EMB_DIM,
                    'n_filters': N_FILTERS,
                    'seg_len': SEG_LEN,
                    'lead': LEAD,
                    'fs': FS,
                }
            }, os.path.join(OUTPUT_DIR, "checkpoints", "tokenizer_best.pt"))

    return model, train_losses, val_losses, codebook_usages, val_ds


# ──────────────────────────────────────────────────────────────
# 4. EVALUATION & VISUALIZATION
# ──────────────────────────────────────────────────────────────

def plot_results(model, train_losses, val_losses, codebook_usages, val_ds):
    model.eval()
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=True)
    x_sample, _ = next(iter(val_loader))
    x_sample = x_sample.to(DEVICE)

    with torch.no_grad():
        recon, indices, _ = model(x_sample)

    x_np = x_sample.cpu().numpy()
    r_np = recon.cpu().numpy()

    # ── Figure 1: Reconstruction quality ──
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.suptitle('ECG VQ-VAE — Learned Filterbank Tokenizer\nOriginal vs Reconstructed', fontsize=14, fontweight='bold')
    t = np.arange(SEG_LEN) / FS * 1000  # time in ms

    for i in range(4):
        axes[i, 0].plot(t, x_np[i, 0], color='steelblue', linewidth=1.2)
        axes[i, 0].set_ylabel('Amplitude (norm)')
        axes[i, 0].set_title(f'Original — segment {i+1}')
        if i == 3:
            axes[i, 0].set_xlabel('Time (ms)')

        axes[i, 1].plot(t, x_np[i, 0], color='steelblue', linewidth=0.6, alpha=0.5, label='Original')
        axes[i, 1].plot(t, r_np[i, 0], color='tomato', linewidth=1.2, label='Reconstructed')
        axes[i, 1].set_title(f'Reconstructed from discrete tokens')
        axes[i, 1].legend(fontsize=8)
        if i == 3:
            axes[i, 1].set_xlabel('Time (ms)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ecg_reconstruction.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: ecg_reconstruction.png")

    # ── Figure 2: Training curves + codebook usage ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle('VQ-VAE Training Diagnostics', fontsize=13, fontweight='bold')

    axes[0].plot(train_losses, label='Train', color='steelblue')
    axes[0].plot(val_losses,   label='Val',   color='tomato')
    axes[0].set_title('Reconstruction Loss'); axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(codebook_usages, color='mediumseagreen')
    axes[1].axhline(N_CODEBOOK, color='gray', linestyle='--', alpha=0.5, label=f'Max ({N_CODEBOOK})')
    axes[1].set_title('Codebook Utilization'); axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Active tokens'); axes[1].legend()

    # Codebook usage histogram
    with torch.no_grad():
        all_idx = []
        loader = DataLoader(val_ds, batch_size=256)
        for x, _ in loader:
            _, idx, _ = model(x.to(DEVICE))
            all_idx.append(idx.cpu())
        all_idx = torch.cat(all_idx).numpy()

    counts = np.bincount(all_idx, minlength=N_CODEBOOK)
    axes[2].bar(range(N_CODEBOOK), np.sort(counts)[::-1], color='mediumpurple', width=1)
    axes[2].set_title('Token Usage Distribution\n(sorted by frequency)')
    axes[2].set_xlabel('Token rank'); axes[2].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ecg_training_diagnostics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: ecg_training_diagnostics.png")

    # ── Figure 3: Token sequences across different segments ──
    # Show that model produces consistent token patterns
    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    fig.suptitle('ECG Token Sequences — Discrete Representation\n(each position = one learned token ID)', fontsize=13, fontweight='bold')

    colors = ['steelblue', 'tomato', 'mediumseagreen']
    labels = ['Segment A', 'Segment B', 'Segment C']

    for i in range(3):
        x_i = x_sample[i:i+1]
        with torch.no_grad():
            _, idx_i, _ = model(x_i)
        token_seq = idx_i.cpu().numpy()

        # Show signal on top and token sequence on bottom
        ax = axes[i]
        t_tokens = np.arange(len(token_seq))
        ax.step(t_tokens, token_seq, color=colors[i], linewidth=1.5, where='mid')
        ax.fill_between(t_tokens, token_seq, alpha=0.2, color=colors[i], step='mid')
        ax.set_ylabel('Token ID')
        ax.set_title(f'{labels[i]} — {len(token_seq)} tokens for {SEG_LEN/FS:.2f}s of ECG')
        ax.set_ylim(0, N_CODEBOOK)

    axes[2].set_xlabel('Token position (time →)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ecg_token_sequences.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: ecg_token_sequences.png")

    print(f"\n✓ Best val loss : {min(val_losses):.4f}")
    print(f"✓ Final codebook usage: {codebook_usages[-1]}/{N_CODEBOOK} tokens ({100*codebook_usages[-1]/N_CODEBOOK:.1f}%)")
    print(f"✓ Checkpoint saved to: {OUTPUT_DIR}/checkpoints/tokenizer_best.pt")
    print(f"\nNext step: load the checkpoint and train a GPT-style transformer")
    print(f"on the token sequences to get the generative model (Stage 2).")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  ECG VQ-VAE Tokenizer — Stage 1")
    print("=" * 55)
    print(f"Data dir  : {DATA_DIR}")
    print(f"Lead      : {LEAD} (0-indexed)")
    print(f"Seg len   : {SEG_LEN} samples = {SEG_LEN/FS:.2f}s")
    print(f"Codebook  : {N_CODEBOOK} tokens × {EMB_DIM}d")
    print(f"Epochs    : {N_EPOCHS}")
    print()

    model, train_losses, val_losses, codebook_usages, val_ds = train()
    plot_results(model, train_losses, val_losses, codebook_usages, val_ds)