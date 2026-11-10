"""
wave_model.py — Learnable Hierarchical Wave-Transformer Hybrid

Key insight from previous runs:
  Fixed Fourier features are a good inductive bias but cause plateaus —
  attention can't reshape a frozen encoding to fit what it discovers.

  Solution: make frequencies and phase scales learnable parameters.
  The Fourier structure is the initialization — the inductive bias.
  Training then refines it. Encoding and attention co-adapt.

Architecture:
  1. Learnable wave encoding at three hierarchy levels
     - Frequencies initialized as [1, 2, 3...n] but learned during training
     - Phase scales (word length norm, phrase length norm) also learned
     - Character, word, phrase levels each have their own frequency sets
  2. Linear projection to d_model
  3. N transformer blocks (causal self-attention + FFN)
  4. Output head to vocab

The model starts knowing language has wave-like hierarchical structure.
Training then discovers which specific frequencies carry the most signal.
"""

import torch
import torch.nn as nn
import math


# ─────────────────────────────────────────────
# VECTORIZED POSITION-WITHIN-SEGMENT
# ─────────────────────────────────────────────

def positions_within_segment(boundary_mask):
    """
    boundary_mask: (B, T) bool
    Returns:       (B, T) float — position within current segment
    Fully vectorized using cummax — no Python loops.
    """
    B, T   = boundary_mask.shape
    device = boundary_mask.device
    counter    = torch.arange(T, device=device).float().unsqueeze(0).expand(B, T)
    reset_vals = counter * boundary_mask.float()
    carried, _ = reset_vals.cummax(dim=1)
    return (counter - carried).float()


# ─────────────────────────────────────────────
# LEARNABLE FOURIER FEATURES
# ─────────────────────────────────────────────

class LearnableWaveLevel(nn.Module):
    """
    One level of wave encoding with learnable frequencies and phase scale.

    Frequencies initialized as [1, 2, ..., n_freqs] — the natural harmonic
    series. But they are nn.Parameter — gradients move them during training.

    Phase scale initialized at a sensible physical value (word length,
    phrase length) but also learned — the model discovers the right scale.

    Output: (B, T, n_freqs * 2) — sin and cos per frequency per position
    """
    def __init__(self, vocab_size, n_freqs, init_phase_scale):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_freqs    = n_freqs

        # Learnable frequencies — initialized as harmonic series
        # Shape: (n_freqs,)
        init_freqs = torch.arange(1, n_freqs + 1, dtype=torch.float32)
        self.freqs = nn.Parameter(init_freqs)

        # Learnable phase scale — initialized at physical segment length
        # Scalar — the model learns the right normalization
        self.phase_scale = nn.Parameter(torch.tensor(init_phase_scale, dtype=torch.float32))

        # Learnable amplitude per frequency — lets model weight harmonics
        # Initialized to 1.0 (equal weighting)
        self.amplitudes = nn.Parameter(torch.ones(n_freqs, dtype=torch.float32))

    def forward(self, x, phase_positions):
        """
        x               : (B, T) integer token indices
        phase_positions : (B, T) float — raw position within segment
        Returns         : (B, T, n_freqs * 2)
        """
        # Clamp frequencies to stay positive — negative frequencies are redundant
        freqs = torch.clamp(self.freqs, min=0.1)

        # Character angle — encodes token identity
        # (B, T, n_freqs)
        char_angle = (
            2 * math.pi
            * freqs.view(1, 1, -1)
            * x.float().unsqueeze(-1)
            / self.vocab_size
        )

        # Phase from position within segment — encodes where we are
        # Learnable scale normalizes to [0, pi]
        phase_scale = torch.clamp(self.phase_scale, min=1.0)
        phase = (phase_positions / phase_scale * math.pi).unsqueeze(-1)  # (B, T, 1)

        angles = char_angle + phase  # (B, T, n_freqs)

        # Apply learnable amplitudes — model weights which harmonics matter
        amps = torch.clamp(self.amplitudes, min=0.0).view(1, 1, -1)

        sin_feat = amps * torch.sin(angles)  # (B, T, n_freqs)
        cos_feat = amps * torch.cos(angles)  # (B, T, n_freqs)

        return torch.cat([sin_feat, cos_feat], dim=-1)  # (B, T, n_freqs * 2)


# ─────────────────────────────────────────────
# HIERARCHICAL LEARNABLE WAVE ENCODER
# ─────────────────────────────────────────────

class HierarchicalWaveEncoder(nn.Module):
    """
    Three learnable wave levels — each with its own frequencies, phase scale,
    and amplitude weights. All co-adapt with the transformer during training.

    Level 1 — Character : phase = position in full context (scale ~ ctx_len)
    Level 2 — Word      : phase = position within word (scale init ~ 15)
    Level 3 — Phrase    : phase = position within phrase (scale init ~ 40)
    """
    def __init__(self, vocab_size, ctx_len, c2i, n_freqs=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.ctx_len    = ctx_len
        self.n_freqs    = n_freqs
        self.out_dim    = n_freqs * 2 * 3  # three levels concatenated

        # Context-level positions — fixed (0 to ctx_len-1)
        positions = torch.arange(ctx_len, dtype=torch.float32)
        self.register_buffer("ctx_positions", positions)

        # Three learnable encoding levels
        self.char_level   = LearnableWaveLevel(vocab_size, n_freqs,
                                               init_phase_scale=float(ctx_len))
        self.word_level   = LearnableWaveLevel(vocab_size, n_freqs,
                                               init_phase_scale=15.0)
        self.phrase_level = LearnableWaveLevel(vocab_size, n_freqs,
                                               init_phase_scale=40.0)

        # Boundary index sets
        word_ids, phrase_ids = [], []
        WORD_CODES   = {32, 9}
        PHRASE_CODES = {44, 46, 33, 63, 58, 59, 10, 13}

        for char, idx in c2i.items():
            if len(char) == 1:
                code = ord(char)
                if code in WORD_CODES:   word_ids.append(idx)
                if code in PHRASE_CODES: phrase_ids.append(idx)

        self.register_buffer("word_ids",
            torch.tensor(word_ids,   dtype=torch.long) if word_ids
            else torch.tensor([-1],  dtype=torch.long))
        self.register_buffer("phrase_ids",
            torch.tensor(phrase_ids, dtype=torch.long) if phrase_ids
            else torch.tensor([-1],  dtype=torch.long))

    def _boundary_mask(self, x, ids):
        return (x.unsqueeze(-1) == ids.view(1, 1, -1)).any(dim=-1)

    def forward(self, x):
        B, T = x.shape

        # Level 1: Character — position within full context
        char_pos  = self.ctx_positions[:T].unsqueeze(0).expand(B, T)
        char_feat = self.char_level(x, char_pos)       # (B, T, n_freqs*2)

        # Level 2: Word — position within current word
        word_mask = self._boundary_mask(x, self.word_ids)
        word_pos  = positions_within_segment(word_mask)
        word_feat = self.word_level(x, word_pos)       # (B, T, n_freqs*2)

        # Level 3: Phrase — position within current phrase
        phrase_mask = self._boundary_mask(x, self.phrase_ids)
        phrase_pos  = positions_within_segment(phrase_mask)
        phrase_feat = self.phrase_level(x, phrase_pos) # (B, T, n_freqs*2)

        # Concatenate — (B, T, n_freqs * 2 * 3)
        return torch.cat([char_feat, word_feat, phrase_feat], dim=-1)


# ─────────────────────────────────────────────
# TRANSFORMER BLOCKS
# ─────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, ctx_len, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = 1.0 / math.sqrt(self.d_head)

        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(ctx_len, ctx_len))
        self.register_buffer("mask", mask.view(1, 1, ctx_len, ctx_len))

    def forward(self, x):
        B, T, D = x.shape
        H = self.n_heads
        qkv = self.qkv(x).reshape(B, T, 3, H, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn   = self.drop(torch.softmax(scores, dim=-1))
        out    = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ctx_len, d_ff, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, ctx_len, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ─────────────────────────────────────────────
# WAVE-TRANSFORMER HYBRID — LEARNABLE VERSION
# ─────────────────────────────────────────────

class WaveModel(nn.Module):
    """
    Learnable Wave-Transformer Hybrid.

    Wave encoder is no longer frozen — frequencies, phase scales,
    and amplitudes are all parameters that train alongside attention.

    The inductive bias is the initialization:
      - Harmonic frequency series (1, 2, 3...)
      - Physically meaningful phase scales (ctx_len, word_len, phrase_len)
      - Equal amplitude weighting

    Training then discovers which harmonics matter for Shakespeare,
    what the true effective word and phrase scales are, and which
    amplitudes to boost or suppress.
    """
    def __init__(self, vocab_size, ctx_len, c2i,
                 n_freqs=8, d_model=128, n_heads=4,
                 n_layers=2, d_ff=256, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.ctx_len    = ctx_len

        # Learnable hierarchical wave encoder
        self.wave_encoder = HierarchicalWaveEncoder(
            vocab_size, ctx_len, c2i, n_freqs
        )
        enc_dim = self.wave_encoder.out_dim  # n_freqs * 2 * 3

        # Project wave features to d_model
        self.input_proj = nn.Linear(enc_dim, d_model)
        self.drop       = nn.Dropout(dropout)

        # Transformer blocks — attend over learned wave representations
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ctx_len, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

        print(f"  enc_dim    : {enc_dim} → d_model: {d_model}")
        print(f"  n_layers   : {n_layers} transformer blocks")
        print(f"  frequencies: learnable (init = harmonic series)")
        print(f"  phase scale: learnable (init = physical lengths)")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Learnable wave encoding — (B, T, enc_dim)
        wave = self.wave_encoder(x)

        # Project to d_model — (B, T, d_model)
        x = self.drop(self.input_proj(wave))

        # Transformer blocks — attention over wave representations
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.head(x[:, -1, :])

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_learned_freqs(self):
        """Inspect what frequencies the model learned."""
        return {
            "char_freqs"  : self.wave_encoder.char_level.freqs.detach().cpu().tolist(),
            "word_freqs"  : self.wave_encoder.word_level.freqs.detach().cpu().tolist(),
            "phrase_freqs": self.wave_encoder.phrase_level.freqs.detach().cpu().tolist(),
            "word_phase_scale"  : self.wave_encoder.word_level.phase_scale.item(),
            "phrase_phase_scale": self.wave_encoder.phrase_level.phase_scale.item(),
            "char_amplitudes"   : self.wave_encoder.char_level.amplitudes.detach().cpu().tolist(),
            "word_amplitudes"   : self.wave_encoder.word_level.amplitudes.detach().cpu().tolist(),
            "phrase_amplitudes" : self.wave_encoder.phrase_level.amplitudes.detach().cpu().tolist(),
        }