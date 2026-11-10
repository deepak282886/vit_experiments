"""
transformer_model.py — Clean minimal transformer in PyTorch

Architecture:
  - Token embedding table
  - Sinusoidal positional encoding added on top (standard approach)
  - N_LAYERS transformer blocks (causal self-attention + FFN)
  - Layer norm, residual connections throughout
  - Output projection to vocab

Fair comparison to wave model — same training setup, proper autograd.
"""

import torch
import torch.nn as nn
import math


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

        # Causal mask
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

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
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


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, ctx_len,
                 d_model=128, n_heads=4, n_layers=3, d_ff=256, dropout=0.1):
        super().__init__()
        self.ctx_len    = ctx_len
        self.vocab_size = vocab_size

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop    = nn.Dropout(dropout)

        # Sinusoidal positional encoding — fixed, bolted on
        pe = torch.zeros(ctx_len, d_model)
        pos = torch.arange(ctx_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, T, D)

        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, ctx_len, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f  = nn.LayerNorm(d_model)
        self.head  = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, x):
        B, T = x.shape
        tok  = self.tok_emb(x)                  # (B, T, D)
        x    = self.drop(tok + self.pe[:, :T])  # add positional encoding
        x    = self.blocks(x)                   # transformer blocks
        x    = self.ln_f(x)
        return self.head(x[:, -1, :])           # predict from last token

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)