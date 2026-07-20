"""
The Hybrid LSTM + Transformer CTC acoustic model, enriched with:
  - Per-day (per-session) linear adaptation layers, to absorb day-to-day
    drift in neural recordings.
  - A learned Gaussian smoothing front-end over the raw features.
  - RoPE (rotary position embeddings) inside self-attention.
  - Stochastic depth (drop path) regularization on the Transformer blocks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .utils import drop_path


class RoPE(nn.Module):
    """Rotary position embedding applied to query/key vectors."""

    def __init__(self, head_dim, max_seq_len=2048):
        super().__init__()
        half_dim = head_dim // 2
        freq = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))
        t = torch.arange(max_seq_len).float().unsqueeze(1)
        angles = t * freq.unsqueeze(0)
        cos = torch.cos(angles).repeat_interleave(2, dim=1)
        sin = torch.sin(angles).repeat_interleave(2, dim=1)
        self.register_buffer("cos", cos.unsqueeze(0))
        self.register_buffer("sin", sin.unsqueeze(0))

    def forward(self, x, seq_len):
        cos = self.cos[:, :seq_len, :].to(x.device)
        sin = self.sin[:, :seq_len, :].to(x.device)
        x1, x2 = x.chunk(2, -1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class WideAttention(nn.Module):
    """Multi-head self-attention with a wider-than-usual head_dim and RoPE."""

    def __init__(self, d_model, n_heads, head_dim, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.inner_dim = n_heads * head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(d_model, self.inner_dim * 3, bias=False)
        self.out = nn.Linear(self.inner_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rope = RoPE(head_dim)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.rope(q, T)
        k = self.rope(k, T)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = self.dropout(F.softmax(attn, -1))
        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        return self.out(out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with stochastic depth around each
    residual branch."""

    def __init__(self, d_model, n_heads, d_ff, dropout, head_dim, attn_dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = WideAttention(d_model, n_heads, head_dim, attn_dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None, drop_path_rate=0.0):
        x = x + drop_path(self.attn(self.norm1(x), mask), drop_path_rate, self.training)
        x = x + drop_path(self.ffn(self.norm2(x)), drop_path_rate, self.training)
        return x


class HybridLSTMTransformerCTC(nn.Module):
    """CNN -> BiLSTM -> patch embedding -> Transformer -> CTC head, with a
    per-day linear adaptation layer and learned Gaussian smoothing applied
    to the raw input features before the CNN."""

    def __init__(self, n_days, input_size=512, d_model=384, n_heads=6, n_layers=4,
                 d_ff=1536, patch_size=3, vocab_size=50, dropout=0.4, head_dim=256,
                 attn_dropout=0.5, lstm_hidden=256, lstm_layers=2, smooth_std=2.0,
                 smooth_size=100, drop_path_rate=0.2):
        super().__init__()
        self.patch_size = patch_size
        self.n_days = n_days

        # --- Gaussian smoothing kernel (fixed, not learned) ---
        inp = np.zeros(smooth_size, dtype=np.float32)
        inp[smooth_size // 2] = 1
        gauss_kernel = gaussian_filter1d(inp, smooth_std)
        valid_idx = np.argwhere(gauss_kernel > 0.01)
        gauss_kernel = gauss_kernel[valid_idx]
        gauss_kernel = np.squeeze(gauss_kernel / np.sum(gauss_kernel))
        self.register_buffer("gauss_kernel", torch.tensor(gauss_kernel, dtype=torch.float32).view(1, 1, -1))

        # --- Day-specific linear adaptation ---
        self.day_weights = nn.ParameterList([nn.Parameter(torch.eye(input_size)) for _ in range(n_days)])
        self.day_biases = nn.ParameterList([nn.Parameter(torch.zeros(1, input_size)) for _ in range(n_days)])
        self.day_activation = nn.Softsign()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout * 0.5)
        )

        self.lstm = nn.LSTM(
            256, lstm_hidden, lstm_layers, batch_first=True, bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        lstm_output_dim = lstm_hidden * 2
        self.patch_embed = nn.Sequential(
            nn.LayerNorm(lstm_output_dim * patch_size),
            nn.Linear(lstm_output_dim * patch_size, d_model),
            nn.LayerNorm(d_model), nn.Dropout(dropout)
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, head_dim, attn_dropout)
            for _ in range(n_layers)
        ])

        self.drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, lengths, day_idx):
        B, T, C = x.shape

        # 1. Day-specific linear adaptation
        W = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        b = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)
        x = torch.einsum("btd,bdk->btk", x, W) + b
        x = self.day_activation(x)

        # 2. Gaussian smoothing via depthwise Conv1D
        x = x.permute(0, 2, 1)  # [B, C, T]
        kernel = self.gauss_kernel.repeat(C, 1, 1).to(x.device)
        x = F.conv1d(x, kernel, padding='same', groups=C)

        # 3. CNN feature extraction
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # [B, T, C]

        # 4. BiLSTM
        x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        lstm_out, _ = self.lstm(x_packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        # 5. Patch embedding (groups of `patch_size` timesteps into one token)
        T_lstm = lstm_out.shape[1]
        n_patches = T_lstm // self.patch_size

        if n_patches == 0:
            n_patches = 1
            x = lstm_out.mean(dim=1, keepdim=True)
            x = self.patch_embed(x.reshape(B, 1, -1))
            patch_lens = torch.ones(B, dtype=torch.long, device=x.device)
        else:
            x = lstm_out[:, :n_patches * self.patch_size].reshape(B, n_patches, -1)
            x = self.patch_embed(x)
            patch_lens = torch.clamp((lengths // self.patch_size).to(x.device), min=1)

        # 6. Transformer encoder
        mask = (torch.arange(n_patches, device=x.device)[None, :] < patch_lens[:, None])
        mask = mask[:, None, None, :]

        for i, block in enumerate(self.blocks):
            x = block(x, mask, drop_path_rate=self.drop_path_rates[i])

        # 7. CTC output projection
        logits = self.head(self.norm(x))
        log_probs = torch.log_softmax(logits, dim=-1)

        return log_probs.transpose(0, 1), patch_lens

    @classmethod
    def from_config(cls, config, n_days, vocab_size):
        """Convenience constructor that builds the model straight from the
        shared CONFIG dict plus the two values that depend on the dataset
        (number of days seen, and vocabulary size)."""
        return cls(
            n_days=n_days,
            input_size=512,
            d_model=config['d_model'], n_heads=config['n_heads'],
            n_layers=config['n_layers'], d_ff=config['d_ff'], patch_size=config['patch_size'],
            vocab_size=vocab_size, dropout=config['dropout'], head_dim=config['head_dim'],
            attn_dropout=config['attn_dropout'], lstm_hidden=config['lstm_hidden'],
            lstm_layers=config['lstm_layers'], smooth_std=config['smooth_kernel_std'],
            smooth_size=config['smooth_kernel_size'], drop_path_rate=config['drop_path_rate'],
        )
