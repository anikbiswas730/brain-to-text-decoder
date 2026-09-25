"""
src/model.py — B2TNetV4, the v6 acoustic model, plus its losses and the EMA.

    x [B,T,512]
      -> low-rank per-day adapter (rank 32, identity at init) + Softsign
      -> fixed Gaussian smoothing (depthwise, std 2 bins)
      -> in-model time masking
      -> patch embedding as a STRIDED Conv1d (patch 14, stride 4) -> d_model 512
      -> 4 x residual BiGRU (packed, enforce_sorted=False)
      -> 3 x Conformer block (RoPE MHSA + conv module + macaron FFN)
      -> phoneme CTC head (41 classes)
      + inter-CTC auxiliary heads after the GRU stack and after block 2

30.15M parameters; ~7.1 GB peak on a T4 at max_bins_per_batch = 32*1400.

Changes from the v2/v3 lineage, each of which fixed a measured failure:
  * ConvModule used GroupNorm(1, C) over [C,T], which pools over TIME and
    therefore leaks padded-frame statistics into valid frames. Replaced by a
    per-frame LayerNorm plus explicit masking.
  * The day adapter is an exact identity at init (V zero-init; the scale is
    parameterised as 1 + gain, not as a free scale), so weight decay pulls each
    session toward the SHARED solution instead of toward zero.
  * Patch embedding as a strided Conv1d never materialises the [B, T', 512*14]
    unfold tensor, which with its LayerNorm was ~40% of GPU memory.
  * ResBiGRU packs with enforce_sorted=False because a CR-CTC batch is two
    concatenated views and is not length-sorted.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import N_CLASSES, BLANK_ID


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------
def drop_path(x, p, training):
    if p == 0.0 or not training:
        return x
    keep = 1 - p
    mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
    return x * mask / keep


class RoPE(nn.Module):
    def __init__(self, dim, max_len=8192, base=10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        ang = torch.outer(torch.arange(max_len).float(), inv)
        self.register_buffer('cos', ang.cos().repeat_interleave(2, -1), persistent=False)
        self.register_buffer('sin', ang.sin().repeat_interleave(2, -1), persistent=False)

    def forward(self, x):
        T = x.shape[-2]
        cos = self.cos[:T].to(x.dtype)
        sin = self.sin[:T].to(x.dtype)
        x_r = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)
        return x * cos + x_r * sin


class MHSA(nn.Module):
    def __init__(self, d, h, p):
        super().__init__()
        self.h, self.dk, self.p = h, d // h, p
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d)
        self.rope = RoPE(self.dk)

    def forward(self, x, pad_mask):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.h, self.dk).transpose(1, 2) for t in (q, k, v)]
        q, k = self.rope(q), self.rope(k)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=pad_mask[:, None, None, :],
                                           dropout_p=self.p if self.training else 0.0)
        return self.out(o.transpose(1, 2).reshape(B, T, D))


class ConvModule(nn.Module):
    def __init__(self, d, kernel, p):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.dwnorm = nn.LayerNorm(d)          # per-frame: no padded-frame leakage
        self.act = nn.SiLU()
        self.pw2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(p)

    def forward(self, x, pad_mask):
        m = pad_mask[:, :, None].to(x.dtype)
        y = (self.norm(x) * m).transpose(1, 2)
        y = self.dw(F.glu(self.pw1(y), dim=1))
        y = self.act(self.dwnorm(y.transpose(1, 2)) * m).transpose(1, 2)
        return self.drop(self.pw2(y).transpose(1, 2))


class ConformerBlock(nn.Module):
    def __init__(self, d, h, d_ff, kernel, p, p_attn):
        super().__init__()

        def ff():
            return nn.Sequential(nn.Linear(d, d_ff), nn.SiLU(), nn.Dropout(p),
                                 nn.Linear(d_ff, d), nn.Dropout(p))

        self.ff1_norm, self.ff1 = nn.LayerNorm(d), ff()
        self.attn_norm, self.attn, self.attn_drop = nn.LayerNorm(d), MHSA(d, h, p_attn), nn.Dropout(p)
        self.conv = ConvModule(d, kernel, p)
        self.ff2_norm, self.ff2 = nn.LayerNorm(d), ff()
        self.final_norm = nn.LayerNorm(d)

    def forward(self, x, pad_mask, dp=0.0):
        x = x + 0.5 * drop_path(self.ff1(self.ff1_norm(x)), dp, self.training)
        x = x + drop_path(self.attn_drop(self.attn(self.attn_norm(x), pad_mask)), dp, self.training)
        x = x + drop_path(self.conv(x, pad_mask), dp, self.training)
        x = x + 0.5 * drop_path(self.ff2(self.ff2_norm(x)), dp, self.training)
        return self.final_norm(x)


class ResBiGRU(nn.Module):
    def __init__(self, d, hidden, p):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.gru = nn.GRU(d, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * hidden, d)
        self.drop = nn.Dropout(p)

    def forward(self, x, lengths_cpu):
        y = self.norm(x)
        packed = pack_padded_sequence(y, lengths_cpu, batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        return x + self.drop(self.proj(out.to(x.dtype)))


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
class B2TNetV4(nn.Module):
    ARCH = 'B2TNetV4'

    def __init__(self, n_days, cfg):
        super().__init__()
        C = cfg['input_size']
        self.cfg = cfg
        self.patch, self.stride = cfg['patch_size'], cfg['patch_stride']
        self.n_days, self.generic_idx = n_days, n_days
        self.day_dropout_p = cfg.get('day_dropout_p', 0.0)
        D = n_days + 1                      # +1 = the generic slot used for unseen days

        k = np.zeros(cfg['smooth_kernel_size'], np.float32)
        k[len(k) // 2] = 1
        gk = gaussian_filter1d(k, cfg['smooth_kernel_std'])
        gk = np.squeeze(gk[np.argwhere(gk > 0.01)])
        gk = gk / gk.sum()
        self.register_buffer('gauss_kernel', torch.tensor(gk, dtype=torch.float32).view(1, 1, -1))

        r = cfg.get('day_rank', 32)
        self.day_U = nn.Parameter(torch.randn(D, C, r) * 0.02)
        self.day_V = nn.Parameter(torch.zeros(D, r, C))          # exact identity at init
        self.day_bias = nn.Parameter(torch.zeros(D, 1, C))
        self.day_gain = nn.Parameter(torch.zeros(D, 1, C))       # scale = 1 + gain
        self.day_act = nn.Softsign()

        d = cfg['d_model']
        self.in_norm = nn.LayerNorm(C)
        self.in_conv = nn.Conv1d(C, d, kernel_size=self.patch, stride=self.stride)
        self.in_act = nn.GELU()
        self.in_drop = nn.Dropout(cfg['dropout'])
        self.grus = nn.ModuleList([ResBiGRU(d, cfg['gru_hidden'], cfg['dropout'])
                                   for _ in range(cfg['gru_layers'])])
        self.blocks = nn.ModuleList([ConformerBlock(d, cfg['n_heads'], cfg['d_ff'], cfg['conv_kernel'],
                                                    cfg['dropout'], cfg['attn_dropout'])
                                     for _ in range(cfg['n_conformer'])])
        self.dp_rates = [float(v) for v in torch.linspace(0, cfg['drop_path_rate'], cfg['n_conformer'])]
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, N_CLASSES)
        self.aux_head_gru = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, N_CLASSES))
        self.aux_head_mid = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, N_CLASSES))
        self.aux_mid_at = max(cfg['n_conformer'] - 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def day_parameters(self):
        """The adapter parameters get their own optimizer group: a higher LR
        (lr_day_scale) and their own weight decay."""
        return [self.day_U, self.day_V, self.day_bias, self.day_gain]

    def out_lengths(self, lengths):
        L = torch.clamp(lengths, min=self.patch)
        return torch.clamp((L - self.patch) // self.stride + 1, min=1)

    def forward(self, x, lengths, day_idx, time_mask=None, return_aux=False):
        B, T, C = x.shape
        if self.training and self.day_dropout_p > 0:
            # day dropout: some trials are routed through the generic slot, which
            # is what keeps that slot usable for sessions with no adapter at all.
            drop = torch.rand(B, device=day_idx.device) < self.day_dropout_p
            if drop.any():
                day_idx = day_idx.clone()
                day_idx[drop] = self.generic_idx
        U, V = self.day_U[day_idx].to(x.dtype), self.day_V[day_idx].to(x.dtype)
        x = x + torch.bmm(torch.bmm(x, U), V)
        x = x * (1.0 + self.day_gain[day_idx].to(x.dtype)) + self.day_bias[day_idx].to(x.dtype)
        x = self.day_act(x)

        x = x.transpose(1, 2)
        x = F.conv1d(x, self.gauss_kernel.repeat(C, 1, 1).to(x.dtype), padding='same', groups=C)
        if time_mask is not None:
            x = x.masked_fill(time_mask[:, None, :T], 0.0)
        if x.shape[-1] < self.patch:
            x = F.pad(x, (0, self.patch - x.shape[-1]))
        x = self.in_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.in_drop(self.in_act(self.in_conv(x))).transpose(1, 2)        # [B,T',d]

        out_len = torch.clamp(self.out_lengths(lengths.to(x.device)), max=x.shape[1])
        Tp = x.shape[1]
        pad_mask = torch.arange(Tp, device=x.device)[None, :] < out_len[:, None]
        x = x * pad_mask[:, :, None].to(x.dtype)
        lens_cpu = out_len.detach().cpu().clamp(min=1)
        for g in self.grus:
            x = g(x, lens_cpu)
        x = x * pad_mask[:, :, None].to(x.dtype)
        aux1 = self.aux_head_gru(x) if return_aux else None
        aux2 = None
        for i, blk in enumerate(self.blocks):
            x = blk(x, pad_mask, dp=self.dp_rates[i]) * pad_mask[:, :, None].to(x.dtype)
            if return_aux and (i + 1) == self.aux_mid_at:
                aux2 = self.aux_head_mid(x)
        lp = torch.log_softmax(self.head(self.norm(x)).float(), dim=-1).transpose(0, 1)   # [T',B,C]
        if return_aux:
            auxs = [torch.log_softmax(a.float(), -1).transpose(0, 1) for a in (aux1, aux2) if a is not None]
            return lp, out_len, auxs
        return lp, out_len


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------
def ctc_mean(lp, targets, out_len, tgt_len):
    return F.ctc_loss(lp, targets, out_len, tgt_len, blank=BLANK_ID,
                      reduction='mean', zero_infinity=True)


def cr_ctc_consistency(lp, out_len, tgt_len, B):
    """CR-CTC: symmetric KL between the two augmented views' frame posteriors,
    with a stop-gradient on each target. Summed over valid frames per utterance
    and divided by target length (the same scale as ctc reduction='mean'), then
    averaged over utterances."""
    lp1, lp2 = lp[:, :B], lp[:, B:]
    T = lp.shape[0]
    valid = (torch.arange(T, device=lp.device)[:, None] < out_len[None, :B]).float()     # [T,B]
    kl12 = F.kl_div(lp1, lp2.detach(), log_target=True, reduction='none').sum(-1)
    kl21 = F.kl_div(lp2, lp1.detach(), log_target=True, reduction='none').sum(-1)
    kl = 0.5 * (kl12 + kl21) * valid
    return (kl.sum(0) / tgt_len[:B].clamp(min=1).float()).mean()


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
class EMA:
    """Exponential moving average of the weights. EVERY checkpoint this pipeline
    writes holds EMA weights, and every evaluation scores the EMA model — the
    raw weights are only ever used to continue optimisation."""

    def __init__(self, model, decay):
        import copy
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.n = 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))      # warm-up the averaging horizon
        pe = [p for p in self.model.parameters()]
        pm = [p.detach() for p in model.parameters()]
        try:
            torch._foreach_lerp_(pe, pm, 1.0 - d)
        except Exception:
            for a, b in zip(pe, pm):
                a.mul_(d).add_(b, alpha=1 - d)


# ---------------------------------------------------------------------------
# inference helper
# ---------------------------------------------------------------------------
@torch.no_grad()
def forward_batch(model, batch, device, norm, clip, day_override=None, amp=True):
    """Normalise, clip, run the model. `day_override` remaps day indices — used
    at decode time to route sessions that no fold ever trained on through the
    generic adapter slot instead of a randomly-initialised one."""
    x = batch['neural'].to(device, non_blocking=True).float()
    x = torch.clamp((x - norm['mean'].to(device)) / norm['std'].to(device), -clip, clip)
    lengths = batch['lengths'].to(device)
    day = batch['day_idx'].to(device)
    if day_override is not None:
        day = day_override[day.cpu()].to(device)
    with torch.autocast('cuda', dtype=torch.float16, enabled=(amp and x.is_cuda)):
        lp, out_len = model(x, lengths, day)
    return lp.float(), out_len
