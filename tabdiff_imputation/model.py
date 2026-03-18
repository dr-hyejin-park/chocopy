"""
TabDiff Denoising Network.

Architecture:
- Sinusoidal time embedding + 2-layer MLP time projector
- Per-categorical-feature learnable embeddings (vocab_size+1 for MASK token)
- Deep residual MLP with FiLM time-conditioning
- Output head: noise prediction (numeric) + logits (each categorical feature)
"""
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional embedding for integer timesteps."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / max(half - 1, 1)
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


# ─────────────────────────────────────────────────────────────────────────────
class FiLM(nn.Module):
    """Feature-wise Linear Modulation: scale + shift conditioned on time."""

    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * feature_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return x * (1.0 + scale) + shift


# ─────────────────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    """Residual block with LayerNorm, GELU, and FiLM time conditioning."""

    def __init__(self, in_dim: int, out_dim: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.film = FiLM(cond_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.skip = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(self.norm1(x)))
        h = self.film(h, cond)
        h = self.drop(self.act(self.fc2(self.norm2(h))))
        return h + self.skip(x)


# ─────────────────────────────────────────────────────────────────────────────
class TabDiffDenoiser(nn.Module):
    """
    Mixed-type denoising network for TabDiff.

    Inputs
    ------
    x_num  : [B, N_num]  noisy continuous features
    x_cat  : [B, N_cat]  categorical indices (MASK token = C_i for feature i)
    t      : [B]         integer timesteps in [0, T-1]

    Outputs
    -------
    noise_pred : [B, N_num]           predicted noise for continuous features
    logits_cat : list of [B, C_i+1]  per-feature logits (first C_i = true classes)
    """

    def __init__(
        self,
        num_numeric: int,
        cat_vocab_sizes: List[int],
        d_embed_cat: int = 8,
        d_time: int = 128,
        hidden_dims: List[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [2048, 2048, 1024, 1024]

        self.num_numeric = num_numeric
        self.cat_vocab_sizes = cat_vocab_sizes
        self.num_categorical = len(cat_vocab_sizes)
        self.d_time = d_time

        # ── Time embedding ────────────────────────────────────────────────────
        cond_dim = d_time * 4
        self.time_proj = nn.Sequential(
            nn.Linear(d_time, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # ── Categorical embeddings (vocab_size + 1 for MASK) ──────────────────
        self.cat_embeds = nn.ModuleList([
            nn.Embedding(C + 1, d_embed_cat)
            for C in cat_vocab_sizes
        ])

        # ── Input projection ──────────────────────────────────────────────────
        input_dim = num_numeric + len(cat_vocab_sizes) * d_embed_cat
        h0 = hidden_dims[0]
        self.input_proj = nn.Linear(input_dim, h0)

        # ── Residual MLP blocks ───────────────────────────────────────────────
        self.blocks = nn.ModuleList()
        dims = hidden_dims
        for i in range(len(dims)):
            in_d = dims[i]
            out_d = dims[i + 1] if i + 1 < len(dims) else dims[-1]
            self.blocks.append(ResBlock(in_d, out_d, cond_dim, dropout))

        final_dim = dims[-1]

        # ── Output heads ──────────────────────────────────────────────────────
        self.num_head = nn.Sequential(
            nn.LayerNorm(final_dim),
            nn.Linear(final_dim, num_numeric),
        )
        # Per-feature categorical heads
        self.cat_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(final_dim),
                nn.Linear(final_dim, C + 1),  # first C = true classes, last = MASK
            )
            for C in cat_vocab_sizes
        ])

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------
    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        # Time conditioning
        t_sin = sinusoidal_embedding(t, self.d_time)          # [B, d_time]
        cond = self.time_proj(t_sin)                          # [B, cond_dim]

        # Categorical embeddings → flatten
        cat_emb = torch.cat(
            [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeds)],
            dim=-1,
        )  # [B, N_cat * d_embed]

        # Concatenate all inputs
        h = torch.cat([x_num, cat_emb], dim=-1)              # [B, input_dim]
        h = self.input_proj(h)

        # Residual MLP
        for block in self.blocks:
            h = block(h, cond)

        # Outputs
        noise_pred = self.num_head(h)                         # [B, N_num]
        logits_cat = [head(h) for head in self.cat_heads]    # list of [B, C_i+1]

        return noise_pred, logits_cat
