"""
TabDiff mixed-type diffusion process.

Continuous features  → Gaussian DDPM with cosine noise schedule
Categorical features → Absorbing (masked) diffusion with cosine masking schedule
"""
import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
def _cosine_alpha_bar(T: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine noise schedule from Nichol & Dhariwal (2021).
    Returns alpha_bar[0..T], shape [T+1].
    """
    t = torch.linspace(0, T, T + 1, dtype=torch.float64)
    f = torch.cos(((t / T) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alpha_bar = f / f[0]
    return alpha_bar.clamp(0.0, 1.0).float()


def _linear_alpha_bar(T: int) -> torch.Tensor:
    """Linear schedule: alpha_bar_t = 1 - t/T."""
    t = torch.arange(T + 1, dtype=torch.float32) / T
    return (1.0 - t).clamp(0.0, 1.0)


def _make_schedule(T: int, schedule: str, s: float) -> torch.Tensor:
    if schedule == "cosine":
        return _cosine_alpha_bar(T, s)
    elif schedule == "linear":
        return _linear_alpha_bar(T)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


# ─────────────────────────────────────────────────────────────────────────────
class TabDiffusion:
    """
    Mixed-type diffusion process for TabDiff.

    All schedule tensors are moved to `device` via .to(device).
    """

    def __init__(self, config):
        T = config.num_timesteps
        self.T = T

        # ── Continuous schedule ───────────────────────────────────────────────
        ab_num = _make_schedule(T, config.num_schedule, config.s_num)  # [T+1]
        # Prepend index-0 = 1.0 (no noise at t=0)
        self._ab_num = ab_num                           # shape [T+1]
        beta_num = torch.cat([
            torch.tensor([0.0]),
            (1.0 - ab_num[1:] / ab_num[:-1].clamp(min=1e-8)),
        ]).clamp(0.0, 0.999)                            # shape [T+1]
        self._beta_num = beta_num
        self._alpha_num = 1.0 - beta_num
        self._sqrt_ab_num = ab_num.sqrt()
        self._sqrt_1m_ab_num = (1.0 - ab_num).sqrt()

        # ── Categorical (absorbing) schedule ─────────────────────────────────
        # alpha_bar_cat[t] = probability that a token is NOT masked at step t
        ab_cat = _make_schedule(T, config.cat_schedule, config.s_cat)
        self._ab_cat = ab_cat                           # shape [T+1]

    # ------------------------------------------------------------------
    def to(self, device: str):
        self._device = device
        for attr in [
            "_ab_num", "_beta_num", "_alpha_num",
            "_sqrt_ab_num", "_sqrt_1m_ab_num", "_ab_cat",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def _at(self, buf: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Index schedule buffer at timesteps t → [B, 1]."""
        return buf[t].unsqueeze(-1)  # [B, 1]

    # ── Forward process ───────────────────────────────────────────────────────

    def q_sample_num(
        self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """q(x_t | x_0) = N(sqrt(ᾱ_t)·x_0, (1−ᾱ_t)·I)."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._at(self._sqrt_ab_num, t)          # [B, 1]
        sqrt_1mab = self._at(self._sqrt_1m_ab_num, t)
        x_t = sqrt_ab * x0 + sqrt_1mab * noise
        return x_t, noise

    def q_sample_cat(
        self,
        x0_cat: torch.Tensor,          # [B, N_cat] integer labels
        t: torch.Tensor,               # [B]
        cat_vocab_sizes: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Absorbing forward: replace each token with MASK (=C_i) independently
        with probability 1 − ᾱ_cat[t].
        Returns (x_t_cat, keep_mask) where keep_mask[b,i]=True means NOT masked.
        """
        B, N_cat = x0_cat.shape
        keep_prob = self._ab_cat[t].unsqueeze(-1).expand(B, N_cat)   # [B, N_cat]
        keep = torch.bernoulli(keep_prob).bool()          # True = original

        x_t = x0_cat.clone()
        for i, C in enumerate(cat_vocab_sizes):
            x_t[~keep[:, i], i] = C  # MASK token = vocab_size

        return x_t, keep

    # ── Reverse process (training-time) ───────────────────────────────────────

    def compute_loss(
        self,
        model_out_num: torch.Tensor,      # [B, N_num]  predicted noise
        noise_true: torch.Tensor,         # [B, N_num]  actual noise
        logits_cat: List[torch.Tensor],   # list of [B, C_i+1]
        x0_cat: torch.Tensor,             # [B, N_cat]  true class labels
        keep_mask_cat: torch.Tensor,      # [B, N_cat]  True = not masked
        lambda_num: float = 1.0,
        lambda_cat: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined diffusion loss:
          L_num : MSE between predicted and true noise (continuous)
          L_cat : CE on masked tokens only (categorical)
        """
        # Continuous MSE
        loss_num = F.mse_loss(model_out_num, noise_true)

        # Categorical CE – only on masked positions
        if logits_cat and keep_mask_cat is not None:
            cat_losses = []
            for i, (logits, C) in enumerate(zip(logits_cat, [l.shape[-1] - 1 for l in logits_cat])):
                is_masked = ~keep_mask_cat[:, i]  # positions that WERE masked
                if is_masked.sum() == 0:
                    continue
                targets = x0_cat[is_masked, i]         # true class
                cat_losses.append(F.cross_entropy(logits[is_masked], targets))
            loss_cat = torch.stack(cat_losses).mean() if cat_losses else torch.tensor(0.0, device=loss_num.device)
        else:
            loss_cat = torch.tensor(0.0, device=loss_num.device)

        total = lambda_num * loss_num + lambda_cat * loss_cat
        return total, loss_num, loss_cat

    # ── DDIM reverse sampling ─────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_step_num(
        self,
        model_out: torch.Tensor,    # [B, N_num] predicted noise at t
        x_t: torch.Tensor,          # [B, N_num] current noisy state
        t_cur: int,
        t_next: int,
        eta: float = 0.0,           # 0 = deterministic DDIM, 1 = DDPM
    ) -> torch.Tensor:
        """DDIM reverse step for continuous features."""
        ab_cur = self._sqrt_ab_num[t_cur] ** 2   # ᾱ_cur
        ab_next = self._sqrt_ab_num[t_next] ** 2  # ᾱ_next

        # Predict x0
        sqrt_ab_cur = self._sqrt_ab_num[t_cur]
        sqrt_1mab_cur = self._sqrt_1m_ab_num[t_cur]
        x0_pred = (x_t - sqrt_1mab_cur * model_out) / sqrt_ab_cur.clamp(1e-8)
        # After QuantileTransformer normalisation the data is ~N(0,1);
        # clamping to ±3 covers 99.7% of the distribution and prevents
        # extreme values from destabilising the reverse process.
        x0_pred = x0_pred.clamp(-3.0, 3.0)

        sqrt_ab_next = self._sqrt_ab_num[t_next]
        sqrt_1mab_next = self._sqrt_1m_ab_num[t_next]

        # Direction pointing to x_t
        sigma = eta * (
            (1 - ab_next) / (1 - ab_cur).clamp(min=1e-8) * (1 - ab_cur / ab_next.clamp(min=1e-8))
        ).clamp(min=0.0).sqrt()

        dir_xt = (1 - ab_next - sigma ** 2).clamp(min=0.0).sqrt() * model_out
        noise = torch.randn_like(x_t) if eta > 0 else 0.0

        x_next = sqrt_ab_next * x0_pred + dir_xt + sigma * noise
        return x_next

    @torch.no_grad()
    def ddim_step_cat(
        self,
        logits_list: List[torch.Tensor],   # list [B, C_i+1]
        x_t_cat: torch.Tensor,             # [B, N_cat]
        cat_vocab_sizes: List[int],
        t_cur: int,
        t_next: int,
    ) -> torch.Tensor:
        """
        One DDIM-style reverse step for categorical absorbing diffusion.
        For each masked position, we predict p(x0|xt) and sample.
        """
        x_next = x_t_cat.clone()
        ab_next = self._ab_cat[t_next].item()

        for i, (logits, C) in enumerate(zip(logits_list, cat_vocab_sizes)):
            is_masked = x_t_cat[:, i] == C            # still MASK at time t_cur
            if not is_masked.any():
                continue
            # Sample x0 from predicted logits (true classes only, not MASK)
            probs = torch.softmax(logits[:, :C], dim=-1)   # [B, C]
            # Guard: NaN probs (from -inf logits) or all-zero rows crash
            # torch.multinomial with a CUDA device-side assert that is then
            # reported asynchronously at the next embedding lookup.
            probs = probs.nan_to_num(nan=1.0 / max(C, 1)).clamp(min=1e-8)
            x0_sampled = torch.multinomial(probs, 1).squeeze(-1)  # [B]

            # With probability ab_next keep the sample, else re-mask
            keep = torch.bernoulli(
                torch.full((is_masked.sum().item(),), ab_next, device=x_t_cat.device)
            ).bool()
            positions = is_masked.nonzero(as_tuple=True)[0]
            x_next[positions[keep], i] = x0_sampled[positions[keep]]
            x_next[positions[~keep], i] = C  # re-mask

        return x_next

    # ── Full DDIM sampling (generation / imputation) ──────────────────────────

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        n_samples: int,
        num_numeric: int,
        cat_vocab_sizes: List[int],
        n_steps: int = 100,
        eta: float = 0.0,
        # Imputation conditioning
        x_obs_num: Optional[torch.Tensor] = None,  # [n_samples, N_num]
        x_obs_cat: Optional[torch.Tensor] = None,  # [n_samples, N_cat]
        obs_mask_num: Optional[torch.Tensor] = None,  # bool [n_samples, N_num], True=observed
        obs_mask_cat: Optional[torch.Tensor] = None,  # bool [n_samples, N_cat], True=observed
        n_resample: int = 1,
        device: str = "cpu",
        chunk_size: int = 128,         # process at most chunk_size rows at once (OOM guard)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DDIM sampling with optional Repaint-style conditioning for imputation.
        Processes in chunks to avoid OOM on large datasets.
        """
        # Build DDIM timestep subsequence
        step_indices = torch.linspace(0, self.T - 1, n_steps + 1).long()
        t_seq = list(reversed(step_indices[1:].tolist()))   # from T-1 down to 0
        t_next_seq = list(reversed(step_indices[:-1].tolist()))  # paired t_{i-1}

        all_num, all_cat = [], []

        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            B = end - start

            # Initialize from prior
            x_num = torch.randn(B, num_numeric, device=device)
            x_cat = torch.stack(
                [torch.tensor(C, device=device).expand(B) for C in cat_vocab_sizes], dim=1
            )  # [B, N_cat] — all MASK

            # Slice conditioning tensors
            on = obs_mask_num[start:end] if obs_mask_num is not None else None
            oc = obs_mask_cat[start:end] if obs_mask_cat is not None else None
            xon = x_obs_num[start:end] if x_obs_num is not None else None
            xoc = x_obs_cat[start:end] if x_obs_cat is not None else None

            for step_idx, (t_cur, t_next) in enumerate(zip(t_seq, t_next_seq)):
                t_tensor = torch.full((B,), t_cur, dtype=torch.long, device=device)

                for resample_idx in range(n_resample):
                    # ── Model forward ──────────────────────────────────────
                    # Clamp x_cat to valid embedding range [0, C_i] before
                    # every model call to prevent OOB from Repaint/re-noise.
                    for _i, _C in enumerate(cat_vocab_sizes):
                        x_cat[:, _i].clamp_(0, _C)
                    model.eval()
                    noise_pred, logits_list = model(x_num, x_cat, t_tensor)

                    # ── Reverse step ───────────────────────────────────────
                    x_num_prev = self.ddim_step_num(
                        noise_pred, x_num, t_cur, t_next, eta
                    )
                    x_cat_prev = self.ddim_step_cat(
                        logits_list, x_cat, cat_vocab_sizes, t_cur, t_next
                    )

                    # ── Repaint conditioning ────────────────────────────────
                    if xon is not None and on is not None and t_next > 0:
                        t_next_tensor = torch.full((B,), t_next, dtype=torch.long, device=device)
                        x_obs_noisy, _noise = self.q_sample_num(xon, t_next_tensor)
                        x_num_prev = torch.where(on, x_obs_noisy, x_num_prev)

                    if xoc is not None and oc is not None and t_next > 0:
                        x_cat_prev = torch.where(oc, xoc, x_cat_prev)

                    if resample_idx < n_resample - 1 and t_next > 0:
                        # Re-noisify for next resample iteration
                        t_tensor2 = torch.full((B,), t_cur, dtype=torch.long, device=device)
                        x_num, _noise2 = self.q_sample_num(x_num_prev, t_tensor2)
                        x_cat = x_cat_prev
                    else:
                        x_num = x_num_prev
                        x_cat = x_cat_prev

                # Final step: ensure observed values are exact
                if xon is not None and on is not None:
                    x_num = torch.where(on, xon, x_num)
                if xoc is not None and oc is not None:
                    x_cat = torch.where(oc, xoc, x_cat)

            all_num.append(x_num.cpu())
            all_cat.append(x_cat.cpu())

        return torch.cat(all_num, dim=0), torch.cat(all_cat, dim=0)
