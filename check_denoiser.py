"""
check_denoiser.py
-----------------
Verifies that GDSSDiffusionDenoiser returns continuous (non-binary) matrices.

Run from the dag-pnp/ directory:
    python check_denoiser.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from dag_pnp import GDSSDiffusionDenoiser

# ── Config ────────────────────────────────────────────────────────────────────
CKPT = 'gdss/checkpoints/sem_SF_d20/Apr16-01:12:06.pth'   # Transformer
D    = 20
SIGMAS = [0.05, 0.1, 0.2, 0.3, 0.5]   # noise levels to probe
DEVICE = 'cpu'

# ── Load denoiser ─────────────────────────────────────────────────────────────
print(f'Loading checkpoint: {CKPT}')
denoiser = GDSSDiffusionDenoiser.from_checkpoint(CKPT, device=DEVICE)
print('Denoiser loaded.\n')

# ── Basic sanity checks at a single sigma ─────────────────────────────────────
W   = torch.randn(D, D) * 0.3
out = denoiser(W, sigma=0.3)

print('── Basic checks (sigma=0.3) ──────────────────────────────────────────')
print(f'  dtype            : {out.dtype}')
print(f'  shape            : {out.shape}')
print(f'  min / max        : {out.min():.4f} / {out.max():.4f}')
print(f'  mean / std       : {out.mean():.4f} / {out.std():.4f}')
n_unique = out.unique().numel()
print(f'  unique values    : {n_unique}  (d*d={D*D} → fully continuous)')
n_binary = ((out == 0.0) | (out == 1.0)).sum().item()
print(f'  entries == 0 or 1: {n_binary}  (should be 0 for continuous output)')
print(f'  diagonal mean    : {out.diagonal().mean():.4f}  (should be ~0)')
print()

if n_unique <= 2:
    print('  !! WARNING: output appears binary — quantize() may be called inside the model.')
else:
    print('  OK: output is continuous.')
print()

# ── Sweep over sigma values ────────────────────────────────────────────────────
print('── Output stats across sigma values ─────────────────────────────────')
print(f'  {"sigma":>6}  {"t":>6}  {"min":>8}  {"max":>8}  {"mean":>8}  {"std":>8}  {"frac>0.5":>9}')
rows = []
for sigma in SIGMAS:
    W_s = torch.randn(D, D) * sigma
    out_s = denoiser(W_s, sigma=sigma)
    t_val = denoiser._sigma_to_t(sigma)
    frac  = (out_s.abs() > 0.5).float().mean().item()
    rows.append((sigma, t_val, out_s.min().item(), out_s.max().item(),
                 out_s.mean().item(), out_s.std().item(), frac))
    print(f'  {sigma:>6.2f}  {t_val:>6.3f}  {out_s.min():>8.4f}  {out_s.max():>8.4f}'
          f'  {out_s.mean():>8.4f}  {out_s.std():>8.4f}  {frac:>9.4f}')
print()

# ── Plot: value distribution at each sigma ────────────────────────────────────
fig, axes = plt.subplots(1, len(SIGMAS), figsize=(4 * len(SIGMAS), 3.5), sharey=False)
for ax, sigma in zip(axes, SIGMAS):
    W_s   = torch.randn(D, D) * sigma
    out_s = denoiser(W_s, sigma=sigma).flatten().numpy()
    ax.hist(out_s, bins=40, color='steelblue', alpha=0.8, density=True)
    ax.axvline( 0.5, color='red',  ls='--', lw=1.2, label='+thr')
    ax.axvline(-0.5, color='red',  ls='--', lw=1.2, label='-thr')
    ax.axvline( 0.0, color='gray', ls=':',  lw=1.0)
    ax.set_title(f'σ={sigma}', fontsize=10)
    ax.set_xlabel('denoised adj value')
    if ax is axes[0]:
        ax.set_ylabel('density')

fig.suptitle('Tweedie denoiser output distribution\n'
             'continuous = good  |  only {0,1} = quantize() leak', fontsize=11)
plt.tight_layout()
out_path = 'denoiser_check.png'
plt.savefig(out_path, dpi=120)
print(f'Plot saved to {out_path}')
plt.show()
