#!/usr/bin/env python3
"""
Standalone overfit script: builds a single deterministic batch and attempts to overfit the model to it.

Usage: python scripts/overfit_one_batch.py

This script does NOT rely on the `PolarDecDataset` class; it calls `generate_data` directly
so we can compute the LLRs exactly and control randomness.
"""
import os
import sys
import math
import random
import numpy as np
import torch

# ensure src is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
sys.path.insert(0, SRC)

from generate_dataset import generate_data
from models.wrappers.mamba_32bits import MambaPolarDecoder


def calculate_loss(frozen_bit_prior, target_vector, predicted_vector, reliable_only=False):
    """Masked BCEWithLogitsLoss over message bits.

    frozen_bit_prior: (B, N) Long/Float tensor with 1 for frozen positions
    target_vector: (B, N) float tensor {0,1}
    predicted_vector: (B, N) raw logits
    """
    import torch.nn.functional as F
    device = predicted_vector.device
    if reliable_only:
        mask = (frozen_bit_prior != 1).float().to(device)  # 1 for message bits
    else:
        mask = torch.ones_like(target_vector, dtype=torch.float32).to(device)

    per_elem = F.binary_cross_entropy_with_logits(predicted_vector, target_vector.float().to(device), reduction='none')
    masked = per_elem * mask
    denom = mask.sum()
    if denom.item() == 0:
        return masked.mean()
    return masked.sum() / denom


def build_batch(batch_size, N, message_bit_size_choices, SNR_db):
    """Generate a batch where each sample may use a different message_bit_size.
    Returns: llr (B,N float32), frozen (B,N int64), snr (B float32), target (B,N float32), Ks (list)
    """
    Ys = []
    Frozens = []
    Targets = []
    Ks = []

    for _ in range(batch_size):
        K = int(np.random.choice(message_bit_size_choices))   # choose K for this sample
        Ks.append(K)
        y, frozen_prior, target = generate_data(message_bit_size=K, SNRs_db=[SNR_db])
        Ys.append(y.astype(np.float32))
        Frozens.append(np.array(frozen_prior, dtype=np.int64))
        Targets.append(np.array(target, dtype=np.float32))

    Ys = np.stack(Ys, axis=0)        # shape (B, N)
    Frozens = np.stack(Frozens, axis=0)
    Targets = np.stack(Targets, axis=0)

    # per-sample code rates and sigma^2
    Ks_arr = np.array(Ks, dtype=float)         # shape (B,)
    code_rates = Ks_arr / float(N)             # shape (B,)
    SNR_lin = 10.0 ** (SNR_db / 10.0)
    sigma2 = 1.0 / (2.0 * code_rates * SNR_lin)  # shape (B,)

    # expand sigma2 to (B,1) so broadcasting is explicit
    sigma2 = sigma2[:, None]  # shape (B,1)
    llrs = 2.0 * Ys / sigma2   # shape (B,N) – broadcasts sigma2 along N

    llr_t = torch.tensor(llrs, dtype=torch.float32)
    frozen_t = torch.tensor(Frozens, dtype=torch.long)
    snr_t = torch.tensor([SNR_db] * batch_size, dtype=torch.float32)
    target_t = torch.tensor(Targets, dtype=torch.float32)

    return llr_t, frozen_t, snr_t, target_t


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device=', device)

    # hyperparameters for the overfit test
    N = 32
    message_bit_size = [8, 16]  # fix K to make problem simpler
    batch_size = 256
    SNR_db = 10
    num_steps = 2000
    lr = 1e-3

    # build one deterministic batch
    llr, frozen, snr, target = build_batch(batch_size=batch_size, N=N, message_bit_size_choices=message_bit_size, SNR_db=SNR_db)
    llr = llr.to(device)
    frozen = frozen.to(device)
    snr = snr.to(device)
    target = target.to(device)

    print('batch shapes llr,frozen,target:', llr.shape, frozen.shape, target.shape)
    print('target ones fraction:', (target==1.0).float().mean().item())

    # instantiate a modest model to speed things up
    model = MambaPolarDecoder(d_model=32, num_layer_encoder=1, num_layers_bimamba_block=4, seq_len=N, d_state=16, d_conv=4, expand=2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    # baselines
    with torch.no_grad():
        zero_logits = torch.zeros_like(model(llr, frozen, snr))
        import torch.nn.functional as F
        bce_zero = F.binary_cross_entropy_with_logits(zero_logits, target)
        p_mean = float(target.mean().item())
        const_logit = torch.full_like(zero_logits, math.log(p_mean / max(1e-6, (1 - p_mean))))
        bce_mean = F.binary_cross_entropy_with_logits(const_logit, target)
        print('BCE baseline predict-0.5:', bce_zero.item(), 'predict-mean:', bce_mean.item())

    # overfit loop
    last_print = -1
    for it in range(1, num_steps + 1):
        optimizer.zero_grad()
        outputs = model(llr, frozen, snr)
        loss = calculate_loss(frozen, target, outputs, reliable_only=False)
        loss.backward()

        # gradient norm
        total_grad = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_grad += float(p.grad.data.norm(2).item())

        optimizer.step()

        if it % 50 == 0 or it == 1:
            # compute BER on message bits and frozen bits
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long()
                frozen_mask = (frozen == 1)
                msg_mask = ~frozen_mask
                if msg_mask.sum() > 0:
                    ber_msg = (preds[msg_mask] != target.long()[msg_mask]).float().mean().item()
                else:
                    ber_msg = float('nan')
                if frozen_mask.sum() > 0:
                    ber_frozen = (preds[frozen_mask] != target.long()[frozen_mask]).float().mean().item()
                else:
                    ber_frozen = float('nan')

            print(f"it={it:4d} loss={loss.item():.6f} grad_norm={total_grad:.6f} BER_msg={ber_msg:.6f} BER_frozen={ber_frozen:.6f}")

    # final diagnostics
    with torch.no_grad():
        outputs = model(llr, frozen, snr)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).long()
        frozen_mask = (frozen == 1)
        msg_mask = ~frozen_mask
        ber_msg = (preds[msg_mask] != target.long()[msg_mask]).float().mean().item()
        ber_frozen = (preds[frozen_mask] != target.long()[frozen_mask]).float().mean().item()
    print('FINAL: BER_msg=', ber_msg, 'BER_frozen=', ber_frozen)


if __name__ == '__main__':
    main()
