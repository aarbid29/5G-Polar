#!/usr/bin/env python3
"""
Purpose:
- Generate ONE fixed batch of Polar-coded data
- Re-train on that same batch repeatedly
- Check if the model can fully overfit (loss → 0, BER → 0)

Failure here means a fundamental bug (data, LLRs, loss, or model).
"""

import os
import sys
import math
import random
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
sys.path.insert(0, SRC)

from generate_dataset import generate_data
from models.wrappers.mamba_32bits import MambaPolarDecoder


def calculate_loss(frozen_bit_prior, target_vector, predicted_vector, reliable_only=False):
    """
    Masked BCEWithLogits loss.

    frozen_bit_prior: 1 → frozen bit, 0 → message bit
    reliable_only=True → loss only on message bits
    """
    import torch.nn.functional as F
    device = predicted_vector.device

    # Mask selects which bits contribute to the loss
    if reliable_only:
        mask = (frozen_bit_prior != 1).float().to(device)  # message bits only
    else:
        mask = torch.ones_like(target_vector, dtype=torch.float32).to(device)

    # Element-wise BCE on raw logits (numerically stable)
    per_elem = F.binary_cross_entropy_with_logits(
        predicted_vector,
        target_vector.float().to(device),
        reduction='none'
    )

    # Apply mask
    masked = per_elem * mask

    # Normalize by number of active bits
    denom = mask.sum()
    if denom.item() == 0:
        return masked.mean()

    return masked.sum() / denom


def build_batch(batch_size, N, message_bit_size_choices, SNR_db):
    """
    Builds one batch of Polar-coded samples with exact LLR computation.

    Each sample may use a different message length K.
    """
    Ys = []        # raw channel outputs
    Frozens = []   # frozen-bit masks
    Targets = []   # true transmitted bits
    Ks = []        # message lengths

    for _ in range(batch_size):
        # Randomly select message length (code rate varies)
        K = int(np.random.choice(message_bit_size_choices))
        Ks.append(K)

        # Generate Polar-coded noisy sample
        y, frozen_prior, target = generate_data(
            message_bit_size=K,
            SNRs_db=[SNR_db]
        )

        Ys.append(y.astype(np.float32))
        Frozens.append(np.array(frozen_prior, dtype=np.int64))
        Targets.append(np.array(target, dtype=np.float32))

    # Stack into batch arrays
    Ys = np.stack(Ys, axis=0)        # (B, N)
    Frozens = np.stack(Frozens, axis=0)
    Targets = np.stack(Targets, axis=0)

    # --------------------------------------------------
    # Compute exact AWGN LLRs:
    # sigma² = 1 / (2 * R * SNR)
    # LLR = 2y / sigma²
    # --------------------------------------------------
    Ks_arr = np.array(Ks, dtype=float)
    code_rates = Ks_arr / float(N)
    SNR_lin = 10.0 ** (SNR_db / 10.0)
    sigma2 = 1.0 / (2.0 * code_rates * SNR_lin)
    sigma2 = sigma2[:, None]  # (B,1) for broadcasting
    llrs = 2.0 * Ys / sigma2  # (B,N)

    # Convert to torch tensors
    llr_t = torch.tensor(llrs, dtype=torch.float32)
    frozen_t = torch.tensor(Frozens, dtype=torch.long)
    snr_t = torch.tensor([SNR_db] * batch_size, dtype=torch.float32)
    target_t = torch.tensor(Targets, dtype=torch.float32)

    return llr_t, frozen_t, snr_t, target_t


def main():
    # Fix all randomness → deterministic batch
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device=', device)

    # Overfit-test hyperparameters
    N = 32
    message_bit_size = [8, 16]   # limited K values → easier to overfit
    batch_size = 256
    SNR_db = 10
    num_steps = 2000
    lr = 1e-3

    # Build ONE fixed batch (never changes)
    llr, frozen, snr, target = build_batch(
        batch_size=batch_size,
        N=N,
        message_bit_size_choices=message_bit_size,
        SNR_db=SNR_db
    )

    llr = llr.to(device)
    frozen = frozen.to(device)
    snr = snr.to(device)
    target = target.to(device)

    print('batch shapes llr,frozen,target:', llr.shape, frozen.shape, target.shape)
    print('target ones fraction:', (target==1.0).float().mean().item())

    # Instantiate a small Mamba-based Polar decoder
    model = MambaPolarDecoder(
        d_model=32,
        num_layer_encoder=1,
        num_layers_bimamba_block=4,
        seq_len=N,
        d_state=16,
        d_conv=4,
        expand=2
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=0.0
    )

    # Baseline losses for sanity checking
    with torch.no_grad():
        zero_logits = torch.zeros_like(model(llr, frozen, snr))
        import torch.nn.functional as F
        bce_zero = F.binary_cross_entropy_with_logits(zero_logits, target)

        # Constant predictor using empirical bit probability
        p_mean = float(target.mean().item())
        const_logit = torch.full_like(
            zero_logits,
            math.log(p_mean / max(1e-6, (1 - p_mean)))
        )
        bce_mean = F.binary_cross_entropy_with_logits(const_logit, target)

        print(
            'BCE baseline predict-0.5:',
            bce_zero.item(),
            'predict-mean:',
            bce_mean.item()
        )

    # Overfit loop: same batch every iteration
    last_print = -1
    for it in range(1, num_steps + 1):
        optimizer.zero_grad()

        outputs = model(llr, frozen, snr)

        # Loss over all bits (message + frozen)
        loss = calculate_loss(
            frozen,
            target,
            outputs,
            reliable_only=False
        )

        loss.backward()

        # Gradient norm → checks gradient flow
        total_grad = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_grad += float(p.grad.data.norm(2).item())

        optimizer.step()

        # Periodic BER evaluation
        if it % 50 == 0 or it == 1:
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long()

                frozen_mask = (frozen == 1)
                msg_mask = ~frozen_mask

                if msg_mask.sum() > 0:
                    ber_msg = (
                        (preds[msg_mask] != target.long()[msg_mask])
                        .float().mean().item()
                    )
                else:
                    ber_msg = float('nan')

                if frozen_mask.sum() > 0:
                    ber_frozen = (
                        (preds[frozen_mask] != target.long()[frozen_mask])
                        .float().mean().item()
                    )
                else:
                    ber_frozen = float('nan')

            print(
                f"it={it:4d} "
                f"loss={loss.item():.6f} "
                f"grad_norm={total_grad:.6f} "
                f"BER_msg={ber_msg:.6f} "
                f"BER_frozen={ber_frozen:.6f}"
            )

    # Final evaluation on the same batch
    with torch.no_grad():
        outputs = model(llr, frozen, snr)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).long()

        frozen_mask = (frozen == 1)
        msg_mask = ~frozen_mask

        ber_msg = (
            (preds[msg_mask] != target.long()[msg_mask])
            .float().mean().item()
        )
        ber_frozen = (
            (preds[frozen_mask] != target.long()[frozen_mask])
            .float().mean().item()
        )

    print('FINAL: BER_msg=', ber_msg, 'BER_frozen=', ber_frozen)


if __name__ == '__main__':
    main()
