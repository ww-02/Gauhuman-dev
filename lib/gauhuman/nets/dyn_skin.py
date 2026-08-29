"""Dynamic Skinning Weights (single-frame variant, RnD-Avatar-inspired).

Given a canonical Gaussian position x and the *current* body pose theta_t,
produce a per-Gaussian, per-joint logit correction Delta_w applied on top of
KNN-soft SMPL weights:

    w_final = softmax( log(bweights_soft + eps) + Delta_w )

The last layer of the MLP is zero-initialised so the module is identity at
init (Delta_w = 0), giving stable training that only departs from KNN-soft if
that improves reconstruction.

This is the "spatial-only, single-frame" variant of the Dynamic Skinning
Weights Encoder from RnD-Avatar (arXiv:2512.09335). No temporal attention or
spatial attention -- just a pointwise MLP -- to keep FPS impact minimal.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class FourierEmbedder(nn.Module):
    def __init__(self, in_dim: int, num_freqs: int, include_input: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.num_freqs = num_freqs
        self.include_input = include_input
        freq_bands = 2.0 ** torch.linspace(0.0, float(num_freqs - 1), num_freqs)
        self.register_buffer("freq_bands", freq_bands, persistent=False)
        self.out_dim = in_dim * (2 * num_freqs + (1 if include_input else 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim)
        outs = [x] if self.include_input else []
        for f in self.freq_bands:
            outs.append(torch.sin(x * f))
            outs.append(torch.cos(x * f))
        return torch.cat(outs, dim=-1)


class DynamicSkinWeights(nn.Module):
    """Point-wise MLP that predicts a per-Gaussian, per-joint logit residual.

    Args:
        num_joints: SMPL joint count (24 for SMPL).
        pos_pe_L: L for positional Fourier encoding on canonical xyz.
        hidden_dim: hidden width.
        n_layers: total number of Linear layers (including input/output).

    Forward:
        x_can:   (bs, N, 3) canonical Gaussian positions.
        theta_t: (bs, J*3) or (bs, J, 3) axis-angle pose of the *current* frame
                 (i.e. so3_log(rot_mats_source) or SMPL 'poses').
    Returns:
        delta_w: (bs, N, J) logit residual (zero-init at start of training).
    """

    def __init__(
        self,
        num_joints: int = 24,
        pos_pe_L: int = 4,
        hidden_dim: int = 128,
        n_layers: int = 3,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.pe = FourierEmbedder(in_dim=3, num_freqs=pos_pe_L, include_input=True)
        pose_dim = num_joints * 3
        in_dim = self.pe.out_dim + pose_dim
        assert n_layers >= 2
        layers = []
        prev = in_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_joints)
        # zero-init output head so Delta_w == 0 at init.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x_can: torch.Tensor, theta_t: torch.Tensor) -> torch.Tensor:
        bs, N, _ = x_can.shape
        # Fourier PE on positions.
        pe = self.pe(x_can)  # (bs, N, pe_dim)
        # Broadcast pose over N.
        if theta_t.dim() == 3:
            theta_t = theta_t.reshape(theta_t.shape[0], -1)  # (bs, J*3)
        theta_b = theta_t.unsqueeze(1).expand(bs, N, -1)
        feat = torch.cat([pe, theta_b], dim=-1)
        h = self.trunk(feat)
        delta_w = self.head(h)  # (bs, N, J)
        return delta_w
