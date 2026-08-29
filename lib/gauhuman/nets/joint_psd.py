"""
Per-joint Pose Space Deformation (PSD) for GauHuman.

Corrects LBS artifacts (volume collapse, "candy-wrapper"/糖葫芦 connections, skin
self-intersection, discontinuous contours, wrong Gaussian stretch direction) at
skeleton joints by learning a *local*, pose-dependent correction per joint:

    d_mu, d_s, d_r = f_j( log(R_j R_{j,0}^{-1}),  x_i^local,  n_i,  s_i )

Design choices:
  * per-joint small MLPs  -> fewer params than one global deformation MLP,
                             interpretable as "pose-space correction of LBS
                             distortion local to joint j"
  * so(3) log-map of the *local* relative joint rotation -> compact pose feature;
                             the elbow bend is not polluted by the shoulder rotation
  * local coordinates + smooth distance falloff (hard cutoff) -> correction only
                             touches Gaussians near the joint
  * zero-initialized output head -> training starts from identity (LBS unchanged)

Conventions (match the parent repo):
  * quaternions are (w, x, y, z), matching utils.general_utils.build_rotation
  * joint transforms are rigid (bs, J, 4, 4) in the SMPL-local frame
  * corrections are produced in CANONICAL (big-pose) space so the existing LBS
    step can rigidly articulate the corrected geometry
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general_utils import build_rotation


# SMPL-24 kinematic parents (standard SMPL tree, -1 = root)
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

# Aggregation table for joints whose OWN rel-log is a poor pose signal (spine
# joints hardly bend relative to their direct parent in typical ZJU poses, so
# the raw f_j is near-zero). For these joints we replace f_j with the *mean*
# rel-log of the listed neighbour joints -- letting spine1 see "how the hips
# and mid-back are moving right now", etc. The keys are joint indices; the
# values are indices of joints whose rel-log should be averaged.
AGGREGATION = {
    3: [1, 2, 6],              # spine1  <- L/R hip, spine2
    6: [3, 9, 13, 14],         # spine2  <- spine1, spine3, L/R collar
    9: [12, 13, 14, 16, 17],   # spine3  <- neck, L/R collar, L/R shoulder
}

# Joints where LBS artifacts are most visible (shoulders/collars for the armpit,
# elbows, wrists, hips, knees, ankles, neck) plus the three SMPL spine joints
# (3/6/9) that cover the large torso surface (belly, mid chest, upper chest/back).
IMPORTANT_JOINTS_SMPL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

# Per-joint influence radius in meters (SMPL is ~1.7m). Tune to your actor scale.
# Spine radii are intentionally large: torso is a broad surface with few joints,
# and PSD contributions are convex-blended so an overlapping region is fine.
SUGGESTED_RADII = {
    1: 0.10, 2: 0.10,    # hips
    3: 0.22,             # spine1 (lower back / belly)
    4: 0.07, 5: 0.07,    # knees
    6: 0.22,             # spine2 (mid chest / back)
    7: 0.05, 8: 0.05,    # ankles
    9: 0.20,             # spine3 (upper chest / shoulder blades)
    12: 0.06,            # neck
    13: 0.15, 14: 0.15,  # collars (armpit -> shoulder region), widened from 0.10
    16: 0.10, 17: 0.10,  # shoulders
    18: 0.06, 19: 0.06,  # elbows
    20: 0.04, 21: 0.04,  # wrists
}


# ---------------------------------------------------------------------------
# SO(3) / quaternion helpers
# ---------------------------------------------------------------------------

def skew(v: torch.Tensor) -> torch.Tensor:
    """(..., 3) -> (..., 3, 3) skew-symmetric matrix."""
    z = torch.zeros_like(v[..., 0])
    return torch.stack(
        [
            torch.stack([z, -v[..., 2], v[..., 1]], dim=-1),
            torch.stack([v[..., 2], z, -v[..., 0]], dim=-1),
            torch.stack([-v[..., 1], v[..., 0], z], dim=-1),
        ],
        dim=-2,
    )


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """so(3) log map: SO(3) (..., 3, 3) -> axis-angle (..., 3). Stable near 0."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    sin_theta = theta.sin()

    w = 0.5 * torch.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        dim=-1,
    )
    safe_sin = torch.where(theta > 1e-4, sin_theta, torch.ones_like(sin_theta))
    axis = w / safe_sin.unsqueeze(-1)
    return axis * theta.unsqueeze(-1)


def invert_rigid(T: torch.Tensor) -> torch.Tensor:
    """Invert rigid transform (..., 4, 4) = [[R, t], [0, 1]]."""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = R.transpose(-1, -2)
    T_inv = T.clone()
    T_inv[..., :3, :3] = Rt
    T_inv[..., :3, 3] = -(Rt @ t.unsqueeze(-1)).squeeze(-1)
    return T_inv


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of quaternions (w, x, y, z)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack([ow, ox, oy, oz], dim=-1)


def axis_angle_to_quat(w: torch.Tensor) -> torch.Tensor:
    """Axis-angle (..., 3) -> quaternion (..., 4) in (w, x, y, z)."""
    theta = w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    axis = w / theta
    half = 0.5 * theta
    return torch.cat([half.cos(), half.sin() * axis], dim=-1)


def apply_rotation_correction(
    q: torch.Tensor, d_rot: torch.Tensor, right: bool = False
) -> torch.Tensor:
    """Compose an axis-angle residual `d_rot` onto quaternion `q`.

    `d_rot` is a correction expressed in CANONICAL (world) space, so the default
    is a world-frame pre-multiply (right=False): q' = dq (x) q  ==  R(dq) @ R(q).
    This is consistent with the canonical-space position correction d_xyz, and
    with how build_rotation maps local -> world.

    right=True instead post-multiplies (q (x) dq), a local/body-frame correction;
    only use it when you are feeding a local-frame residual.
    """
    dq = F.normalize(axis_angle_to_quat(d_rot), dim=-1)
    out = quat_mul(dq, q) if not right else quat_mul(q, dq)
    return F.normalize(out, dim=-1)


def gaussian_normal(rotation_quat: torch.Tensor, log_scales: torch.Tensor) -> torch.Tensor:
    """Per-Gaussian normal = direction of its shortest scale axis.

    The codebase stores no explicit normal; the principal surface normal is the
    axis the Gaussian is thinnest along.
    """
    R = build_rotation(rotation_quat)  # (N, 3, 3)
    idx = log_scales.argmin(dim=-1)  # (N,)
    arange = torch.arange(rotation_quat.shape[0], device=rotation_quat.device)
    return R[arange, :, idx]  # (N, 3)


# ---------------------------------------------------------------------------
# Per-joint MLP
# ---------------------------------------------------------------------------

def _build_mlp(in_dim: int, hidden_dim: int, n_layers: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    d = in_dim
    for _ in range(n_layers):
        layers.append(nn.Linear(d, hidden_dim))
        layers.append(nn.ELU())
        d = hidden_dim
    return nn.Sequential(*layers)


class JointPoseSpaceDeformation(nn.Module):
    """Split PSD head: low-freq (pose-only) + high-freq (local-geom-only).

    Two zero-init heads that live in *orthogonal input spaces*:
      * LF: `pose_pe`                         -> (out_dim,)          per joint
      * HF: `[xyz_pe, normal, log_scale]`     -> (N, out_dim)        per Gaussian

    Final correction = LF + HF, so gradients on the two paths cannot cancel
    each other (they respond to disjoint inputs). Zero-init on both heads
    keeps identity at initialisation.

    When `split=False`, falls back to the original single-MLP behaviour
    consuming the concatenated feature.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        out_dim: int = 10,
        split: bool = False,
        lf_in_dim: int = 0,
        hf_in_dim: int = 0,
    ) -> None:
        super().__init__()
        self.split = bool(split)
        if not self.split:
            self.net = _build_mlp(in_dim, hidden_dim, n_layers)
            self.head = nn.Linear(hidden_dim, out_dim)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        else:
            assert lf_in_dim > 0 and hf_in_dim > 0
            self.lf_net = _build_mlp(lf_in_dim, hidden_dim, n_layers)
            self.lf_head = nn.Linear(hidden_dim, out_dim)
            self.hf_net = _build_mlp(hf_in_dim, hidden_dim, n_layers)
            self.hf_head = nn.Linear(hidden_dim, out_dim)
            for h in (self.lf_head, self.hf_head):
                nn.init.zeros_(h.weight)
                nn.init.zeros_(h.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(feat))


# ---------------------------------------------------------------------------
# Full PSD module (all joints)
# ---------------------------------------------------------------------------

class PoseSpaceDeformationModule(nn.Module):
    """Per-joint pose-space correction for a set of important joints.

    Args:
        joint_indices: which joints get a correction MLP.
        parents:       parent index per joint (len == n_joints). Joints with a valid
                       parent use the *local* relative rotation as the pose feature,
                       otherwise the global relative rotation.
        radii:         per-joint influence radius (meters), keyed by joint index.
        default_radius: fallback radius.
        cutoff:        hard localization cutoff as a multiple of the radius.
        use_normal, use_scale: include local normal / log-scale in the input.
    """

    def __init__(
        self,
        joint_indices: List[int],
        parents: Optional[List[int]] = None,
        hidden_dim: int = 96,
        n_layers: int = 2,
        radii: Optional[Dict[int, float]] = None,
        default_radius: float = 0.08,
        cutoff: float = 3.0,
        use_normal: bool = True,
        use_scale: bool = True,
        pe_L: int = 4,
        rot_clamp: float = 0.1,
        alpha_clamp: float = 1.0,
        split: bool = False,
    ) -> None:
        super().__init__()
        self.joint_indices = list(joint_indices)
        self.parents = parents
        self.radii = radii if radii is not None else {}
        self.default_radius = default_radius
        self.cutoff = cutoff
        self.use_normal = use_normal
        self.use_scale = use_scale
        # PE freqs: 2^0..2^(L-1). Applied to pose_feat and normalized local xyz.
        self.pe_L = int(pe_L)
        self.register_buffer(
            "pe_freqs",
            torch.tensor([2.0 ** i for i in range(self.pe_L)], dtype=torch.float32) if self.pe_L > 0 else torch.zeros(0),
        )
        # d_rot is composed onto the Gaussian quaternion; clamp to avoid extreme
        # long-axis flips at large joint bends. |d_rot| <= rot_clamp rad.
        self.rot_clamp = float(rot_clamp)
        self.alpha_clamp = float(alpha_clamp)

        pe_extra = 2 * 3 * self.pe_L  # sin+cos of L freqs for each 3-vec input we encode
        pose_dim = 3 + pe_extra                       # pose_feat + PE
        xyz_dim = 3 + pe_extra                        # local_xyz/r + PE
        geom_extra = (3 if use_normal else 0) + (3 if use_scale else 0)
        hf_dim = xyz_dim + geom_extra                 # HF branch: geom-only
        in_dim = pose_dim + hf_dim                    # single-head fallback

        self.split = bool(split)
        if not self.split:
            self.mlps = nn.ModuleList(
                [JointPoseSpaceDeformation(in_dim, hidden_dim, n_layers, out_dim=10)
                 for _ in self.joint_indices]
            )
        else:
            # Two orthogonal-input heads: LF sees pose only, HF sees geom only.
            self.mlps = nn.ModuleList([
                JointPoseSpaceDeformation(
                    0, hidden_dim, n_layers, out_dim=10,
                    split=True, lf_in_dim=pose_dim, hf_in_dim=hf_dim,
                ) for _ in self.joint_indices
            ])

    def _radius(self, j: int) -> float:
        return float(self.radii.get(j, self.default_radius))

    def _rel_log_for(self, joints, rest_T, pose_T):
        """Per-joint parent-relative SO(3) log for the given joint indices.

        Returns (len(joints), 3). Uses local (parent-relative) rel-log when the
        joint has a parent, else the global rel-log. Access pattern matches the
        original _pose_feature_batch, so it is safe against whatever J the
        SMPL/SMPLX layer produces.
        """
        idx_t = torch.tensor(list(joints), device=rest_T.device, dtype=torch.long)
        R0 = rest_T[idx_t, :3, :3]
        Rp = pose_T[idx_t, :3, :3]
        if self.parents is None:
            return so3_log(Rp @ R0.transpose(1, 2))
        has_parent = torch.tensor(
            [self.parents[j] >= 0 for j in joints], device=rest_T.device
        )
        parent_idx = torch.tensor(
            [self.parents[j] if self.parents[j] >= 0 else j for j in joints],
            device=rest_T.device, dtype=torch.long,
        )
        R0_p = rest_T[parent_idx, :3, :3]
        Rp_p = pose_T[parent_idx, :3, :3]
        rel0 = torch.einsum('kab,kbc->kac', R0_p.transpose(1, 2), R0)
        relp = torch.einsum('kab,kbc->kac', Rp_p.transpose(1, 2), Rp)
        feat_local = so3_log(relp @ rel0.transpose(1, 2))
        feat_global = so3_log(Rp @ R0.transpose(1, 2))
        return torch.where(has_parent[:, None], feat_local, feat_global)

    def _pose_feature_batch(self, idx, rest_T, pose_T):
        """Pose feature (K, 3) for the K joints in `idx`.

        Default: each joint's own parent-relative rel-log.
        Override: for joints in AGGREGATION, use the mean rel-log of listed
        neighbours -- so spine joints see hips/collars/shoulders, since their
        own rel-log is near zero and carries no usable pose signal.
        """
        feats = self._rel_log_for(idx, rest_T, pose_T).clone()  # (K, 3)
        for k, j in enumerate(idx):
            neigh = AGGREGATION.get(j)
            if neigh:
                feats[k] = self._rel_log_for(neigh, rest_T, pose_T).mean(dim=0)
        return feats

    def _pe(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 3) -> (..., 3 + 3*2*L). Sin/cos over freqs 2^0..2^(L-1).
        if self.pe_L <= 0:
            return x
        # broadcast: (..., 3, 1) * (L,) -> (..., 3, L)
        xf = x.unsqueeze(-1) * self.pe_freqs
        s = xf.sin()
        c = xf.cos()
        # flatten last two dims -> (..., 3*L*2)
        enc = torch.stack([s, c], dim=-1).reshape(*x.shape[:-1], 3 * self.pe_L * 2)
        return torch.cat([x, enc], dim=-1)

    @staticmethod
    def _run_batched(mlps, feat, net_attr: str, head_attr: str):
        """Apply a per-joint 2-layer MLP + head across (K, ..., in_dim) inputs.

        `feat` may have shape (K, in_dim) — a per-joint vector — or
        (K, N, in_dim) — a per-Gaussian tensor. The einsum broadcasts the same
        way in either case because we contract only the input axis.
        """
        nets = [getattr(m, net_attr) for m in mlps]
        heads = [getattr(m, head_attr) for m in mlps]
        W1 = torch.stack([n[0].weight for n in nets])   # (K, hidden, in)
        b1 = torch.stack([n[0].bias for n in nets])     # (K, hidden)
        W2 = torch.stack([n[2].weight for n in nets])
        b2 = torch.stack([n[2].bias for n in nets])
        Wh = torch.stack([h.weight for h in heads])     # (K, out, hidden)
        bh = torch.stack([h.bias for h in heads])

        # Broadcast bias over any middle dim (N) if present.
        if feat.dim() == 2:
            h = torch.einsum('koi,ki->ko', W1, feat) + b1
            h = F.elu(h)
            h = torch.einsum('koi,ki->ko', W2, h) + b2
            h = F.elu(h)
            return torch.einsum('koi,ki->ko', Wh, h) + bh          # (K, out)
        else:
            h = torch.einsum('koi,kni->kno', W1, feat) + b1[:, None, :]
            h = F.elu(h)
            h = torch.einsum('koi,kni->kno', W2, h) + b2[:, None, :]
            h = F.elu(h)
            return torch.einsum('koi,kni->kno', Wh, h) + bh[:, None, :]  # (K, N, out)

    def _batched_mlp(self, feat):
        return self._run_batched(self.mlps, feat, 'net', 'head')

    def forward(
        self,
        xyz_canonical: torch.Tensor,
        normals: torch.Tensor,
        log_scales: torch.Tensor,
        rest_transforms: torch.Tensor,
        pose_transforms: torch.Tensor,
    ):
        """
        Args:
            xyz_canonical:    (bs, N, 3) Gaussian means in canonical space.
            normals:          (bs, N, 3) canonical-space normals.
            log_scales:       (bs, N, 3) log-scales.
            rest_transforms:  (bs, J, 4, 4) joint transforms at rest (big pose).
            pose_transforms:  (bs, J, 4, 4) joint transforms at the current pose.

        Returns:
            d_xyz:   (bs, N, 3) canonical-space mean correction.
            d_scale: (bs, N, 3) log-scale correction.
            d_rot:   (bs, N, 3) axis-angle rotation correction.
        """
        bs, N, _ = xyz_canonical.shape
        dev = xyz_canonical.device
        K = len(self.joint_indices)
        idx = self.joint_indices

        d_xyz = torch.zeros(bs, N, 3, device=dev)
        d_scale = torch.zeros(bs, N, 3, device=dev)
        d_rot = torch.zeros(bs, N, 3, device=dev)
        d_alpha = torch.zeros(bs, N, 1, device=dev)

        r = torch.tensor([self._radius(j) for j in idx], device=dev, dtype=xyz_canonical.dtype)  # (K,)

        for b in range(bs):
            R0 = rest_transforms[b][idx, :3, :3]   # (K, 3, 3)
            t0 = rest_transforms[b][idx, :3, 3]    # (K, 3)
            pose_feat = self._pose_feature_batch(idx, rest_transforms[b], pose_transforms[b])  # (K, 3)

            R0_T = R0.transpose(1, 2)
            xyz_b = xyz_canonical[b]               # (N, 3)

            # local coordinates: R0^T @ (xyz - t0)
            xyz_rel = xyz_b[None] - t0[:, None, :]                     # (K, N, 3)
            x_local = torch.einsum('kab,knb->kna', R0_T, xyz_rel)      # (K, N, 3)

            # distance falloff weight
            d = x_local.norm(dim=-1)                                   # (K, N)
            w = torch.exp(-(d / r[:, None]) ** 2) * (d < self.cutoff * r[:, None]).float()  # (K, N)

            # feature vector: apply PE to pose_feat and normalized local xyz
            xn = x_local / r[:, None, None]                            # (K, N, 3)
            pose_pe = self._pe(pose_feat)                              # (K, 3+PE)
            xyz_pe = self._pe(xn)                                      # (K, N, 3+PE)

            hf_parts = [xyz_pe]
            if self.use_normal:
                hf_parts.append(torch.einsum('kab,nb->kna', R0_T, normals[b]))
            if self.use_scale:
                hf_parts.append(log_scales[b][None].expand(K, N, 3))
            hf_feat = torch.cat(hf_parts, dim=-1)                      # (K, N, hf_dim)

            if not self.split:
                pose_broadcast = pose_pe[:, None, :].expand(K, N, pose_pe.shape[-1])
                feat = torch.cat([pose_broadcast, hf_feat], dim=-1)    # (K, N, in_dim)
                out = self._batched_mlp(feat)                          # (K, N, 10)
            else:
                out_lf = self._run_batched(self.mlps, pose_pe, 'lf_net', 'lf_head')   # (K, 10)
                out_hf = self._run_batched(self.mlps, hf_feat, 'hf_net', 'hf_head')   # (K, N, 10)
                out = out_lf[:, None, :] + out_hf                      # (K, N, 10)
            d_xyz_l, d_scale_j, d_rot_l, d_alpha_l = out.split([3, 3, 3, 1], dim=-1)
            # bound rotation residual to avoid long-axis flips at large bends
            d_rot_l = self.rot_clamp * torch.tanh(d_rot_l)
            # bound opacity residual in logit space (identity at init)
            d_alpha_l = self.alpha_clamp * torch.tanh(d_alpha_l)

            # local displacement -> canonical space. The rotation residual is a
            # so(3) vector; it transforms under the adjoint R0 (x) w_local, so the
            # same R0 maps it to the canonical frame (keeps d_rot consistent with
            # d_xyz, which apply_rotation_correction then composes in world frame).
            d_xyz_j = torch.einsum('kba,kna->knb', R0, d_xyz_l)        # (K, N, 3)
            d_rot_j = torch.einsum('kba,kna->knb', R0, d_rot_l)        # (K, N, 3)

            # convex blend across joints
            wsum = w.sum(dim=0, keepdim=True).clamp_min(1e-6)          # (1, N)
            d_xyz[b] = (w.unsqueeze(-1) * d_xyz_j).sum(dim=0) / wsum.transpose(0, 1)
            d_scale[b] = (w.unsqueeze(-1) * d_scale_j).sum(dim=0) / wsum.transpose(0, 1)
            d_rot[b] = (w.unsqueeze(-1) * d_rot_j).sum(dim=0) / wsum.transpose(0, 1)
            d_alpha[b] = (w.unsqueeze(-1) * d_alpha_l).sum(dim=0) / wsum.transpose(0, 1)

        return d_xyz, d_scale, d_rot, d_alpha


def pose_residual_loss(
    d_xyz: torch.Tensor,
    d_scale: torch.Tensor,
    d_rot: torch.Tensor,
    d_alpha: Optional[torch.Tensor] = None,
    w_xyz: float = 1.0,
    w_scale: float = 1.0,
    w_rot: float = 1.0,
    w_alpha: float = 1.0,
) -> torch.Tensor:
    """L1 residual loss L_res = sum_i (|d_mu_i|_1 + |d_s_i|_1 + |d_r_i|_1 [+ |d_a_i|_1])."""
    out = (
        w_xyz * d_xyz.abs().mean()
        + w_scale * d_scale.abs().mean()
        + w_rot * d_rot.abs().mean()
    )
    if d_alpha is not None:
        out = out + w_alpha * d_alpha.abs().mean()
    return out
