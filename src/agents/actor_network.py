"""
actor_network.py — Decentralised Actor (Policy) Network for MAPPO
===================================================================

This module implements the Actor network used in the Multi-Agent Proximal
Policy Optimization (MAPPO) algorithm with Centralised Training,
Decentralised Execution (CTDE).

Architecture
------------
During **decentralised execution**, each agent uses its own Actor to
select actions based on **local** observations only:

    π_θ(a_i | o_i) = Categorical(softmax(masked_logits))

Network layout::

    local_obs (obs_dim)
        ↓
    Linear(obs_dim, 256) → LayerNorm → ReLU
        ↓
    Linear(256, 128) → LayerNorm → ReLU
        ↓
    Linear(128, 64) → LayerNorm → ReLU
        ↓
    Linear(64, action_dim)   →  raw logits
        ↓
    apply action_mask (−1e9 for invalid)
        ↓
    Categorical(logits=masked_logits)

Key design choices
------------------
1. **Orthogonal initialisation** — Proven to improve gradient flow in
   deep RL (Andrychowicz et al., 2021).
2. **LayerNorm** — Stabilises activations across heterogeneous
   observation scales (vehicle states, road health, waste volumes).
3. **Action masking** — Hard constraint: invalid actions receive logit
   = −10⁹, yielding exact-zero probability after softmax.  This is
   critical for constraint satisfaction (e.g., no travel on destroyed
   roads, no pickup when full).
4. **Device-agnostic** — All tensors are created on the model's device;
   the caller should move observations to the same device.

References
----------
- Yu et al. (2022). "The Surprising Effectiveness of PPO in Cooperative
  Multi-Agent Games." NeurIPS.
- Schulman et al. (2017). "Proximal Policy Optimization Algorithms."

Author  : Muhammed Şara
License : MIT
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def _orthogonal_init(
    module: nn.Module,
    gain: float = 1.0,
) -> None:
    """Apply orthogonal initialisation to ``Linear`` layers.

    Orthogonal init preserves gradient norms across layers and has been
    shown to improve training stability in deep RL networks.

    Parameters
    ----------
    module : nn.Module
        The module to initialise (operates in-place).
    gain : float
        Scaling factor (√2 for ReLU layers, 0.01 for the output layer
        to encourage initial exploration).
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# Actor Network
# ---------------------------------------------------------------------------

class ActorNetwork(nn.Module):
    """Decentralised policy network for a single MAPPO agent.

    The actor maps a local observation vector to a probability
    distribution over discrete actions, respecting action masks.

    Parameters
    ----------
    obs_dim : int
        Dimension of the agent's local observation vector ``o_i``.
    action_dim : int
        Size of the discrete action space.
    hidden_sizes : Tuple[int, ...]
        Widths of hidden MLP layers.  Default ``(256, 128, 64)``.
    use_layer_norm : bool
        Whether to apply ``LayerNorm`` after each hidden layer.
    activation : str
        Activation function: ``"relu"`` or ``"tanh"``.

    Attributes
    ----------
    network : nn.Sequential
        The feature extraction MLP.
    action_head : nn.Linear
        Final projection to raw logits.

    Examples
    --------
    >>> actor = ActorNetwork(obs_dim=25, action_dim=9)
    >>> obs = torch.randn(4, 25)           # batch of 4
    >>> mask = torch.ones(4, 9)            # all actions valid
    >>> dist, logits = actor(obs, mask)
    >>> actions = dist.sample()
    """

    MASK_VALUE: float = -1e9
    """Large negative value applied to masked (invalid) logits."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 128, 64),
        use_layer_norm: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        self._obs_dim = obs_dim
        self._action_dim = action_dim

        # --- Select activation ---
        act_fn = nn.ReLU if activation == "relu" else nn.Tanh
        act_gain = (
            nn.init.calculate_gain("relu")
            if activation == "relu"
            else nn.init.calculate_gain("tanh")
        )

        # --- Build hidden MLP ---
        layers: list[nn.Module] = []
        prev_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(act_fn())
            prev_dim = h

        self.network = nn.Sequential(*layers)

        # --- Action logit head ---
        self.action_head = nn.Linear(prev_dim, action_dim)

        # --- Orthogonal initialisation ---
        # Hidden layers: gain = √2 for ReLU, 1.0 for tanh
        self.network.apply(lambda m: _orthogonal_init(m, gain=act_gain))
        # Output layer: small gain → initial near-uniform policy
        _orthogonal_init(self.action_head, gain=0.01)

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Tuple[Categorical, torch.Tensor]:
        """Forward pass: observation → masked action distribution.

        Parameters
        ----------
        obs : torch.Tensor
            Local observation, shape ``(batch, obs_dim)`` or ``(obs_dim,)``.
        action_mask : torch.Tensor
            Binary mask, shape ``(batch, action_dim)`` or ``(action_dim,)``.
            1 = valid action, 0 = invalid action.

        Returns
        -------
        dist : Categorical
            Action probability distribution (masked).
        logits : torch.Tensor
            Raw (pre-mask) logits for logging / debugging.

        Notes
        -----
        The masking procedure:

        .. math::

            \\text{logit}_a^{\\text{masked}} =
            \\begin{cases}
                \\text{logit}_a       & \\text{if } m_a = 1 \\\\
                -10^9                & \\text{if } m_a = 0
            \\end{cases}

        This guarantees that softmax(masked_logits) assigns exactly 0
        probability to invalid actions, which is essential for
        constraint-safe exploration.
        """
        # Handle unbatched input
        squeeze = False
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)
            squeeze = True

        # Feature extraction
        features = self.network(obs)

        # Raw logits
        logits = self.action_head(features)

        # ---- Action masking (CRITICAL) ----
        # Where mask == 0, set logit to −1e9 → softmax → prob ≈ 0
        masked_logits = logits.clone()
        masked_logits[action_mask == 0] = self.MASK_VALUE

        # Build categorical distribution from masked logits
        dist = Categorical(logits=masked_logits)

        if squeeze:
            logits = logits.squeeze(0)

        return dist, logits

    def get_action(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action and return log-prob + entropy.

        This is the primary interface used during rollout collection.

        Parameters
        ----------
        obs : torch.Tensor
            Local observation.
        action_mask : torch.Tensor
            Binary validity mask.
        deterministic : bool
            If True, select the highest-probability action (argmax).
            Used during evaluation, not training.

        Returns
        -------
        action : torch.Tensor
            Sampled (or argmax) action index.
        log_prob : torch.Tensor
            Log-probability of the selected action.
        entropy : torch.Tensor
            Entropy of the distribution (for the entropy bonus term).
        """
        dist, _ = self.forward(obs, action_mask)

        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log-probs and entropy for given (obs, action) pairs.

        Used during the PPO update step to compute the importance ratio
        π_θ(a|o) / π_θ_old(a|o).

        Parameters
        ----------
        obs : torch.Tensor
            Batch of observations, shape ``(batch, obs_dim)``.
        action_mask : torch.Tensor
            Batch of masks, shape ``(batch, action_dim)``.
        actions : torch.Tensor
            Batch of previously taken actions, shape ``(batch,)``.

        Returns
        -------
        log_probs : torch.Tensor
            Log π_θ(a|o) for each (obs, action) pair.
        entropy : torch.Tensor
            Per-sample entropy.
        """
        dist, _ = self.forward(obs, action_mask)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"ActorNetwork(obs={self._obs_dim}, act={self._action_dim}, "
            f"params={self.num_parameters:,})"
        )
