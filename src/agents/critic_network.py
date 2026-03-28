"""
critic_network.py — Centralised Critic (Value) Network for MAPPO
==================================================================

This module implements the Critic network used in the Multi-Agent
Proximal Policy Optimization (MAPPO) algorithm with Centralised
Training, Decentralised Execution (CTDE).

Architecture
------------
During **centralised training**, the critic receives the **global
state** vector (all agents' observations + environment state) to
estimate the state value function:

    V_φ(s) = ℝ  (scalar value estimate)

This follows the CTDE paradigm where the critic has access to
privileged global information during training but is not needed
during deployment (decentralised execution uses actors only).

Network layout::

    global_state (state_dim)
        ↓
    Linear(state_dim, 512) → LayerNorm → ReLU
        ↓
    Linear(512, 256) → LayerNorm → ReLU
        ↓
    Linear(256, 128) → LayerNorm → ReLU
        ↓
    Linear(128, 1)   →  V(s)  (scalar)

The critic is wider than the actor because the global state
vector is much larger (all edge healths + all vehicle states +
all waste storages + facility capacities + time).

Key design choices
------------------
1. **Orthogonal initialisation** — Same rationale as the actor;
   the value head uses gain=1.0 for stable initial predictions.
2. **LayerNorm** — Particularly important for the critic because
   the global state concatenates heterogeneous features (edge health
   in [0,1], waste tonnes in [0, 10000+], normalised time in [0,1]).
3. **Single scalar output** — Standard state-value function V(s).
   No advantage decomposition here; that is computed in the PPO
   update (A = R − V).
4. **Device-agnostic** — Works on CPU or CUDA.

References
----------
- Yu et al. (2022). "The Surprising Effectiveness of PPO in Cooperative
  Multi-Agent Games." NeurIPS.
- de Witt et al. (2020). "Is Independent Learning All You Need in the
  StarCraft Multi-Agent Challenge?" arXiv:2011.09533.

Author  : Muhammed Şara
License : MIT
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Initialisation helper (shared logic with actor)
# ---------------------------------------------------------------------------

def _orthogonal_init(
    module: nn.Module,
    gain: float = 1.0,
) -> None:
    """Apply orthogonal initialisation to ``Linear`` layers.

    Parameters
    ----------
    module : nn.Module
        Module to initialise in-place.
    gain : float
        Scaling factor.
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# Critic Network
# ---------------------------------------------------------------------------

class CriticNetwork(nn.Module):
    """Centralised value network for MAPPO (CTDE).

    The critic takes the global state vector as input and outputs a
    single scalar value V(s), estimating the expected discounted
    return from the current state.

    Because all agents share the same global state, a **single**
    critic is typically shared across agents (parameter sharing).

    Parameters
    ----------
    state_dim : int
        Dimension of the global state vector.
    hidden_sizes : Tuple[int, ...]
        Widths of hidden MLP layers.  Default ``(512, 256, 128)``.
    use_layer_norm : bool
        Whether to apply ``LayerNorm`` after each hidden layer.
    activation : str
        Activation function: ``"relu"`` or ``"tanh"``.

    Attributes
    ----------
    network : nn.Sequential
        Feature extraction MLP.
    value_head : nn.Linear
        Final projection to scalar V(s).

    Examples
    --------
    >>> critic = CriticNetwork(state_dim=122)
    >>> state = torch.randn(4, 122)       # batch of 4
    >>> values = critic(state)             # shape (4, 1)
    """

    def __init__(
        self,
        state_dim: int,
        hidden_sizes: Tuple[int, ...] = (512, 256, 128),
        use_layer_norm: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        self._state_dim = state_dim

        # --- Select activation ---
        act_fn = nn.ReLU if activation == "relu" else nn.Tanh
        act_gain = (
            nn.init.calculate_gain("relu")
            if activation == "relu"
            else nn.init.calculate_gain("tanh")
        )

        # --- Build hidden MLP ---
        layers: list[nn.Module] = []
        prev_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(act_fn())
            prev_dim = h

        self.network = nn.Sequential(*layers)

        # --- Value head: single scalar output ---
        self.value_head = nn.Linear(prev_dim, 1)

        # --- Orthogonal initialisation ---
        self.network.apply(lambda m: _orthogonal_init(m, gain=act_gain))
        # Value head: gain=1.0 for stable initial value estimates
        _orthogonal_init(self.value_head, gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass: global state → value estimate.

        Parameters
        ----------
        state : torch.Tensor
            Global state vector, shape ``(batch, state_dim)`` or
            ``(state_dim,)``.

        Returns
        -------
        torch.Tensor
            Value estimates, shape ``(batch, 1)`` or ``(1,)``.

        Notes
        -----
        The value function:

        .. math::

            V_\\phi(s) = W_{\\text{out}} \\cdot
            \\text{ReLU}(\\text{LN}(W_3 \\cdot
            \\text{ReLU}(\\text{LN}(W_2 \\cdot
            \\text{ReLU}(\\text{LN}(W_1 \\cdot s + b_1))
            + b_2)) + b_3)) + b_{\\text{out}}
        """
        squeeze = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze = True

        features = self.network(state)
        value = self.value_head(features)

        if squeeze:
            value = value.squeeze(0)

        return value

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper: return V(s) as a flat tensor.

        Parameters
        ----------
        state : torch.Tensor
            Global state, shape ``(batch, state_dim)``.

        Returns
        -------
        torch.Tensor
            Values, shape ``(batch,)`` — squeezed last dimension.
        """
        return self.forward(state).squeeze(-1)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"CriticNetwork(state={self._state_dim}, "
            f"params={self.num_parameters:,})"
        )
