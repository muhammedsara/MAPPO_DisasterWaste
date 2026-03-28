"""
buffer.py — Rollout Buffer with Generalised Advantage Estimation (GAE)
========================================================================

This module implements the experience buffer for on-policy PPO training.
It stores transitions ``(obs, action, reward, value, log_prob, mask,
global_state)`` collected during environment rollouts and computes
GAE-based advantages for the policy gradient update.

Generalised Advantage Estimation (Schulman et al., 2016)
--------------------------------------------------------
For each time step *t* in the rollout of length *T*:

.. math::

    \\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)

    \\hat{A}_t = \\sum_{l=0}^{T-t-1} (\\gamma \\lambda)^l \\delta_{t+l}

    R_t = \\hat{A}_t + V(s_t)

where γ is the discount factor and λ is the GAE smoothing parameter.

- λ = 0 → TD(0) advantage (high bias, low variance)
- λ = 1 → Monte-Carlo advantage (low bias, high variance)
- λ ≈ 0.95 → typical PPO setting

This buffer stores data for **all agents** simultaneously, following
the MAPPO convention where each agent has its own trajectory but the
buffer processes them together for vectorised computation.

Author  : Muhammed Şara
License : MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Transition data class
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """Single time-step transition for one agent.

    Attributes
    ----------
    obs : np.ndarray
        Local observation, shape ``(obs_dim,)``.
    global_state : np.ndarray
        Global state for the centralised critic, shape ``(state_dim,)``.
    action : int
        Discrete action taken.
    reward : float
        Reward received.
    value : float
        Critic value estimate V(s).
    log_prob : float
        Log-probability of the action under the policy.
    action_mask : np.ndarray
        Binary action validity mask, shape ``(action_dim,)``.
    done : bool
        Whether the episode terminated/truncated after this step.
    """
    obs: np.ndarray
    global_state: np.ndarray
    action: int
    reward: float
    value: float
    log_prob: float
    action_mask: np.ndarray
    done: bool


# ---------------------------------------------------------------------------
# Mini-batch data class
# ---------------------------------------------------------------------------

@dataclass
class MiniBatch:
    """Batched tensors ready for the PPO update step.

    All tensors have shape ``(batch_size, *)`` and reside on the
    specified device.
    """
    obs: torch.Tensor            # (B, obs_dim)
    global_state: torch.Tensor   # (B, state_dim)
    actions: torch.Tensor        # (B,) long
    old_log_probs: torch.Tensor  # (B,)
    advantages: torch.Tensor     # (B,)
    returns: torch.Tensor        # (B,)
    action_masks: torch.Tensor   # (B, action_dim)
    old_values: torch.Tensor     # (B,)


# ---------------------------------------------------------------------------
# Rollout Buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """On-policy rollout buffer with GAE for multi-agent PPO.

    The buffer collects transitions for ``n_agents`` agents over
    ``rollout_length`` environment steps, then computes GAE advantages
    and discounted returns for the policy update.

    Parameters
    ----------
    n_agents : int
        Number of agents in the environment.
    obs_dim : int
        Dimension of each agent's local observation.
    state_dim : int
        Dimension of the global state vector (for the critic).
    action_dim : int
        Size of the discrete action space (for masks).
    rollout_length : int
        Number of environment steps per rollout before an update.
    gamma : float
        Discount factor γ ∈ [0, 1].
    gae_lambda : float
        GAE smoothing parameter λ ∈ [0, 1].
    device : torch.device
        Target device for output tensors.

    Examples
    --------
    >>> buf = RolloutBuffer(n_agents=4, obs_dim=25, state_dim=122,
    ...                     action_dim=9, rollout_length=128)
    >>> for t in range(128):
    ...     buf.store(agent_idx=0, transition=...)
    >>> buf.compute_returns_and_advantages(last_values)
    >>> for batch in buf.generate_batches(mini_batch_size=64):
    ...     # PPO update
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        rollout_length: int = 128,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self._n_agents = n_agents
        self._obs_dim = obs_dim
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._rollout_length = rollout_length
        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self._device = device

        # Pre-allocate numpy arrays: (rollout_length, n_agents, *)
        self._obs = np.zeros(
            (rollout_length, n_agents, obs_dim), dtype=np.float32
        )
        self._global_states = np.zeros(
            (rollout_length, n_agents, state_dim), dtype=np.float32
        )
        self._actions = np.zeros(
            (rollout_length, n_agents), dtype=np.int64
        )
        self._rewards = np.zeros(
            (rollout_length, n_agents), dtype=np.float32
        )
        self._values = np.zeros(
            (rollout_length, n_agents), dtype=np.float32
        )
        self._log_probs = np.zeros(
            (rollout_length, n_agents), dtype=np.float32
        )
        self._action_masks = np.zeros(
            (rollout_length, n_agents, action_dim), dtype=np.float32
        )
        self._dones = np.zeros(
            (rollout_length, n_agents), dtype=np.float32
        )

        # GAE outputs (computed after rollout)
        self._advantages = np.zeros(
            (rollout_length, n_agents), dtype=np.float32
        )
        self._returns = np.zeros(
            (rollout_length, n_agents), dtype=np.float32
        )

        self._step = 0
        self._ready = False

    # ==================================================================
    # Data collection
    # ==================================================================

    def store(
        self,
        step: int,
        agent_idx: int,
        transition: Transition,
    ) -> None:
        """Store a single transition at the given (step, agent) position.

        Parameters
        ----------
        step : int
            Time step index within the rollout [0, rollout_length).
        agent_idx : int
            Agent index [0, n_agents).
        transition : Transition
            The transition data to store.
        """
        self._obs[step, agent_idx] = transition.obs
        self._global_states[step, agent_idx] = transition.global_state
        self._actions[step, agent_idx] = transition.action
        self._rewards[step, agent_idx] = transition.reward
        self._values[step, agent_idx] = transition.value
        self._log_probs[step, agent_idx] = transition.log_prob
        self._action_masks[step, agent_idx] = transition.action_mask
        self._dones[step, agent_idx] = float(transition.done)

    def store_step(
        self,
        step: int,
        obs_all: np.ndarray,
        states_all: np.ndarray,
        actions_all: np.ndarray,
        rewards_all: np.ndarray,
        values_all: np.ndarray,
        log_probs_all: np.ndarray,
        masks_all: np.ndarray,
        dones_all: np.ndarray,
    ) -> None:
        """Store all agents' data for a single time step at once.

        More efficient than calling ``store()`` per-agent when data
        is already batch-formatted.

        Parameters
        ----------
        step : int
            Time step index.
        obs_all : np.ndarray, shape (n_agents, obs_dim)
        states_all : np.ndarray, shape (n_agents, state_dim)
        actions_all : np.ndarray, shape (n_agents,)
        rewards_all : np.ndarray, shape (n_agents,)
        values_all : np.ndarray, shape (n_agents,)
        log_probs_all : np.ndarray, shape (n_agents,)
        masks_all : np.ndarray, shape (n_agents, action_dim)
        dones_all : np.ndarray, shape (n_agents,)
        """
        self._obs[step] = obs_all
        self._global_states[step] = states_all
        self._actions[step] = actions_all
        self._rewards[step] = rewards_all
        self._values[step] = values_all
        self._log_probs[step] = log_probs_all
        self._action_masks[step] = masks_all
        self._dones[step] = dones_all

    # ==================================================================
    # GAE computation
    # ==================================================================

    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,
        last_dones: np.ndarray,
    ) -> None:
        """Compute GAE advantages and discounted returns.

        Must be called after the rollout is complete, before
        ``generate_batches()``.

        The algorithm implements the recursive GAE formula:

        .. math::

            \\delta_t = r_t + \\gamma (1 - d_{t+1}) V(s_{t+1}) - V(s_t)

            \\hat{A}_t = \\delta_t + \\gamma \\lambda (1 - d_{t+1}) \\hat{A}_{t+1}

            R_t = \\hat{A}_t + V(s_t)

        Parameters
        ----------
        last_values : np.ndarray, shape (n_agents,)
            Critic value estimates for the state **after** the last
            rollout step (bootstrap value).
        last_dones : np.ndarray, shape (n_agents,)
            Whether each agent's episode ended at the last step.
        """
        gae = np.zeros(self._n_agents, dtype=np.float32)
        T = self._rollout_length

        for t in reversed(range(T)):
            if t == T - 1:
                next_values = last_values
                next_dones = last_dones
            else:
                next_values = self._values[t + 1]
                next_dones = self._dones[t + 1]

            # TD residual: δ_t = r_t + γ(1-d_{t+1})V(s_{t+1}) - V(s_t)
            delta = (
                self._rewards[t]
                + self._gamma * (1.0 - next_dones) * next_values
                - self._values[t]
            )

            # GAE recursion: Â_t = δ_t + γλ(1-d_{t+1})Â_{t+1}
            gae = delta + self._gamma * self._gae_lambda * (1.0 - next_dones) * gae

            self._advantages[t] = gae

        # Returns = advantages + values
        self._returns = self._advantages + self._values
        self._ready = True

    # ==================================================================
    # Mini-batch generation
    # ==================================================================

    def generate_batches(
        self,
        mini_batch_size: int = 64,
        shuffle: bool = True,
    ) -> Generator[MiniBatch, None, None]:
        """Yield mini-batches for the PPO update.

        Flattens the (rollout_length × n_agents) buffer into a single
        dataset, optionally shuffles, and yields ``MiniBatch`` objects.

        Parameters
        ----------
        mini_batch_size : int
            Number of samples per mini-batch.
        shuffle : bool
            Whether to shuffle the dataset before batching.

        Yields
        ------
        MiniBatch
            Batched tensors on ``self._device``.
        """
        if not self._ready:
            raise RuntimeError(
                "Call compute_returns_and_advantages() before generate_batches()."
            )

        total = self._rollout_length * self._n_agents

        # Flatten: (T, N, *) → (T*N, *)
        obs_flat = self._obs.reshape(total, self._obs_dim)
        states_flat = self._global_states.reshape(total, self._state_dim)
        actions_flat = self._actions.reshape(total)
        log_probs_flat = self._log_probs.reshape(total)
        advantages_flat = self._advantages.reshape(total)
        returns_flat = self._returns.reshape(total)
        masks_flat = self._action_masks.reshape(total, self._action_dim)
        values_flat = self._values.reshape(total)

        # Normalise advantages (zero mean, unit variance)
        adv_mean = advantages_flat.mean()
        adv_std = advantages_flat.std() + 1e-8
        advantages_flat = (advantages_flat - adv_mean) / adv_std

        # Generate indices
        indices = np.arange(total)
        if shuffle:
            np.random.shuffle(indices)

        # Yield mini-batches
        for start in range(0, total, mini_batch_size):
            end = min(start + mini_batch_size, total)
            idx = indices[start:end]

            yield MiniBatch(
                obs=torch.tensor(
                    obs_flat[idx], dtype=torch.float32, device=self._device
                ),
                global_state=torch.tensor(
                    states_flat[idx], dtype=torch.float32, device=self._device
                ),
                actions=torch.tensor(
                    actions_flat[idx], dtype=torch.long, device=self._device
                ),
                old_log_probs=torch.tensor(
                    log_probs_flat[idx], dtype=torch.float32, device=self._device
                ),
                advantages=torch.tensor(
                    advantages_flat[idx], dtype=torch.float32, device=self._device
                ),
                returns=torch.tensor(
                    returns_flat[idx], dtype=torch.float32, device=self._device
                ),
                action_masks=torch.tensor(
                    masks_flat[idx], dtype=torch.float32, device=self._device
                ),
                old_values=torch.tensor(
                    values_flat[idx], dtype=torch.float32, device=self._device
                ),
            )

    # ==================================================================
    # Utility
    # ==================================================================

    def reset(self) -> None:
        """Clear the buffer for a new rollout."""
        self._obs[:] = 0
        self._global_states[:] = 0
        self._actions[:] = 0
        self._rewards[:] = 0
        self._values[:] = 0
        self._log_probs[:] = 0
        self._action_masks[:] = 0
        self._dones[:] = 0
        self._advantages[:] = 0
        self._returns[:] = 0
        self._step = 0
        self._ready = False

    @property
    def total_samples(self) -> int:
        return self._rollout_length * self._n_agents

    @property
    def is_ready(self) -> bool:
        return self._ready

    def get_episode_statistics(self) -> Dict[str, float]:
        """Return summary statistics of the stored rollout."""
        return {
            "mean_reward": float(self._rewards.mean()),
            "std_reward": float(self._rewards.std()),
            "mean_value": float(self._values.mean()),
            "mean_advantage": float(self._advantages.mean()),
            "mean_return": float(self._returns.mean()),
        }

    def __repr__(self) -> str:
        return (
            f"RolloutBuffer(agents={self._n_agents}, "
            f"rollout={self._rollout_length}, "
            f"total={self.total_samples}, "
            f"ready={self._ready})"
        )
