"""
mappo.py — Multi-Agent Proximal Policy Optimization (MAPPO)
=============================================================

This module implements the MAPPO algorithm with Centralised Training,
Decentralised Execution (CTDE) for disaster waste management.

Algorithm overview (per update cycle)
--------------------------------------
1. Collect rollout of length T using current policy π_θ_old.
2. Compute GAE advantages Â_t and returns R_t.
3. For K epochs, for each mini-batch B of transitions:

   (a) **Actor loss** — Clipped surrogate objective:

       .. math::

           r_t(\\theta) = \\frac{\\pi_\\theta(a_t | o_t)}{\\pi_{\\theta_{old}}(a_t | o_t)}

           L^{CLIP} = -\\mathbb{E}_t\\Big[
               \\min\\big(r_t \\hat{A}_t,\\;
               \\text{clip}(r_t, 1-\\varepsilon, 1+\\varepsilon)\\hat{A}_t \\big)
           \\Big]

   (b) **Critic loss** — Clipped value function MSE:

       .. math::

           L^{VF} = \\mathbb{E}_t\\Big[
               \\max\\big(
                   (V_\\phi(s_t) - R_t)^2,\\;
                   (\\text{clip}(V_\\phi(s_t), V_{old} \\pm \\varepsilon_v) - R_t)^2
               \\big)
           \\Big]

   (c) **Entropy bonus** — Encourage exploration:

       .. math::

           L^{ENT} = -c_{ent} \\cdot \\mathbb{E}_t[H(\\pi_\\theta(\\cdot | o_t))]

   (d) **Total loss** = L^{CLIP} + c_v · L^{VF} + L^{ENT}

4. Update θ, φ via Adam.

References
----------
- Yu et al. (2022). "The Surprising Effectiveness of PPO in Cooperative
  Multi-Agent Games." NeurIPS.
- Schulman et al. (2017). "Proximal Policy Optimization Algorithms."
- Schulman et al. (2016). "High-Dimensional Continuous Control Using
  Generalized Advantage Estimation." ICLR.

Author  : Muhammed Şara
License : MIT
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .actor_network import ActorNetwork
from .critic_network import CriticNetwork
from .buffer import MiniBatch, RolloutBuffer, Transition


# ---------------------------------------------------------------------------
# Hyper-parameter configuration
# ---------------------------------------------------------------------------

@dataclass
class MAPPOConfig:
    """Full hyper-parameter set for MAPPO training.

    Default values follow Yu et al. (2022) recommendations for
    cooperative multi-agent tasks.

    Parameters
    ----------
    lr_actor : float
        Actor learning rate.
    lr_critic : float
        Critic learning rate (typically 2-5× actor LR).
    gamma : float
        Discount factor γ.
    gae_lambda : float
        GAE smoothing parameter λ.
    clip_ratio : float
        PPO clipping ε for the surrogate objective.
    clip_value : float
        Value function clipping ε_v.  Set to 0 to disable.
    entropy_coef : float
        Entropy bonus coefficient c_ent.
    value_coef : float
        Value loss coefficient c_v.
    max_grad_norm : float
        Gradient clipping (L2 norm).
    n_epochs : int
        Number of PPO update epochs per rollout.
    mini_batch_size : int
        Mini-batch size for SGD.
    rollout_length : int
        Environment steps per rollout collection.
    total_timesteps : int
        Total training budget (environment steps).
    eval_interval : int
        Evaluate every N rollouts.
    save_interval : int
        Save model checkpoint every N rollouts.
    log_interval : int
        Log training metrics every N rollouts.

    actor_hidden : Tuple[int, ...]
        Actor network hidden layer sizes.
    critic_hidden : Tuple[int, ...]
        Critic network hidden layer sizes.
    use_layer_norm : bool
        Use LayerNorm in both networks.
    activation : str
        Activation function ("relu" or "tanh").

    use_linear_lr_decay : bool
        Whether to linearly decay learning rates to 0 over training.
    use_value_clip : bool
        Whether to clip value function updates.
    use_huber_loss : bool
        Use Huber loss instead of MSE for critic.
    huber_delta : float
        Huber loss δ.
    """
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    clip_value: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    mini_batch_size: int = 64
    rollout_length: int = 128
    total_timesteps: int = 1_000_000
    eval_interval: int = 10
    save_interval: int = 50
    log_interval: int = 5

    actor_hidden: Tuple[int, ...] = (256, 128, 64)
    critic_hidden: Tuple[int, ...] = (512, 256, 128)
    use_layer_norm: bool = True
    activation: str = "relu"

    use_linear_lr_decay: bool = True
    use_value_clip: bool = True
    use_huber_loss: bool = False
    huber_delta: float = 10.0


# ---------------------------------------------------------------------------
# Update statistics
# ---------------------------------------------------------------------------

@dataclass
class UpdateStats:
    """Metrics from a single PPO update cycle."""
    actor_loss: float = 0.0
    critic_loss: float = 0.0
    entropy: float = 0.0
    total_loss: float = 0.0
    explained_variance: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    mean_advantage: float = 0.0
    mean_return: float = 0.0
    actor_lr: float = 0.0
    critic_lr: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# MAPPO Algorithm
# ---------------------------------------------------------------------------

class MAPPO:
    """Multi-Agent Proximal Policy Optimization with CTDE.

    This class manages:
        - Shared Actor and centralised Critic networks
        - Rollout buffer with GAE
        - PPO update with clipped surrogate + value clip + entropy bonus
        - Learning rate scheduling
        - TensorBoard logging
        - Model checkpointing

    Parameters
    ----------
    n_agents : int
        Number of agents.
    obs_dim : int
        Agent's local observation dimension.
    state_dim : int
        Global state dimension (for the critic).
    action_dim : int
        Discrete action space size.
    config : MAPPOConfig, optional
        Hyper-parameter configuration.
    log_dir : str, optional
        Directory for TensorBoard logs.
    device : torch.device, optional
        Computation device.

    Examples
    --------
    >>> mappo = MAPPO(n_agents=4, obs_dim=25, state_dim=122, action_dim=9)
    >>> stats = mappo.update(buffer)
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        config: Optional[MAPPOConfig] = None,
        log_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self._config = config or MAPPOConfig()
        self._n_agents = n_agents
        self._obs_dim = obs_dim
        self._state_dim = state_dim
        self._action_dim = action_dim

        # Device selection
        if device is not None:
            self._device = device
        else:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        # --- Build networks ---
        self.actor = ActorNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=self._config.actor_hidden,
            use_layer_norm=self._config.use_layer_norm,
            activation=self._config.activation,
        ).to(self._device)

        self.critic = CriticNetwork(
            state_dim=state_dim,
            hidden_sizes=self._config.critic_hidden,
            use_layer_norm=self._config.use_layer_norm,
            activation=self._config.activation,
        ).to(self._device)

        # --- Optimisers ---
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self._config.lr_actor, eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self._config.lr_critic, eps=1e-5
        )

        # --- Rollout buffer ---
        self.buffer = RolloutBuffer(
            n_agents=n_agents,
            obs_dim=obs_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            rollout_length=self._config.rollout_length,
            gamma=self._config.gamma,
            gae_lambda=self._config.gae_lambda,
            device=self._device,
        )

        # --- TensorBoard ---
        self._writer: Optional[SummaryWriter] = None
        if log_dir is not None:
            self._writer = SummaryWriter(log_dir=log_dir)

        # --- Training counters ---
        self._total_steps: int = 0
        self._update_count: int = 0

    # ==================================================================
    # Core PPO update
    # ==================================================================

    def update(self, buffer: Optional[RolloutBuffer] = None) -> UpdateStats:
        """Perform one PPO update cycle over a completed rollout.

        Executes K epochs of mini-batch SGD with the clipped surrogate
        objective, value-clipped critic loss, and entropy bonus.

        Parameters
        ----------
        buffer : RolloutBuffer, optional
            If ``None``, uses ``self.buffer``.

        Returns
        -------
        UpdateStats
            Aggregate training metrics for this update.
        """
        buf = buffer or self.buffer
        cfg = self._config

        if not buf.is_ready:
            raise RuntimeError("Buffer not ready: call compute_returns_and_advantages first.")

        # Accumulators for logging
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_frac = 0.0
        n_updates = 0

        for _epoch in range(cfg.n_epochs):
            for batch in buf.generate_batches(cfg.mini_batch_size):
                actor_loss, critic_loss, entropy, approx_kl, clip_frac = (
                    self._compute_losses(batch)
                )

                # --- Total loss ---
                total_loss = (
                    actor_loss
                    + cfg.value_coef * critic_loss
                    - cfg.entropy_coef * entropy
                )

                # --- Actor update ---
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), cfg.max_grad_norm
                )
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), cfg.max_grad_norm
                )

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # Accumulate metrics
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                total_approx_kl += approx_kl.item()
                total_clip_frac += clip_frac.item()
                n_updates += 1

        self._update_count += 1
        n = max(n_updates, 1)

        # Explained variance
        buf_stats = buf.get_episode_statistics()

        stats = UpdateStats(
            actor_loss=total_actor_loss / n,
            critic_loss=total_critic_loss / n,
            entropy=total_entropy / n,
            total_loss=(total_actor_loss + cfg.value_coef * total_critic_loss) / n,
            approx_kl=total_approx_kl / n,
            clip_fraction=total_clip_frac / n,
            mean_advantage=buf_stats["mean_advantage"],
            mean_return=buf_stats["mean_return"],
            actor_lr=self.actor_optimizer.param_groups[0]["lr"],
            critic_lr=self.critic_optimizer.param_groups[0]["lr"],
        )

        # TensorBoard logging
        if self._writer is not None:
            step = self._update_count
            for key, val in stats.to_dict().items():
                self._writer.add_scalar(f"train/{key}", val, step)

        return stats

    def _compute_losses(
        self, batch: MiniBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute actor, critic, and entropy losses for a mini-batch.

        Parameters
        ----------
        batch : MiniBatch
            Tensors from the rollout buffer.

        Returns
        -------
        actor_loss : torch.Tensor
            Clipped surrogate objective (scalar).
        critic_loss : torch.Tensor
            Value function loss (scalar).
        entropy : torch.Tensor
            Mean policy entropy (scalar).
        approx_kl : torch.Tensor
            Approximate KL divergence (for monitoring).
        clip_fraction : torch.Tensor
            Fraction of samples that were clipped.
        """
        cfg = self._config

        # ---- Actor: evaluate actions under current policy ----
        new_log_probs, entropies = self.actor.evaluate_actions(
            batch.obs, batch.action_masks, batch.actions
        )

        # ---- Importance ratio ----
        # r_t(θ) = π_θ(a|o) / π_θ_old(a|o) = exp(log π_θ − log π_old)
        log_ratio = new_log_probs - batch.old_log_probs
        ratio = torch.exp(log_ratio)

        # ---- Approximate KL (for diagnostics) ----
        # KL ≈ (r − 1) − log(r)  [from Schulman's approximation]
        approx_kl = ((ratio - 1.0) - log_ratio).mean()

        # ---- Clipped surrogate objective ----
        # L1 = r_t · Â_t
        surr1 = ratio * batch.advantages
        # L2 = clip(r_t, 1−ε, 1+ε) · Â_t
        surr2 = (
            torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
            * batch.advantages
        )
        actor_loss = -torch.min(surr1, surr2).mean()

        # Clip fraction (monitoring)
        clip_fraction = (
            (torch.abs(ratio - 1.0) > cfg.clip_ratio).float().mean()
        )

        # ---- Critic: value function loss ----
        new_values = self.critic.get_value(batch.global_state)

        if cfg.use_value_clip and cfg.clip_value > 0:
            # Clipped value loss (prevents large value function updates)
            v_clipped = batch.old_values + torch.clamp(
                new_values - batch.old_values,
                -cfg.clip_value,
                cfg.clip_value,
            )
            vf_loss1 = (new_values - batch.returns) ** 2
            vf_loss2 = (v_clipped - batch.returns) ** 2
            critic_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
        else:
            if cfg.use_huber_loss:
                critic_loss = nn.functional.huber_loss(
                    new_values, batch.returns, delta=cfg.huber_delta
                )
            else:
                critic_loss = 0.5 * ((new_values - batch.returns) ** 2).mean()

        # ---- Entropy ----
        entropy = entropies.mean()

        return actor_loss, critic_loss, entropy, approx_kl, clip_fraction

    # ==================================================================
    # Rollout collection
    # ==================================================================

    def collect_rollout(self, env) -> Dict[str, float]:
        """Collect a full rollout from the environment.

        Runs the policy in the environment for ``rollout_length`` steps,
        stores transitions in the buffer, and computes GAE.

        Parameters
        ----------
        env : DisasterWasteEnv
            PettingZoo ParallelEnv instance.

        Returns
        -------
        dict
            Rollout summary statistics.
        """
        cfg = self._config
        self.buffer.reset()

        # Get initial observations
        obs_dict, infos = env.reset()
        agent_list = env.possible_agents

        total_rewards = np.zeros(self._n_agents, dtype=np.float32)

        for t in range(cfg.rollout_length):
            # --- Gather observations and states ---
            obs_np = np.array([
                obs_dict[agent]["obs"] for agent in agent_list
            ], dtype=np.float32)
            states_np = np.array([
                obs_dict[agent]["global_state"] for agent in agent_list
            ], dtype=np.float32)
            masks_np = np.array([
                obs_dict[agent]["action_mask"] for agent in agent_list
            ], dtype=np.float32)

            # --- Select actions ---
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self._device)
            masks_t = torch.tensor(masks_np, dtype=torch.float32, device=self._device)
            states_t = torch.tensor(states_np, dtype=torch.float32, device=self._device)

            with torch.no_grad():
                actions_t, log_probs_t, _ = self.actor.get_action(obs_t, masks_t)
                values_t = self.critic.get_value(states_t)

            actions_np = actions_t.cpu().numpy()
            log_probs_np = log_probs_t.cpu().numpy()
            values_np = values_t.cpu().numpy()

            # --- Environment step ---
            action_dict = {
                agent: int(actions_np[i]) for i, agent in enumerate(agent_list)
            }
            next_obs_dict, rewards_dict, terms, truncs, infos = env.step(action_dict)

            rewards_np = np.array([
                rewards_dict[agent] for agent in agent_list
            ], dtype=np.float32)
            dones_np = np.array([
                float(terms[agent] or truncs[agent]) for agent in agent_list
            ], dtype=np.float32)

            # --- Store in buffer ---
            self.buffer.store_step(
                step=t,
                obs_all=obs_np,
                states_all=states_np,
                actions_all=actions_np,
                rewards_all=rewards_np,
                values_all=values_np,
                log_probs_all=log_probs_np,
                masks_all=masks_np,
                dones_all=dones_np,
            )

            total_rewards += rewards_np
            obs_dict = next_obs_dict
            self._total_steps += self._n_agents

            # Handle episode end (reset if truncated)
            if any(truncs.values()) or any(terms.values()):
                obs_dict, infos = env.reset()

        # --- Bootstrap value for GAE ---
        last_obs_np = np.array([
            obs_dict[agent]["global_state"] for agent in agent_list
        ], dtype=np.float32)
        last_states_t = torch.tensor(
            last_obs_np, dtype=torch.float32, device=self._device
        )
        with torch.no_grad():
            last_values = self.critic.get_value(last_states_t).cpu().numpy()

        last_dones = np.zeros(self._n_agents, dtype=np.float32)

        # --- Compute GAE ---
        self.buffer.compute_returns_and_advantages(last_values, last_dones)

        return {
            "mean_episode_reward": float(total_rewards.mean()),
            "total_steps": self._total_steps,
            "rollout_length": cfg.rollout_length,
        }

    # ==================================================================
    # Training loop
    # ==================================================================

    def train(
        self,
        env,
        total_timesteps: Optional[int] = None,
        callback: Optional[Any] = None,
    ) -> Dict[str, List[float]]:
        """Run the full MAPPO training loop.

        Parameters
        ----------
        env : DisasterWasteEnv
            Training environment.
        total_timesteps : int, optional
            Override ``config.total_timesteps``.
        callback : callable, optional
            Called with ``(mappo, update_count, stats)`` after each update.

        Returns
        -------
        dict
            Training history (per-update metrics).
        """
        budget = total_timesteps or self._config.total_timesteps
        cfg = self._config
        steps_per_rollout = cfg.rollout_length * self._n_agents

        history: Dict[str, List[float]] = {
            "actor_loss": [], "critic_loss": [], "entropy": [],
            "mean_reward": [], "approx_kl": [], "clip_fraction": [],
        }

        print(f"[MAPPO] Starting training: {budget:,} timesteps, "
              f"device={self._device}")
        print(f"[MAPPO] Actor: {self.actor}")
        print(f"[MAPPO] Critic: {self.critic}")

        t_start = time.time()

        while self._total_steps < budget:
            # --- LR decay ---
            if cfg.use_linear_lr_decay:
                frac = 1.0 - self._total_steps / budget
                for pg in self.actor_optimizer.param_groups:
                    pg["lr"] = cfg.lr_actor * frac
                for pg in self.critic_optimizer.param_groups:
                    pg["lr"] = cfg.lr_critic * frac

            # --- Collect rollout ---
            rollout_info = self.collect_rollout(env)

            # --- PPO update ---
            stats = self.update()

            # Record history
            history["actor_loss"].append(stats.actor_loss)
            history["critic_loss"].append(stats.critic_loss)
            history["entropy"].append(stats.entropy)
            history["mean_reward"].append(rollout_info["mean_episode_reward"])
            history["approx_kl"].append(stats.approx_kl)
            history["clip_fraction"].append(stats.clip_fraction)

            # --- Logging ---
            if self._update_count % cfg.log_interval == 0:
                elapsed = time.time() - t_start
                fps = self._total_steps / max(elapsed, 1)
                print(
                    f"[MAPPO] Update {self._update_count:4d} | "
                    f"Steps: {self._total_steps:>9,} / {budget:,} | "
                    f"Reward: {rollout_info['mean_episode_reward']:>8.3f} | "
                    f"Loss: {stats.actor_loss:.4f} / {stats.critic_loss:.4f} | "
                    f"Ent: {stats.entropy:.4f} | "
                    f"KL: {stats.approx_kl:.4f} | "
                    f"Clip: {stats.clip_fraction:.3f} | "
                    f"FPS: {fps:.0f}"
                )

            # --- Checkpointing ---
            if cfg.save_interval > 0 and self._update_count % cfg.save_interval == 0:
                self.save(f"checkpoint_{self._update_count}.pt")

            # --- Callback ---
            if callback is not None:
                callback(self, self._update_count, stats)

        elapsed = time.time() - t_start
        print(f"[MAPPO] Training complete: {self._total_steps:,} steps "
              f"in {elapsed:.1f}s ({self._total_steps/elapsed:.0f} FPS)")

        return history

    # ==================================================================
    # Evaluation
    # ==================================================================

    def evaluate(
        self,
        env,
        n_episodes: int = 5,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """Evaluate the current policy without training.

        Parameters
        ----------
        env : DisasterWasteEnv
        n_episodes : int
            Number of evaluation episodes.
        deterministic : bool
            Use argmax action selection.

        Returns
        -------
        dict
            Evaluation metrics (averaged over episodes).
        """
        all_rewards: List[float] = []
        all_metrics: List[Dict[str, float]] = []

        for _ep in range(n_episodes):
            obs_dict, _ = env.reset()
            agent_list = env.possible_agents
            episode_reward = 0.0
            done = False

            while not done:
                obs_np = np.array([
                    obs_dict[agent]["obs"] for agent in agent_list
                ], dtype=np.float32)
                masks_np = np.array([
                    obs_dict[agent]["action_mask"] for agent in agent_list
                ], dtype=np.float32)

                obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self._device)
                masks_t = torch.tensor(masks_np, dtype=torch.float32, device=self._device)

                with torch.no_grad():
                    actions_t, _, _ = self.actor.get_action(
                        obs_t, masks_t, deterministic=deterministic
                    )

                action_dict = {
                    agent: int(actions_t[i].item())
                    for i, agent in enumerate(agent_list)
                }
                obs_dict, rewards, terms, truncs, _ = env.step(action_dict)

                episode_reward += sum(rewards.values()) / len(rewards)
                done = any(truncs.values()) or any(terms.values())

            all_rewards.append(episode_reward)
            all_metrics.append(env.get_episode_metrics())

        return {
            "mean_reward": float(np.mean(all_rewards)),
            "std_reward": float(np.std(all_rewards)),
            "mean_cost": float(np.mean([m["total_cost"] for m in all_metrics])),
            "mean_emission": float(np.mean([m["total_emission"] for m in all_metrics])),
            "mean_service_level": float(np.mean([m["service_level"] for m in all_metrics])),
        }

    # ==================================================================
    # Save / Load
    # ==================================================================

    def save(self, filepath: str) -> None:
        """Save model parameters and optimiser states."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "update_count": self._update_count,
            "total_steps": self._total_steps,
            "config": self._config,
        }, str(path))

    def load(self, filepath: str, load_optimizer: bool = True) -> None:
        """Load model parameters from a checkpoint."""
        ckpt = torch.load(filepath, map_location=self._device, weights_only=False)

        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])

        if load_optimizer and "actor_optimizer" in ckpt:
            self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
            self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])

        self._update_count = ckpt.get("update_count", 0)
        self._total_steps = ckpt.get("total_steps", 0)

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def config(self) -> MAPPOConfig:
        return self._config

    def __repr__(self) -> str:
        return (
            f"MAPPO(agents={self._n_agents}, "
            f"actor={self.actor}, critic={self.critic}, "
            f"device={self._device})"
        )
