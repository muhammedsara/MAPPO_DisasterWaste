"""
single_ppo.py — Single-Agent PPO Baseline (Ablation Study)
============================================================

This module implements a standard (single-agent) PPO controller that
manages **all vehicles** from a centralised perspective.  It serves as
an ablation baseline to quantify the benefit of MAPPO's multi-agent
decentralised execution over a monolithic centralised controller.

Key differences from MAPPO
---------------------------
+-------------------+-----------------------------+-----------------------------+
| Aspect            | MAPPO (ours)                | Single-PPO (this)           |
+===================+=============================+=============================+
| Architecture      | N decentralised actors      | 1 centralised actor         |
|                   | + 1 shared critic           | + 1 critic                  |
+-------------------+-----------------------------+-----------------------------+
| Observation       | Local obs per agent         | Global state                |
+-------------------+-----------------------------+-----------------------------+
| Action space      | Discrete per agent          | MultiDiscrete (all agents)  |
|                   | (action_dim)                | [action_dim]^N              |
+-------------------+-----------------------------+-----------------------------+
| Scalability       | O(1) per agent              | O(N × action_dim)           |
|                   | (inference)                 | (exponential)               |
+-------------------+-----------------------------+-----------------------------+

The single-agent formulation suffers from:
    - **Exponential action space**: |A|^N where N = num_vehicles.
    - **Credit assignment**: Cannot attribute reward to individual agents.
    - **No specialisation**: All vehicles share one decision-maker.

Author  : Muhammed Şara
License : MIT
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.environment.disaster_waste_env import DisasterWasteEnv


# ---------------------------------------------------------------------------
# Centralised Actor for Multi-Discrete action space
# ---------------------------------------------------------------------------

def _orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    """Apply orthogonal initialisation to Linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class CentralisedActor(nn.Module):
    """Single actor controlling all vehicles via MultiDiscrete output.

    Input:  global_state (state_dim)
    Output: N independent Categorical distributions, one per vehicle.

    Each vehicle head produces logits over action_dim, with action
    masking applied per-vehicle.

    Parameters
    ----------
    state_dim : int
        Global state dimension.
    n_agents : int
        Number of vehicles.
    action_dim : int
        Per-vehicle action space size.
    hidden_sizes : Tuple[int, ...]
        Shared backbone hidden layer widths.
    """

    MASK_VALUE: float = -1e9

    def __init__(
        self,
        state_dim: int,
        n_agents: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (512, 256, 128),
    ) -> None:
        super().__init__()
        self._state_dim = state_dim
        self._n_agents = n_agents
        self._action_dim = action_dim

        # Shared backbone
        layers: list[nn.Module] = []
        prev = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev = h
        self.backbone = nn.Sequential(*layers)

        # Per-vehicle action heads
        self.heads = nn.ModuleList([
            nn.Linear(prev, action_dim) for _ in range(n_agents)
        ])

        # Init
        self.backbone.apply(lambda m: _orthogonal_init(m, gain=nn.init.calculate_gain("relu")))
        for head in self.heads:
            _orthogonal_init(head, gain=0.01)

    def forward(
        self,
        state: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> List[Categorical]:
        """Forward pass → list of per-vehicle Categorical distributions.

        Parameters
        ----------
        state : torch.Tensor
            Shape ``(batch, state_dim)`` or ``(state_dim,)``.
        action_masks : torch.Tensor
            Shape ``(batch, n_agents, action_dim)`` or
            ``(n_agents, action_dim)``.

        Returns
        -------
        list of Categorical
            One distribution per vehicle.
        """
        squeeze = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            action_masks = action_masks.unsqueeze(0)
            squeeze = True

        features = self.backbone(state)  # (B, hidden)

        distributions = []
        for i, head in enumerate(self.heads):
            logits = head(features)  # (B, action_dim)
            mask = action_masks[:, i, :]  # (B, action_dim)
            logits = logits.clone()
            logits[mask == 0] = self.MASK_VALUE
            distributions.append(Categorical(logits=logits))

        return distributions

    def get_actions(
        self,
        state: torch.Tensor,
        action_masks: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions for all vehicles.

        Returns
        -------
        actions : (n_agents,) or (B, n_agents)
        log_probs : (n_agents,) or (B, n_agents)
        entropy : (n_agents,) or (B, n_agents)
        """
        squeeze = state.dim() == 1
        dists = self.forward(state, action_masks)
        actions_list, lp_list, ent_list = [], [], []

        for dist in dists:
            if deterministic:
                a = dist.probs.argmax(dim=-1)
            else:
                a = dist.sample()
            actions_list.append(a)
            lp_list.append(dist.log_prob(a))
            ent_list.append(dist.entropy())

        actions = torch.stack(actions_list, dim=-1)  # (B, n_agents)
        log_probs = torch.stack(lp_list, dim=-1)
        entropy = torch.stack(ent_list, dim=-1)

        if squeeze:
            actions = actions.squeeze(0)
            log_probs = log_probs.squeeze(0)
            entropy = entropy.squeeze(0)

        return actions, log_probs, entropy

    def evaluate_actions(
        self,
        state: torch.Tensor,
        action_masks: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log-probs and entropy for given actions.

        Parameters
        ----------
        actions : (B, n_agents)
        """
        dists = self.forward(state, action_masks)
        lp_list, ent_list = [], []

        for i, dist in enumerate(dists):
            lp_list.append(dist.log_prob(actions[:, i]))
            ent_list.append(dist.entropy())

        return torch.stack(lp_list, dim=-1), torch.stack(ent_list, dim=-1)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Centralised Critic (reuse concept from critic_network.py)
# ---------------------------------------------------------------------------

class CentralisedCritic(nn.Module):
    """Value network for single-agent PPO. Same as MAPPO critic."""

    def __init__(
        self,
        state_dim: int,
        hidden_sizes: Tuple[int, ...] = (512, 256, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev = h
        self.network = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev, 1)

        self.network.apply(lambda m: _orthogonal_init(m, gain=nn.init.calculate_gain("relu")))
        _orthogonal_init(self.value_head, gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.value_head(self.network(state)).squeeze(-1)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Single-PPO configuration
# ---------------------------------------------------------------------------

@dataclass
class SinglePPOConfig:
    """Hyper-parameters for single-agent PPO."""
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    mini_batch_size: int = 64
    rollout_length: int = 128
    total_timesteps: int = 1_000_000
    log_interval: int = 5


# ---------------------------------------------------------------------------
# Single-PPO Algorithm
# ---------------------------------------------------------------------------

class SinglePPO:
    """Standard single-agent PPO managing all vehicles centrally.

    Parameters
    ----------
    n_agents : int
        Number of vehicles.
    state_dim : int
        Global state dimension.
    action_dim : int
        Per-vehicle discrete action space size.
    config : SinglePPOConfig, optional
    device : torch.device, optional

    Examples
    --------
    >>> sppo = SinglePPO(n_agents=4, state_dim=122, action_dim=9)
    >>> metrics = sppo.train(env, total_timesteps=10000)
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        action_dim: int,
        config: Optional[SinglePPOConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self._config = config or SinglePPOConfig()
        self._n_agents = n_agents
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.actor = CentralisedActor(
            state_dim=state_dim,
            n_agents=n_agents,
            action_dim=action_dim,
        ).to(self._device)

        self.critic = CentralisedCritic(
            state_dim=state_dim,
        ).to(self._device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self._config.lr_actor, eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self._config.lr_critic, eps=1e-5
        )

        self._total_steps = 0
        self._update_count = 0

    def collect_and_update(
        self, env: DisasterWasteEnv,
    ) -> Dict[str, float]:
        """Collect rollout + PPO update (one cycle).

        Returns
        -------
        dict
            Training metrics for this cycle.
        """
        cfg = self._config
        T = cfg.rollout_length

        # Storage
        states_buf = np.zeros((T, self._state_dim), dtype=np.float32)
        masks_buf = np.zeros((T, self._n_agents, self._action_dim), dtype=np.float32)
        actions_buf = np.zeros((T, self._n_agents), dtype=np.int64)
        log_probs_buf = np.zeros((T, self._n_agents), dtype=np.float32)
        values_buf = np.zeros(T, dtype=np.float32)
        rewards_buf = np.zeros(T, dtype=np.float32)
        dones_buf = np.zeros(T, dtype=np.float32)

        # Collect rollout
        obs_dict, _ = env.reset()
        agent_list = env.possible_agents

        for t in range(T):
            # Global state (same for all agents, take first)
            state_np = obs_dict[agent_list[0]]["global_state"]
            mask_np = np.array([
                obs_dict[a]["action_mask"] for a in agent_list
            ], dtype=np.float32)

            state_t = torch.tensor(state_np, dtype=torch.float32, device=self._device)
            mask_t = torch.tensor(mask_np, dtype=torch.float32, device=self._device)

            with torch.no_grad():
                actions_t, lp_t, _ = self.actor.get_actions(state_t, mask_t)
                value_t = self.critic(state_t)

            actions_np = actions_t.cpu().numpy()
            action_dict = {
                a: int(actions_np[i]) for i, a in enumerate(agent_list)
            }

            next_obs, rewards, terms, truncs, _ = env.step(action_dict)
            mean_reward = sum(rewards.values()) / len(rewards)
            done = any(truncs.values()) or any(terms.values())

            states_buf[t] = state_np
            masks_buf[t] = mask_np
            actions_buf[t] = actions_np
            log_probs_buf[t] = lp_t.cpu().numpy()
            values_buf[t] = value_t.item()
            rewards_buf[t] = mean_reward
            dones_buf[t] = float(done)

            obs_dict = next_obs
            self._total_steps += self._n_agents

            if done:
                obs_dict, _ = env.reset()

        # Bootstrap value
        last_state = torch.tensor(
            obs_dict[agent_list[0]]["global_state"],
            dtype=torch.float32, device=self._device
        )
        with torch.no_grad():
            last_value = self.critic(last_state).item()

        # GAE
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_val = last_value
                next_done = 0.0
            else:
                next_val = values_buf[t + 1]
                next_done = dones_buf[t + 1]
            delta = rewards_buf[t] + cfg.gamma * (1 - next_done) * next_val - values_buf[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * (1 - next_done) * gae
            advantages[t] = gae

        returns = advantages + values_buf
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        indices = np.arange(T)
        for _epoch in range(cfg.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, cfg.mini_batch_size):
                end = min(start + cfg.mini_batch_size, T)
                idx = indices[start:end]

                s_t = torch.tensor(states_buf[idx], dtype=torch.float32, device=self._device)
                m_t = torch.tensor(masks_buf[idx], dtype=torch.float32, device=self._device)
                a_t = torch.tensor(actions_buf[idx], dtype=torch.long, device=self._device)
                old_lp = torch.tensor(log_probs_buf[idx], dtype=torch.float32, device=self._device)
                adv_t = torch.tensor(advantages[idx], dtype=torch.float32, device=self._device)
                ret_t = torch.tensor(returns[idx], dtype=torch.float32, device=self._device)

                new_lp, ent = self.actor.evaluate_actions(s_t, m_t, a_t)
                new_val = self.critic(s_t)

                # Sum log probs across agents for joint probability
                new_lp_sum = new_lp.sum(dim=-1)
                old_lp_sum = old_lp.sum(dim=-1)

                ratio = torch.exp(new_lp_sum - old_lp_sum)
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv_t
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = 0.5 * ((new_val - ret_t) ** 2).mean()
                entropy = ent.mean()

                loss = actor_loss + cfg.value_coef * critic_loss - cfg.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        self._update_count += 1
        n = max(n_updates, 1)

        return {
            "actor_loss": total_actor_loss / n,
            "critic_loss": total_critic_loss / n,
            "entropy": total_entropy / n,
            "mean_reward": float(rewards_buf.mean()),
            "total_steps": self._total_steps,
        }

    def train(
        self,
        env: DisasterWasteEnv,
        total_timesteps: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Run the full training loop.

        Parameters
        ----------
        env : DisasterWasteEnv
        total_timesteps : int, optional

        Returns
        -------
        dict
            Training history.
        """
        budget = total_timesteps or self._config.total_timesteps
        cfg = self._config

        history: Dict[str, List[float]] = {
            "actor_loss": [], "critic_loss": [], "entropy": [], "mean_reward": [],
        }

        print(f"[SinglePPO] Training: {budget:,} steps, device={self._device}")
        print(f"[SinglePPO] Actor: {self.actor.num_parameters:,} params")
        print(f"[SinglePPO] Critic: {self.critic.num_parameters:,} params")

        t_start = time.time()

        while self._total_steps < budget:
            stats = self.collect_and_update(env)

            for k in history:
                if k in stats:
                    history[k].append(stats[k])

            if self._update_count % cfg.log_interval == 0:
                elapsed = time.time() - t_start
                fps = self._total_steps / max(elapsed, 1)
                print(
                    f"[SinglePPO] Update {self._update_count:4d} | "
                    f"Steps: {self._total_steps:>9,} | "
                    f"Reward: {stats['mean_reward']:>8.3f} | "
                    f"Loss: {stats['actor_loss']:.4f} / {stats['critic_loss']:.4f} | "
                    f"Ent: {stats['entropy']:.4f} | "
                    f"FPS: {fps:.0f}"
                )

        elapsed = time.time() - t_start
        print(f"[SinglePPO] Done: {self._total_steps:,} steps in {elapsed:.1f}s")
        return history

    def evaluate(
        self,
        env: DisasterWasteEnv,
        n_episodes: int = 5,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """Evaluate without training."""
        all_rewards: List[float] = []
        all_metrics: List[Dict] = []

        for _ep in range(n_episodes):
            obs_dict, _ = env.reset()
            agents = env.possible_agents
            ep_reward = 0.0
            done = False

            while not done:
                state_np = obs_dict[agents[0]]["global_state"]
                mask_np = np.array([
                    obs_dict[a]["action_mask"] for a in agents
                ], dtype=np.float32)

                state_t = torch.tensor(state_np, dtype=torch.float32, device=self._device)
                mask_t = torch.tensor(mask_np, dtype=torch.float32, device=self._device)

                with torch.no_grad():
                    acts, _, _ = self.actor.get_actions(state_t, mask_t, deterministic)

                act_np = acts.cpu().numpy()
                action_dict = {a: int(act_np[i]) for i, a in enumerate(agents)}

                obs_dict, rewards, terms, truncs, _ = env.step(action_dict)
                ep_reward += sum(rewards.values()) / len(rewards)
                done = any(truncs.values()) or any(terms.values())

            all_rewards.append(ep_reward)
            all_metrics.append(env.get_episode_metrics())

        return {
            "mean_reward": float(np.mean(all_rewards)),
            "std_reward": float(np.std(all_rewards)),
            "mean_cost": float(np.mean([m["total_cost"] for m in all_metrics])),
            "mean_emission": float(np.mean([m["total_emission"] for m in all_metrics])),
            "algorithm": "SinglePPO",
        }

    def __repr__(self) -> str:
        return (
            f"SinglePPO(agents={self._n_agents}, "
            f"actor_params={self.actor.num_parameters:,}, "
            f"critic_params={self.critic.num_parameters:,}, "
            f"device={self._device})"
        )
