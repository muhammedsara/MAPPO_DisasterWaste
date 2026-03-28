"""
train.py — MAPPO Training Script with W&B Integration
========================================================

Usage::

    cd /home/kurtar/KURTAR/WorkOut/MAPPO-DisasterWaste
    python src/experiments/train.py [--scenario S2_MEDIUM] [--timesteps 500000]

This script:
    1. Initialises a DisasterWasteEnv with the specified scenario.
    2. Creates a MAPPO agent with CTDE architecture.
    3. Trains using PPO with GAE, logging to TensorBoard + W&B.
    4. Saves the final model to results/models/mappo_final.pt.

Author  : Muhammed Şara
License : MIT
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

# W&B integration
try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False
    print("[WARN] wandb not installed. Logging to TensorBoard only.")

from src.environment import DisasterWasteEnv, ScenarioGenerator, ScenarioTier
from src.agents import MAPPO, MAPPOConfig


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MAPPO on Disaster Waste Management"
    )
    parser.add_argument(
        "--scenario", type=str, default="S2_MEDIUM",
        choices=["S1_SMALL", "S2_MEDIUM", "S3_LARGE", "S4_SEVERE"],
        help="Disaster scenario tier",
    )
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total training timesteps")
    parser.add_argument("--rollout", type=int, default=128,
                        help="Rollout length per update")
    parser.add_argument("--epochs", type=int, default=4,
                        help="PPO epochs per update")
    parser.add_argument("--lr-actor", type=float, default=3e-4,
                        help="Actor learning rate")
    parser.add_argument("--lr-critic", type=float, default=1e-3,
                        help="Critic learning rate")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Mini-batch size")
    parser.add_argument("--clip", type=float, default=0.2,
                        help="PPO clip ratio")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--entropy", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--eval-interval", type=int, default=10,
                        help="Evaluate every N rollouts")
    parser.add_argument("--save-interval", type=int, default=50,
                        help="Save checkpoint every N rollouts")
    parser.add_argument("--log-interval", type=int, default=5,
                        help="Log metrics every N rollouts")
    parser.add_argument("--wandb-project", type=str,
                        default="MAPPO-DisasterWaste",
                        help="W&B project name")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, or cuda")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    """Execute the MAPPO training pipeline."""

    # --- Seeding ---
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- Device ---
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("  MAPPO Training — Disaster Waste Management")
    print("=" * 60)
    print(f"  Scenario : {args.scenario}")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Device   : {device}")
    print(f"  Seed     : {args.seed}")
    print("=" * 60)

    # --- Directories ---
    results_dir = PROJECT_ROOT / "results"
    model_dir = results_dir / "models"
    log_dir = results_dir / "logs" / f"mappo_{args.scenario}_{args.seed}"
    checkpoint_dir = results_dir / "checkpoints"

    for d in [model_dir, log_dir, checkpoint_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # --- W&B init ---
    if _HAS_WANDB and not args.no_wandb:
        run_name = f"MAPPO_{args.scenario}_{args.timesteps // 1000}K_seed{args.seed}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            sync_tensorboard=True,
            save_code=True,
        )
        print(f"  W&B run : {wandb.run.name}")
    else:
        print("  W&B     : Disabled")

    # --- Environment ---
    tier_map = {
        "S1_SMALL": ScenarioTier.S1_SMALL,
        "S2_MEDIUM": ScenarioTier.S2_MEDIUM,
        "S3_LARGE": ScenarioTier.S3_LARGE,
        "S4_SEVERE": ScenarioTier.S4_SEVERE,
    }
    gen = ScenarioGenerator(seed=args.seed)
    scenario = gen.from_tier(tier_map[args.scenario])
    env = DisasterWasteEnv(scenario=scenario, seed=args.seed)

    n_agents = len(env.possible_agents)
    obs_dim = env._local_obs_dim
    state_dim = env._global_state_dim
    action_dim = env._action_size

    print(f"\n  Environment:")
    print(f"    Agents     : {n_agents}")
    print(f"    Obs dim    : {obs_dim}")
    print(f"    State dim  : {state_dim}")
    print(f"    Action dim : {action_dim}")

    # --- MAPPO config ---
    config = MAPPOConfig(
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip,
        entropy_coef=args.entropy,
        n_epochs=args.epochs,
        mini_batch_size=args.batch_size,
        rollout_length=args.rollout,
        total_timesteps=args.timesteps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
    )

    # --- MAPPO agent ---
    mappo = MAPPO(
        n_agents=n_agents,
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        log_dir=str(log_dir),
        device=device,
    )

    actor_params = sum(p.numel() for p in mappo.actor.parameters())
    critic_params = sum(p.numel() for p in mappo.critic.parameters())
    print(f"\n  MAPPO:")
    print(f"    Actor      : {actor_params:,} params")
    print(f"    Critic     : {critic_params:,} params")
    print(f"    Buffer     : {mappo.buffer.total_samples:,} samples/rollout")

    # --- Training ---
    print("\n" + "-" * 60)
    print("  Starting training...")
    print("-" * 60 + "\n")

    t_start = time.time()

    # Custom callback for evaluation + W&B
    best_reward = float("-inf")

    def training_callback(agent, update_count, stats):
        nonlocal best_reward

        # W&B logging
        if _HAS_WANDB and not args.no_wandb:
            wandb.log({
                "train/actor_loss": stats.actor_loss,
                "train/critic_loss": stats.critic_loss,
                "train/entropy": stats.entropy,
                "train/approx_kl": stats.approx_kl,
                "train/clip_fraction": stats.clip_fraction,
                "train/actor_lr": stats.actor_lr,
                "train/critic_lr": stats.critic_lr,
                "train/total_steps": agent.total_steps,
            }, step=update_count)

        # Periodic evaluation
        if update_count % config.eval_interval == 0:
            eval_metrics = agent.evaluate(env, n_episodes=3, deterministic=True)

            print(f"\n  [EVAL] Update {update_count}: "
                  f"reward={eval_metrics['mean_reward']:.2f}, "
                  f"cost={eval_metrics['mean_cost']:.2f}, "
                  f"emission={eval_metrics['mean_emission']:.2f}")

            if _HAS_WANDB and not args.no_wandb:
                wandb.log({
                    "eval/mean_reward": eval_metrics["mean_reward"],
                    "eval/mean_cost": eval_metrics["mean_cost"],
                    "eval/mean_emission": eval_metrics["mean_emission"],
                    "eval/mean_service_level": eval_metrics["mean_service_level"],
                }, step=update_count)

            # Save best model
            if eval_metrics["mean_reward"] > best_reward:
                best_reward = eval_metrics["mean_reward"]
                best_path = model_dir / "mappo_best.pt"
                agent.save(str(best_path))
                print(f"  [BEST] New best: {best_reward:.2f} → {best_path}")

        # Checkpoint
        if config.save_interval > 0 and update_count % config.save_interval == 0:
            ckpt_path = checkpoint_dir / f"mappo_ckpt_{update_count}.pt"
            agent.save(str(ckpt_path))

    # Run training
    history = mappo.train(
        env=env,
        total_timesteps=args.timesteps,
        callback=training_callback,
    )

    elapsed = time.time() - t_start

    # --- Save final model ---
    final_path = model_dir / "mappo_final.pt"
    mappo.save(str(final_path))

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Total steps : {mappo.total_steps:,}")
    print(f"  Updates     : {mappo.update_count}")
    print(f"  Time        : {elapsed:.1f}s ({mappo.total_steps/elapsed:.0f} FPS)")
    print(f"  Final model : {final_path}")
    print(f"  Best model  : {model_dir / 'mappo_best.pt'}")
    print(f"  TensorBoard : {log_dir}")

    # --- Final evaluation ---
    print("\n  Final Evaluation (5 episodes):")
    final_eval = mappo.evaluate(env, n_episodes=5, deterministic=True)
    for k, v in final_eval.items():
        if isinstance(v, float):
            print(f"    {k:25s}: {v:.4f}")

    if _HAS_WANDB and not args.no_wandb:
        wandb.log({"final/" + k: v for k, v in final_eval.items()
                    if isinstance(v, float)})
        wandb.finish()

    print("\n  Done! ✓")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
