"""
nearest_neighbor.py — Greedy Nearest Neighbour Heuristic for Disaster Waste VRP
==================================================================================

Classic constructive heuristic adapted for the dynamic disaster waste
management problem.  Each vehicle greedily selects the closest
waste-generation node (by effective travel time) that has available
waste, picks up as much as it can carry, then proceeds to the nearest
facility to drop off.

The algorithm runs **inside** the PettingZoo environment, calling
``env.step()`` at each decision point so that all dynamics (Poisson
road damage, stochastic waste generation) are faithfully simulated,
enabling fair comparison against MAPPO.

Algorithm
---------
For each vehicle at each decision point:
    1. If carrying cargo → go to nearest reachable facility and drop off.
    2. If empty → go to nearest reachable waste node with storage > 0.
    3. If no reachable target exists → wait.
    4. Repeat until episode ends (truncation).

Author  : Muhammed Şara
License : MIT
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.environment.disaster_waste_env import DisasterWasteEnv
from src.environment.network import NodeType


class NearestNeighborBaseline:
    """Greedy nearest-neighbour heuristic for disaster waste logistics.

    Parameters
    ----------
    health_threshold : float
        Minimum road health to consider a path traversable.
    pickup_threshold : float
        Minimum waste storage at a node to warrant a pickup.

    Examples
    --------
    >>> nn = NearestNeighborBaseline()
    >>> metrics = nn.solve(env)
    >>> print(metrics["service_level"])
    """

    def __init__(
        self,
        health_threshold: float = 0.1,
        pickup_threshold: float = 0.5,
    ) -> None:
        self._health_thresh = health_threshold
        self._pickup_thresh = pickup_threshold

    def solve(
        self,
        env: DisasterWasteEnv,
        seed: int = 42,
    ) -> Dict[str, float]:
        """Run the nearest-neighbour heuristic for one full episode.

        Parameters
        ----------
        env : DisasterWasteEnv
            Environment instance (will be reset).
        seed : int
            Random seed for the reset.

        Returns
        -------
        dict
            Episode-level KPI metrics (same format as
            ``env.get_episode_metrics()``).
        """
        obs_dict, _ = env.reset(seed=seed)
        agent_list = env.possible_agents
        network = env._network

        total_reward = 0.0
        done = False

        while not done:
            actions: Dict[str, int] = {}

            for agent in env.agents:
                veh = env._agent_vehicle[agent]
                mask = obs_dict[agent]["action_mask"]
                action = self._select_action(env, veh, mask, network)
                actions[agent] = action

            obs_dict, rewards, terms, truncs, infos = env.step(actions)
            total_reward += sum(rewards.values()) / len(rewards)
            done = any(truncs.values()) or any(terms.values())

        metrics = env.get_episode_metrics()
        metrics["total_reward"] = total_reward
        metrics["algorithm"] = "NearestNeighbor"
        return metrics

    def _select_action(
        self,
        env: DisasterWasteEnv,
        veh,
        mask: np.ndarray,
        network,
    ) -> int:
        """Select the best greedy action for a vehicle.

        Decision logic:
            1. If vehicle has cargo → find nearest facility, move toward it.
            2. If at a waste node with storage → pickup.
            3. If at a facility with cargo → dropoff.
            4. If empty → find nearest waste node with storage, move toward it.
            5. Fallback → wait.
        """
        current = veh.current_node
        node_type = network.get_node_type(current)
        neighbours = network.get_neighbors(current)
        max_deg = env._max_degree
        wait_action = max_deg + 2
        pickup_action = max_deg + 0
        dropoff_action = max_deg + 1

        # --- Dropoff if at facility and carrying cargo ---
        if mask[dropoff_action] == 1:
            return dropoff_action

        # --- Pickup if at waste node with waste and not full ---
        if mask[pickup_action] == 1:
            return pickup_action

        # --- Movement: choose target based on vehicle state ---
        if not veh.is_empty:
            # Carrying cargo → head toward nearest facility
            target_types = (
                NodeType.SORTING_FACILITY, NodeType.LANDFILL, NodeType.TCP
            )
            target_nodes = []
            for nt in target_types:
                target_nodes.extend(network.get_nodes_by_type(nt))
        else:
            # Empty → head toward nearest waste node with storage
            gen_ids = network.get_nodes_by_type(NodeType.WASTE_GENERATION)
            target_nodes = [
                nid for nid in gen_ids
                if (nid in env._waste_model.node_ids and
                    env._waste_model.get_node_storage(nid) > self._pickup_thresh)
            ]

        if not target_nodes:
            return wait_action

        # Find the best neighbour that brings us closer to any target
        best_action = wait_action
        best_score = float("inf")

        cur_pos = np.array(network.get_node_position(current))

        for i, nbr in enumerate(neighbours):
            if i >= max_deg or mask[i] == 0:
                continue
            nbr_pos = np.array(network.get_node_position(nbr))

            # Find distance from this neighbour to nearest target
            min_dist = float("inf")
            for tgt in target_nodes:
                tgt_pos = np.array(network.get_node_position(tgt))
                d = float(np.linalg.norm(nbr_pos - tgt_pos))
                min_dist = min(min_dist, d)

            # Penalise low-health edges
            health = network.get_edge_health(current, nbr)
            score = min_dist / max(health, 0.01)

            if score < best_score:
                best_score = score
                best_action = i

        return best_action

    def solve_batch(
        self,
        env: DisasterWasteEnv,
        n_episodes: int = 5,
        seed: int = 42,
    ) -> Dict[str, float]:
        """Run multiple episodes and return averaged metrics."""
        all_metrics: List[Dict[str, float]] = []
        for ep in range(n_episodes):
            m = self.solve(env, seed=seed + ep)
            all_metrics.append(m)

        avg = {}
        keys = [k for k in all_metrics[0] if isinstance(all_metrics[0][k], (int, float))]
        for k in keys:
            avg[f"mean_{k}"] = float(np.mean([m[k] for m in all_metrics]))
            avg[f"std_{k}"] = float(np.std([m[k] for m in all_metrics]))

        avg["algorithm"] = "NearestNeighbor"
        avg["n_episodes"] = n_episodes
        return avg

    def __repr__(self) -> str:
        return (
            f"NearestNeighborBaseline(health_thresh={self._health_thresh}, "
            f"pickup_thresh={self._pickup_thresh})"
        )
