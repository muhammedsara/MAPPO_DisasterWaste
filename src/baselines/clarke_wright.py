"""
clarke_wright.py — Clarke-Wright Savings Algorithm for Disaster Waste VRP
===========================================================================

The Clarke-Wright (1964) savings algorithm is one of the most widely
used constructive heuristics in Vehicle Routing Problem (VRP)
literature.  This implementation adapts it for the dynamic, multi-depot,
heterogeneous-fleet disaster waste management setting.

Algorithm
---------
1. **Initialise**: Create a "pendulum" route for each waste-generation
   node: depot → node → facility → depot.
2. **Compute savings**: For every pair of waste nodes (i, j):

   .. math::

       s(i, j) = d(\\text{depot}, i) + d(\\text{depot}, j) - d(i, j)

   Modified to incorporate road health penalties:

   .. math::

       s^*(i, j) = s(i, j) \\cdot \\min(h_{depot→i},\\, h_{depot→j},\\, h_{i→j})

3. **Merge routes**: Sort savings in descending order.  For each (i, j),
   merge their routes if:
   - i and j are in different routes
   - i is the last node in its route (or j is the first in its route)
   - Combined demand ≤ vehicle capacity.
4. **Execute merged routes** through ``env.step()`` for fair comparison.

Author  : Muhammed Şara
License : MIT
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.environment.disaster_waste_env import DisasterWasteEnv
from src.environment.network import NodeType

from dataclasses import dataclass, field


@dataclass
class _Route:
    """Internal route representation."""
    nodes: List[int]
    demand: float
    vehicle_idx: int = -1


class ClarkeWrightBaseline:
    """Clarke-Wright savings heuristic adapted for disaster waste VRP.

    The algorithm constructs routes offline using the current network
    snapshot, then executes them through the environment step-by-step
    so that dynamic events (road damage, waste generation) are captured.

    Parameters
    ----------
    health_threshold : float
        Minimum edge health for a route to be considered.
    savings_health_weight : float
        Exponent for health penalty in savings: s* = s · h^w.

    Examples
    --------
    >>> cw = ClarkeWrightBaseline()
    >>> metrics = cw.solve(env)
    """

    def __init__(
        self,
        health_threshold: float = 0.1,
        savings_health_weight: float = 1.0,
    ) -> None:
        self._health_thresh = health_threshold
        self._health_weight = savings_health_weight

    def solve(
        self,
        env: DisasterWasteEnv,
        seed: int = 42,
    ) -> Dict[str, float]:
        """Run Clarke-Wright for one full episode.

        Parameters
        ----------
        env : DisasterWasteEnv
        seed : int

        Returns
        -------
        dict
            Episode KPIs.
        """
        obs_dict, _ = env.reset(seed=seed)
        network = env._network
        waste_model = env._waste_model
        vehicles = env._vehicles

        # --- Build route plan from current snapshot ---
        routes = self._build_routes(env, network, waste_model, vehicles)

        # --- Execute routes through env.step() ---
        return self._execute_routes(env, routes, obs_dict)

    def _build_routes(
        self,
        env: DisasterWasteEnv,
        network,
        waste_model,
        vehicles,
    ) -> List[List[int]]:
        """Construct merged routes using the savings heuristic.

        Returns a list of routes, each route is a list of node IDs
        to visit in order (excluding depot, which is implicit).
        """
        depots = network.get_nodes_by_type(NodeType.DEPOT)
        gen_nodes = network.get_nodes_by_type(NodeType.WASTE_GENERATION)
        facilities = (
            network.get_nodes_by_type(NodeType.SORTING_FACILITY)
            + network.get_nodes_by_type(NodeType.TCP)
            + network.get_nodes_by_type(NodeType.LANDFILL)
        )

        if not depots or not gen_nodes:
            return []

        primary_depot = depots[0]

        # Filter nodes with actual waste
        active_nodes = [
            nid for nid in gen_nodes
            if (nid in waste_model.node_ids and
                waste_model.get_node_storage(nid) > 0.1)
        ]

        if not active_nodes:
            return []

        # --- Distance/cost matrix ---
        all_nodes = [primary_depot] + active_nodes
        n = len(all_nodes)
        dist_matrix = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                pos_i = np.array(network.get_node_position(all_nodes[i]))
                pos_j = np.array(network.get_node_position(all_nodes[j]))
                d = float(np.linalg.norm(pos_i - pos_j))

                # Health penalty
                health = 1.0
                if network.graph.has_edge(all_nodes[i], all_nodes[j]):
                    health = network.get_edge_health(all_nodes[i], all_nodes[j])
                dist_matrix[i, j] = d / max(health ** self._health_weight, 0.01)

        # Depot is index 0, customers are 1..n-1
        # --- Compute savings ---
        savings: List[Tuple[float, int, int]] = []
        for i in range(1, n):
            for j in range(i + 1, n):
                s = dist_matrix[0, i] + dist_matrix[0, j] - dist_matrix[i, j]
                if s > 0:
                    savings.append((s, i, j))

        savings.sort(key=lambda x: x[0], reverse=True)

        # --- Initial routes: one per customer ---
        routes: List[_Route] = []
        node_to_route: Dict[int, int] = {}  # customer_idx → route_idx

        demands = {}
        for idx, nid in enumerate(active_nodes):
            d = waste_model.get_node_storage(nid) if nid in waste_model.node_ids else 0.0
            demands[idx + 1] = d  # +1 because depot is index 0

        for idx in range(1, n):
            r = _Route(nodes=[idx], demand=demands.get(idx, 0.0))
            route_idx = len(routes)
            routes.append(r)
            node_to_route[idx] = route_idx

        # Vehicle capacities
        capacities = sorted([v.config.capacity for v in vehicles], reverse=True)
        max_cap = capacities[0] if capacities else float("inf")

        # --- Merge routes by savings ---
        for s_val, i, j in savings:
            ri = node_to_route.get(i)
            rj = node_to_route.get(j)

            if ri is None or rj is None or ri == rj:
                continue

            route_i = routes[ri]
            route_j = routes[rj]

            # Check if i is at end of route_i and j is at start of route_j
            # (or vice versa) — relaxed: allow merge if at boundaries
            can_merge = (
                (route_i.nodes[-1] == i and route_j.nodes[0] == j) or
                (route_i.nodes[-1] == j and route_j.nodes[0] == i) or
                (route_i.nodes[0] == i and route_j.nodes[-1] == j) or
                (route_i.nodes[0] == j and route_j.nodes[-1] == i)
            )

            if not can_merge:
                continue

            combined_demand = route_i.demand + route_j.demand
            if combined_demand > max_cap:
                continue

            # Merge: append route_j to route_i
            if route_i.nodes[-1] == i and route_j.nodes[0] == j:
                route_i.nodes.extend(route_j.nodes)
            elif route_j.nodes[-1] == j and route_i.nodes[0] == i:
                route_j.nodes.extend(route_i.nodes)
                route_i.nodes = route_j.nodes
            else:
                route_i.nodes.extend(route_j.nodes)

            route_i.demand = combined_demand

            # Update mapping
            for node_idx in route_j.nodes:
                node_to_route[node_idx] = ri
            routes[rj] = _Route(nodes=[], demand=0.0)  # mark empty

        # Convert customer indices back to node IDs
        result: List[List[int]] = []
        for route in routes:
            if route.nodes:
                node_ids = [active_nodes[idx - 1] for idx in route.nodes]
                result.append(node_ids)

        # Assign nearest facility to end of each route
        final_routes: List[List[int]] = []
        for route_nodes in result:
            if not route_nodes:
                continue
            last_pos = np.array(network.get_node_position(route_nodes[-1]))
            best_fac = None
            best_dist = float("inf")
            for fid in facilities:
                fpos = np.array(network.get_node_position(fid))
                d = float(np.linalg.norm(last_pos - fpos))
                if d < best_dist:
                    best_dist = d
                    best_fac = fid
            if best_fac is not None:
                route_nodes.append(best_fac)
            final_routes.append(route_nodes)

        return final_routes

    def _execute_routes(
        self,
        env: DisasterWasteEnv,
        routes: List[List[int]],
        obs_dict: Dict,
    ) -> Dict[str, float]:
        """Execute pre-planned routes through env.step().

        Each vehicle follows its assigned route.  If a planned move
        is blocked (mask=0), the vehicle waits.
        """
        agent_list = env.possible_agents
        n_agents = len(agent_list)
        max_deg = env._max_degree
        wait_action = max_deg + 2
        pickup_action = max_deg + 0
        dropoff_action = max_deg + 1

        # Assign routes to vehicles (round-robin)
        agent_routes: Dict[str, List[int]] = {}
        agent_route_idx: Dict[str, int] = {}
        for i, agent in enumerate(agent_list):
            if i < len(routes):
                agent_routes[agent] = routes[i]
            else:
                agent_routes[agent] = []
            agent_route_idx[agent] = 0

        total_reward = 0.0
        done = False

        while not done:
            actions: Dict[str, int] = {}

            for agent in env.agents:
                veh = env._agent_vehicle[agent]
                mask = obs_dict[agent]["action_mask"]
                network = env._network

                # Check for dropoff / pickup opportunities first
                if mask[dropoff_action] == 1:
                    actions[agent] = dropoff_action
                    continue
                if mask[pickup_action] == 1:
                    actions[agent] = pickup_action
                    continue

                # Follow the route
                route = agent_routes.get(agent, [])
                ridx = agent_route_idx.get(agent, 0)

                if ridx < len(route):
                    target = route[ridx]
                    if veh.current_node == target:
                        agent_route_idx[agent] = ridx + 1
                        if ridx + 1 < len(route):
                            target = route[ridx + 1]
                            agent_route_idx[agent] = ridx + 1
                        else:
                            actions[agent] = wait_action
                            continue

                    # Find movement toward target
                    neighbours = network.get_neighbors(veh.current_node)
                    best_action = wait_action
                    best_dist = float("inf")
                    tgt_pos = np.array(network.get_node_position(target))

                    for ni, nbr in enumerate(neighbours):
                        if ni >= max_deg or mask[ni] == 0:
                            continue
                        if nbr == target:
                            best_action = ni
                            break
                        nbr_pos = np.array(network.get_node_position(nbr))
                        d = float(np.linalg.norm(nbr_pos - tgt_pos))
                        if d < best_dist:
                            best_dist = d
                            best_action = ni

                    actions[agent] = best_action
                else:
                    # Route exhausted → greedy fallback
                    actions[agent] = self._greedy_fallback(env, veh, mask, max_deg)

            obs_dict, rewards, terms, truncs, infos = env.step(actions)
            total_reward += sum(rewards.values()) / len(rewards)
            done = any(truncs.values()) or any(terms.values())

        metrics = env.get_episode_metrics()
        metrics["total_reward"] = total_reward
        metrics["algorithm"] = "ClarkeWright"
        return metrics

    def _greedy_fallback(self, env, veh, mask, max_deg) -> int:
        """Fallback to nearest-neighbour when route is exhausted."""
        wait = max_deg + 2
        pickup = max_deg + 0
        dropoff = max_deg + 1

        if mask[dropoff] == 1:
            return dropoff
        if mask[pickup] == 1:
            return pickup

        # Move toward nearest waste node
        network = env._network
        gen_nodes = [
            nid for nid in network.get_nodes_by_type(NodeType.WASTE_GENERATION)
            if (nid in env._waste_model.node_ids and
                env._waste_model.get_node_storage(nid) > 0.1)
        ]
        if not gen_nodes:
            return wait

        cur_pos = np.array(network.get_node_position(veh.current_node))
        neighbours = network.get_neighbors(veh.current_node)
        best_action = wait
        best_score = float("inf")

        for ni, nbr in enumerate(neighbours):
            if ni >= max_deg or mask[ni] == 0:
                continue
            nbr_pos = np.array(network.get_node_position(nbr))
            for tgt in gen_nodes:
                tgt_pos = np.array(network.get_node_position(tgt))
                d = float(np.linalg.norm(nbr_pos - tgt_pos))
                if d < best_score:
                    best_score = d
                    best_action = ni

        return best_action

    def solve_batch(
        self,
        env: DisasterWasteEnv,
        n_episodes: int = 5,
        seed: int = 42,
    ) -> Dict[str, float]:
        """Run multiple episodes and return averaged metrics."""
        all_metrics: List[Dict] = []
        for ep in range(n_episodes):
            m = self.solve(env, seed=seed + ep)
            all_metrics.append(m)

        avg = {}
        keys = [k for k in all_metrics[0] if isinstance(all_metrics[0][k], (int, float))]
        for k in keys:
            avg[f"mean_{k}"] = float(np.mean([m[k] for m in all_metrics]))
            avg[f"std_{k}"] = float(np.std([m[k] for m in all_metrics]))
        avg["algorithm"] = "ClarkeWright"
        avg["n_episodes"] = n_episodes
        return avg

    def __repr__(self) -> str:
        return f"ClarkeWrightBaseline(health_thresh={self._health_thresh})"
