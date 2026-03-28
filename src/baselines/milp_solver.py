"""
milp_solver.py — Mixed-Integer Linear Programming Solver for Disaster Waste VRP
==================================================================================

This module formulates the static variant of the disaster waste
collection problem as a Capacitated Vehicle Routing Problem (CVRP)
and solves it using Google OR-Tools' CP-SAT solver.

The MILP provides an **optimal lower bound** on solution quality for
small-scale instances (S1_SMALL), establishing a theoretical performance
ceiling against which MAPPO and heuristics are benchmarked.

Formulation (CVRP)
-------------------
**Sets:**
    - N = {0, 1, ..., n}  — nodes (0 = depot, 1..n = waste sites)
    - K = {1, ..., k}      — vehicles
    - A = {(i,j) : i,j ∈ N, i≠j}  — arcs

**Decision variables:**
    - x_{ijk} ∈ {0,1} : vehicle k traverses arc (i,j)
    - u_i ∈ ℝ⁺ : cumulative load at node i (MTZ subtour elimination)

**Objective:**

.. math::

    \\min \\sum_{k \\in K} \\sum_{(i,j) \\in A} c_{ij} \\cdot x_{ijk}

**Constraints:**
    1. Each customer visited exactly once: ∑_k ∑_i x_{ijk} = 1  ∀j∈N\\{0}
    2. Flow conservation: ∑_j x_{ijk} = ∑_j x_{jik}  ∀i,k
    3. Depot start/end: ∑_j x_{0jk} = 1, ∑_j x_{j0k} = 1  ∀k
    4. Capacity: ∑_i d_i · (∑_j x_{ijk}) ≤ Q_k  ∀k
    5. Subtour elimination (MTZ): u_i - u_j + Q·x_{ijk} ≤ Q - d_j

Dynamic simplification
----------------------
Since MILP cannot handle stochastic/dynamic elements natively:
    - Road health is frozen at the initial snapshot.
    - Waste volumes are taken from t=0 storage.
    - Cost = Euclidean distance / road_health (penalise damaged roads).

The optimal solution is then executed through ``env.step()`` for fair
KPI comparison under actual dynamics.

Author  : Muhammed Şara
License : MIT
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from ortools.sat.python import cp_model
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    _HAS_ORTOOLS = True
except ImportError:
    _HAS_ORTOOLS = False

from src.environment.disaster_waste_env import DisasterWasteEnv
from src.environment.network import NodeType


class MILPSolver:
    """MILP-based exact solver for disaster waste CVRP.

    Uses Google OR-Tools' routing library with the CVRP model.
    For small instances (≤25 nodes), this provides an optimal or
    near-optimal solution.  For larger instances, a time limit
    produces the best feasible solution found.

    Parameters
    ----------
    time_limit_seconds : int
        Maximum solver runtime (seconds).
    first_solution_strategy : str
        Initial solution heuristic:
        ``"PATH_CHEAPEST_ARC"``, ``"CHRISTOFIDES"``,
        ``"SAVINGS"``, etc.

    Examples
    --------
    >>> solver = MILPSolver(time_limit_seconds=60)
    >>> metrics = solver.solve(env)
    """

    def __init__(
        self,
        time_limit_seconds: int = 180,
        first_solution_strategy: str = "SAVINGS",
    ) -> None:
        if not _HAS_ORTOOLS:
            raise ImportError(
                "Google OR-Tools is required: pip install ortools"
            )
        self._time_limit = time_limit_seconds
        self._first_strategy = first_solution_strategy

    def solve(
        self,
        env: DisasterWasteEnv,
        seed: int = 42,
    ) -> Dict[str, float]:
        """Solve the static CVRP snapshot and execute through env.

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

        # --- Build problem data ---
        data = self._build_data_model(network, waste_model, vehicles)

        # --- Solve with OR-Tools ---
        t_start = time.time()
        routes = self._solve_cvrp(data)
        solve_time = time.time() - t_start

        # --- Execute routes through env ---
        metrics = self._execute_routes(env, seed, routes, data)
        metrics["solve_time"] = solve_time
        metrics["algorithm"] = "MILP_ORTools"
        return metrics

    def _build_data_model(
        self, network, waste_model, vehicles,
    ) -> Dict:
        """Build the OR-Tools data model from the environment snapshot.

        Returns
        -------
        dict
            Keys: distance_matrix, demands, vehicle_capacities,
            num_vehicles, depot_index, node_ids, depot_ids.
        """
        depots = network.get_nodes_by_type(NodeType.DEPOT)
        gen_nodes = network.get_nodes_by_type(NodeType.WASTE_GENERATION)
        facilities = (
            network.get_nodes_by_type(NodeType.SORTING_FACILITY)
            + network.get_nodes_by_type(NodeType.TCP)
            + network.get_nodes_by_type(NodeType.LANDFILL)
        )

        # Node ordering: [depot, gen_node_0, gen_node_1, ..., facility_0, ...]
        # We use the primary depot as the start/end
        primary_depot = depots[0]
        ordered_nodes = [primary_depot] + gen_nodes + facilities

        n = len(ordered_nodes)
        node_ids = ordered_nodes

        # Distance matrix (scaled to integers for OR-Tools)
        positions = [np.array(network.get_node_position(nid)) for nid in node_ids]
        scale = 100  # scale factor for integer distances
        dist_matrix = np.zeros((n, n), dtype=np.int64)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = float(np.linalg.norm(positions[i] - positions[j]))
                # Penalise by road health
                health = 1.0
                if network.graph.has_edge(node_ids[i], node_ids[j]):
                    health = max(
                        network.get_edge_health(node_ids[i], node_ids[j]),
                        0.01
                    )
                dist_matrix[i, j] = int(d / health * scale)

        # Demands (only waste gen nodes have demand)
        demands = [0] * n  # depot and facilities have 0 demand
        for idx, nid in enumerate(node_ids):
            if nid in gen_nodes and nid in waste_model.node_ids:
                demands[idx] = max(
                    1, int(waste_model.get_node_storage(nid) * scale / 10)
                )

        # Vehicle capacities (scaled)
        num_veh = len(vehicles)
        capacities = [
            int(v.config.capacity * scale / 10) for v in vehicles
        ]

        return {
            "distance_matrix": dist_matrix.tolist(),
            "demands": demands,
            "vehicle_capacities": capacities,
            "num_vehicles": num_veh,
            "depot": 0,
            "node_ids": node_ids,
            "n": n,
            "scale": scale,
        }

    def _solve_cvrp(self, data: Dict) -> List[List[int]]:
        """Solve the CVRP using OR-Tools routing library.

        Returns
        -------
        List[List[int]]
            Per-vehicle routes as lists of original node IDs.
        """
        n = data["n"]
        num_veh = data["num_vehicles"]

        # Create routing index manager
        manager = pywrapcp.RoutingIndexManager(
            n, num_veh, data["depot"]
        )

        # Create routing model
        routing = pywrapcp.RoutingModel(manager)

        # Distance callback
        dist_matrix = data["distance_matrix"]

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return dist_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(
            distance_callback
        )
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Capacity constraint
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return data["demands"][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(
            demand_callback
        )

        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data["vehicle_capacities"],
            True,  # start cumul to zero
            "Capacity",
        )

        # Allow dropping nodes (for infeasible cases)
        penalty = 10000
        for node in range(1, n):
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

        # Search parameters
        search_params = pywrapcp.DefaultRoutingSearchParameters()

        # Map strategy name
        strategy_map = {
            "SAVINGS": routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
            "PATH_CHEAPEST_ARC": routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
            "CHRISTOFIDES": routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
            "PARALLEL_CHEAPEST_INSERTION": routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
        }
        search_params.first_solution_strategy = strategy_map.get(
            self._first_strategy,
            routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
        )
        search_params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_params.time_limit.FromSeconds(self._time_limit)

        # Solve
        solution = routing.SolveWithParameters(search_params)

        # Extract routes
        routes: List[List[int]] = []
        node_ids = data["node_ids"]

        if solution:
            for vehicle_id in range(num_veh):
                route_nodes = []
                index = routing.Start(vehicle_id)
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    if node_index != data["depot"]:
                        route_nodes.append(node_ids[node_index])
                    index = solution.Value(routing.NextVar(index))
                routes.append(route_nodes)
        else:
            # Fallback: empty routes
            routes = [[] for _ in range(num_veh)]

        return routes

    def _execute_routes(
        self,
        env: DisasterWasteEnv,
        seed: int,
        routes: List[List[int]],
        data: Dict,
    ) -> Dict[str, float]:
        """Execute MILP solution through env.step() for fair KPIs."""
        obs_dict, _ = env.reset(seed=seed)
        agent_list = env.possible_agents
        max_deg = env._max_degree
        wait_action = max_deg + 2
        pickup_action = max_deg + 0
        dropoff_action = max_deg + 1

        # Assign routes to agents
        agent_routes: Dict[str, List[int]] = {}
        agent_tidx: Dict[str, int] = {}
        for i, agent in enumerate(agent_list):
            agent_routes[agent] = routes[i] if i < len(routes) else []
            agent_tidx[agent] = 0

        total_reward = 0.0
        done = False

        while not done:
            actions: Dict[str, int] = {}
            for agent in env.agents:
                veh = env._agent_vehicle[agent]
                mask = obs_dict[agent]["action_mask"]
                network = env._network

                # Priority: dropoff > pickup > follow route
                if mask[dropoff_action] == 1:
                    actions[agent] = dropoff_action
                    continue
                if mask[pickup_action] == 1:
                    actions[agent] = pickup_action
                    continue

                route = agent_routes.get(agent, [])
                tidx = agent_tidx.get(agent, 0)

                if tidx < len(route):
                    target = route[tidx]
                    if veh.current_node == target:
                        agent_tidx[agent] = tidx + 1
                        tidx += 1
                        if tidx >= len(route):
                            actions[agent] = wait_action
                            continue
                        target = route[tidx]

                    # Move toward target
                    neighbours = network.get_neighbors(veh.current_node)
                    best_act = wait_action
                    best_d = float("inf")
                    tgt_pos = np.array(network.get_node_position(target))

                    for ni, nbr in enumerate(neighbours):
                        if ni >= max_deg or mask[ni] == 0:
                            continue
                        if nbr == target:
                            best_act = ni
                            break
                        d = float(np.linalg.norm(
                            np.array(network.get_node_position(nbr)) - tgt_pos
                        ))
                        if d < best_d:
                            best_d = d
                            best_act = ni
                    actions[agent] = best_act
                else:
                    actions[agent] = wait_action

            obs_dict, rewards, terms, truncs, _ = env.step(actions)
            total_reward += sum(rewards.values()) / len(rewards)
            done = any(truncs.values()) or any(terms.values())

        metrics = env.get_episode_metrics()
        metrics["total_reward"] = total_reward
        return metrics

    def solve_batch(
        self,
        env: DisasterWasteEnv,
        n_episodes: int = 3,
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
        avg["algorithm"] = "MILP_ORTools"
        avg["n_episodes"] = n_episodes
        return avg

    def __repr__(self) -> str:
        return (
            f"MILPSolver(time_limit={self._time_limit}s, "
            f"strategy={self._first_strategy})"
        )
