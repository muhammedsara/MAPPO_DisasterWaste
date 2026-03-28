"""
genetic_algorithm.py — Genetic Algorithm Metaheuristic for Disaster Waste VRP
===============================================================================

Population-based metaheuristic that evolves a set of route solutions
using crossover, mutation, and selection operators.  Routes are encoded
as ordered chromosomes and the multi-objective reward from the
environment (cost, time, emissions, recycling) is used as the fitness.

Chromosome encoding
--------------------
Each chromosome is a permutation of waste-generation node IDs,
partitioned among vehicles via separator genes.  Example for 3 vehicles
and 8 waste nodes::

    chromosome = [3, 7, 1 | 5, 2, 8 | 4, 6]
                 ← veh 0 →  ← veh 1 →  ← v2 →

The split is determined by a separate ``cuts`` array.

Operators
---------
- **Selection**: Tournament selection (k=3)
- **Crossover**: Order Crossover (OX) — preserves relative ordering
- **Mutation**: Swap mutation + insertion mutation
- **Elitism**: Top solutions are carried to the next generation

Fitness function
----------------
Uses the environment's multi-objective reward by executing each
candidate solution through ``env.step()`` (or approximated via a
cost model for speed).

Author  : Muhammed Şara
License : MIT
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.environment.disaster_waste_env import DisasterWasteEnv
from src.environment.network import NodeType


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GAConfig:
    """Genetic Algorithm hyper-parameters.

    Parameters
    ----------
    population_size : int
        Number of chromosomes per generation.
    n_generations : int
        Maximum number of generations.
    crossover_rate : float
        Probability of applying crossover.
    mutation_rate : float
        Probability of mutation (per gene).
    tournament_size : int
        Tournament selection pool size.
    elite_count : int
        Number of top solutions preserved via elitism.
    seed : int
        Random seed.
    """
    population_size: int = 50
    n_generations: int = 100
    crossover_rate: float = 0.85
    mutation_rate: float = 0.15
    tournament_size: int = 3
    elite_count: int = 2
    seed: int = 42


# ---------------------------------------------------------------------------
# Chromosome
# ---------------------------------------------------------------------------

@dataclass
class Chromosome:
    """Route solution encoded as a permutation + split points.

    Attributes
    ----------
    genes : List[int]
        Ordered list of waste-generation node IDs.
    cuts : List[int]
        Split points for vehicle assignment.
    fitness : float
        Fitness value (higher is better).
    """
    genes: List[int] = field(default_factory=list)
    cuts: List[int] = field(default_factory=list)
    fitness: float = float("-inf")

    def get_routes(self, n_vehicles: int) -> List[List[int]]:
        """Decode chromosome into per-vehicle routes."""
        if not self.cuts or not self.genes:
            # Equal split
            chunk = max(1, len(self.genes) // n_vehicles)
            routes = []
            for i in range(n_vehicles):
                start = i * chunk
                end = start + chunk if i < n_vehicles - 1 else len(self.genes)
                routes.append(self.genes[start:end])
            return routes

        routes = []
        prev = 0
        for c in sorted(self.cuts):
            c = min(c, len(self.genes))
            routes.append(self.genes[prev:c])
            prev = c
        routes.append(self.genes[prev:])

        # Pad to n_vehicles
        while len(routes) < n_vehicles:
            routes.append([])

        return routes[:n_vehicles]


# ---------------------------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------------------------

class GeneticAlgorithmBaseline:
    """Population-based GA for disaster waste VRP.

    Parameters
    ----------
    config : GAConfig, optional
        GA hyper-parameters.

    Examples
    --------
    >>> ga = GeneticAlgorithmBaseline()
    >>> metrics = ga.solve(env)
    """

    def __init__(self, config: Optional[GAConfig] = None) -> None:
        self._config = config or GAConfig()
        self._rng = np.random.default_rng(self._config.seed)

    def solve(
        self,
        env: DisasterWasteEnv,
        seed: int = 42,
    ) -> Dict[str, float]:
        """Run the GA and execute the best solution via env.step().

        Parameters
        ----------
        env : DisasterWasteEnv
        seed : int

        Returns
        -------
        dict
            Episode KPIs.
        """
        cfg = self._config
        self._rng = np.random.default_rng(seed)

        # Reset env once to get structure
        obs_dict, _ = env.reset(seed=seed)
        network = env._network
        waste_model = env._waste_model
        n_vehicles = len(env.possible_agents)

        # Waste generation nodes with storage
        gen_nodes = [
            nid for nid in network.get_nodes_by_type(NodeType.WASTE_GENERATION)
            if (nid in waste_model.node_ids and
                waste_model.get_node_storage(nid) > 0.1)
        ]

        if not gen_nodes:
            # No waste to collect — run empty episode
            return self._run_episode_with_routes(env, seed, [[] for _ in range(n_vehicles)])

        # Facilities for dropoff
        facilities = (
            network.get_nodes_by_type(NodeType.SORTING_FACILITY)
            + network.get_nodes_by_type(NodeType.TCP)
            + network.get_nodes_by_type(NodeType.LANDFILL)
        )

        # --- Initialise population ---
        population = self._init_population(gen_nodes, n_vehicles, cfg.population_size)

        # --- Evaluate initial population ---
        for chromo in population:
            chromo.fitness = self._evaluate_fitness(
                chromo, env, seed, n_vehicles, network, facilities
            )

        # --- Evolution loop ---
        for gen in range(cfg.n_generations):
            # Sort by fitness (descending)
            population.sort(key=lambda c: c.fitness, reverse=True)

            # Elitism
            new_pop = [copy.deepcopy(population[i]) for i in range(cfg.elite_count)]

            # Fill rest with offspring
            while len(new_pop) < cfg.population_size:
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)

                if self._rng.random() < cfg.crossover_rate:
                    child1, child2 = self._order_crossover(parent1, parent2)
                else:
                    child1 = copy.deepcopy(parent1)
                    child2 = copy.deepcopy(parent2)

                self._mutate(child1)
                self._mutate(child2)

                child1.fitness = self._evaluate_fitness(
                    child1, env, seed, n_vehicles, network, facilities
                )
                child2.fitness = self._evaluate_fitness(
                    child2, env, seed, n_vehicles, network, facilities
                )

                new_pop.extend([child1, child2])

            population = new_pop[:cfg.population_size]

        # --- Execute best solution ---
        population.sort(key=lambda c: c.fitness, reverse=True)
        best = population[0]
        best_routes = best.get_routes(n_vehicles)

        # Add nearest facility to end of each route
        final_routes = []
        for route in best_routes:
            if route and facilities:
                last_pos = np.array(network.get_node_position(route[-1]))
                best_fac = min(
                    facilities,
                    key=lambda f: float(np.linalg.norm(
                        last_pos - np.array(network.get_node_position(f))
                    ))
                )
                route.append(best_fac)
            final_routes.append(route)

        return self._run_episode_with_routes(env, seed, final_routes)

    # ==================================================================
    # Population initialisation
    # ==================================================================

    def _init_population(
        self,
        gen_nodes: List[int],
        n_vehicles: int,
        pop_size: int,
    ) -> List[Chromosome]:
        """Create initial random population."""
        population = []
        for _ in range(pop_size):
            perm = list(self._rng.permutation(gen_nodes))
            # Random cut points
            n_cuts = max(1, n_vehicles - 1)
            cut_positions = sorted(
                self._rng.choice(
                    range(1, max(2, len(perm))),
                    size=min(n_cuts, max(1, len(perm) - 1)),
                    replace=False,
                ).tolist()
            )
            population.append(Chromosome(genes=perm, cuts=cut_positions))
        return population

    # ==================================================================
    # Genetic operators
    # ==================================================================

    def _tournament_select(self, population: List[Chromosome]) -> Chromosome:
        """Tournament selection."""
        k = min(self._config.tournament_size, len(population))
        indices = self._rng.choice(len(population), size=k, replace=False)
        candidates = [population[i] for i in indices]
        return max(candidates, key=lambda c: c.fitness)

    def _order_crossover(
        self, p1: Chromosome, p2: Chromosome
    ) -> Tuple[Chromosome, Chromosome]:
        """Order Crossover (OX) — preserves relative ordering.

        1. Select a random segment from parent1.
        2. Copy it to child1 at the same positions.
        3. Fill remaining positions with genes from parent2 (in order),
           skipping those already in the segment.
        """
        n = len(p1.genes)
        if n < 2:
            return copy.deepcopy(p1), copy.deepcopy(p2)

        # Random segment
        cx1, cx2 = sorted(self._rng.choice(n, size=2, replace=False))

        child1_genes = self._ox_child(p1.genes, p2.genes, cx1, cx2)
        child2_genes = self._ox_child(p2.genes, p1.genes, cx1, cx2)

        # Inherit cut points from fitter parent
        child1 = Chromosome(genes=child1_genes, cuts=list(p1.cuts))
        child2 = Chromosome(genes=child2_genes, cuts=list(p2.cuts))

        return child1, child2

    @staticmethod
    def _ox_child(
        p1_genes: List[int], p2_genes: List[int], cx1: int, cx2: int
    ) -> List[int]:
        """Create a single OX child."""
        n = len(p1_genes)
        child = [None] * n

        # Copy segment from p1
        segment = set()
        for i in range(cx1, cx2 + 1):
            child[i] = p1_genes[i]
            segment.add(p1_genes[i])

        # Fill from p2
        p2_filtered = [g for g in p2_genes if g not in segment]
        fill_idx = 0
        for i in range(n):
            if child[i] is None:
                child[i] = p2_filtered[fill_idx]
                fill_idx += 1

        return child

    def _mutate(self, chromo: Chromosome) -> None:
        """Apply swap and insertion mutations."""
        n = len(chromo.genes)
        if n < 2:
            return

        # Swap mutation
        if self._rng.random() < self._config.mutation_rate:
            i, j = self._rng.choice(n, size=2, replace=False)
            chromo.genes[i], chromo.genes[j] = chromo.genes[j], chromo.genes[i]

        # Insertion mutation
        if self._rng.random() < self._config.mutation_rate * 0.5:
            i = int(self._rng.integers(0, n))
            j = int(self._rng.integers(0, n))
            gene = chromo.genes.pop(i)
            chromo.genes.insert(j, gene)

        # Mutate cut points
        if chromo.cuts and self._rng.random() < self._config.mutation_rate * 0.3:
            idx = int(self._rng.integers(0, len(chromo.cuts)))
            chromo.cuts[idx] = int(self._rng.integers(1, max(2, n)))
            chromo.cuts.sort()

    # ==================================================================
    # Fitness evaluation
    # ==================================================================

    def _evaluate_fitness(
        self,
        chromo: Chromosome,
        env: DisasterWasteEnv,
        seed: int,
        n_vehicles: int,
        network,
        facilities: List[int],
    ) -> float:
        """Approximate fitness using a cost model (fast, no env.step()).

        For full population evaluation, running env.step() per
        chromosome is too slow.  We use a distance-based cost model:

        fitness = −(total_distance + demand_penalty)

        The best chromosome is then evaluated through env.step() in
        the ``solve()`` method.
        """
        routes = chromo.get_routes(n_vehicles)
        depots = network.get_nodes_by_type(NodeType.DEPOT)
        primary_depot = depots[0] if depots else 0

        total_cost = 0.0
        demand_penalty = 0.0

        for route in routes:
            if not route:
                continue

            # depot → first node
            prev_pos = np.array(network.get_node_position(primary_depot))
            route_dist = 0.0

            for nid in route:
                nid_pos = np.array(network.get_node_position(nid))
                d = float(np.linalg.norm(prev_pos - nid_pos))

                # Health penalty
                health = 1.0
                if network.graph.has_edge(primary_depot, nid):
                    health = max(network.get_edge_health(primary_depot, nid), 0.01)

                route_dist += d / health
                prev_pos = nid_pos

            # last node → nearest facility
            if route and facilities:
                last_pos = np.array(network.get_node_position(route[-1]))
                min_fac_dist = min(
                    float(np.linalg.norm(
                        last_pos - np.array(network.get_node_position(f))
                    ))
                    for f in facilities
                )
                route_dist += min_fac_dist

            total_cost += route_dist

        # Penalty for unserved nodes (nodes not in any route)
        all_served = set()
        for route in routes:
            all_served.update(route)
        gen_nodes = set(network.get_nodes_by_type(NodeType.WASTE_GENERATION))
        unserved = gen_nodes - all_served
        demand_penalty = len(unserved) * 100.0

        return -(total_cost + demand_penalty)

    # ==================================================================
    # Route execution through env
    # ==================================================================

    def _run_episode_with_routes(
        self,
        env: DisasterWasteEnv,
        seed: int,
        routes: List[List[int]],
    ) -> Dict[str, float]:
        """Execute routes through the environment for KPI computation."""
        obs_dict, _ = env.reset(seed=seed)
        agent_list = env.possible_agents
        max_deg = env._max_degree
        wait_action = max_deg + 2
        pickup_action = max_deg + 0
        dropoff_action = max_deg + 1

        # Assign routes
        agent_target_idx: Dict[str, int] = {}
        agent_routes: Dict[str, List[int]] = {}
        for i, agent in enumerate(agent_list):
            agent_routes[agent] = routes[i] if i < len(routes) else []
            agent_target_idx[agent] = 0

        total_reward = 0.0
        done = False

        while not done:
            actions: Dict[str, int] = {}
            for agent in env.agents:
                veh = env._agent_vehicle[agent]
                mask = obs_dict[agent]["action_mask"]
                network = env._network

                # Priority: dropoff > pickup > follow route > wait
                if mask[dropoff_action] == 1:
                    actions[agent] = dropoff_action
                    continue
                if mask[pickup_action] == 1:
                    actions[agent] = pickup_action
                    continue

                route = agent_routes.get(agent, [])
                tidx = agent_target_idx.get(agent, 0)

                if tidx < len(route):
                    target = route[tidx]
                    if veh.current_node == target:
                        agent_target_idx[agent] = tidx + 1
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
        metrics["algorithm"] = "GeneticAlgorithm"
        return metrics

    def solve_batch(
        self,
        env: DisasterWasteEnv,
        n_episodes: int = 5,
        seed: int = 42,
    ) -> Dict[str, float]:
        """Run multiple episodes and average."""
        all_metrics: List[Dict] = []
        for ep in range(n_episodes):
            m = self.solve(env, seed=seed + ep)
            all_metrics.append(m)

        avg = {}
        keys = [k for k in all_metrics[0] if isinstance(all_metrics[0][k], (int, float))]
        for k in keys:
            avg[f"mean_{k}"] = float(np.mean([m[k] for m in all_metrics]))
            avg[f"std_{k}"] = float(np.std([m[k] for m in all_metrics]))
        avg["algorithm"] = "GeneticAlgorithm"
        avg["n_episodes"] = n_episodes
        return avg

    def __repr__(self) -> str:
        cfg = self._config
        return (
            f"GeneticAlgorithmBaseline(pop={cfg.population_size}, "
            f"gen={cfg.n_generations}, cx={cfg.crossover_rate}, "
            f"mut={cfg.mutation_rate})"
        )
