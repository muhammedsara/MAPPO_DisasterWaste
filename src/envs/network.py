"""
network.py — Dynamic Disaster Road Network Model
==================================================

This module implements a NetworkX-based directed graph representing the
post-disaster transportation infrastructure. Key capabilities:

    1. **Node taxonomy**: Waste generation sites, temporary collection points
       (TCPs), sorting/recycling facilities, and landfills/final disposal sites.
    2. **Edge dynamics**: Each road segment carries time-varying health
       coefficients that degrade via a Poisson damage process and recover
       through an exponential repair process.
    3. **Travel-time computation**: Effective travel time on edge (i,j) is
       inversely proportional to road health: τ_ij(t) = d_ij / (v_free · φ_ij(t)).

Mathematical references (see implementation_plan.md §3.1.1):
    P(damage_ij | Δt) = 1 − exp(−λ_damage · Δt)
    φ_ij(t+1) = φ_ij(t) · (1 − Δφ)        # on damage event
    φ_ij(t+1) = min(1, φ_ij(t) + μ_repair · Δt)  # repair recovery

Author  : Muhammed Şara
License : MIT
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Set

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Enumerations & data classes
# ---------------------------------------------------------------------------

class NodeType(Enum):
    """Taxonomy of nodes in the disaster waste logistics network."""
    WASTE_GENERATION = auto()   # Demolition / disaster-affected area
    TCP = auto()                # Temporary Collection Point
    SORTING_FACILITY = auto()   # Sorting / recycling facility
    LANDFILL = auto()           # Final disposal site
    DEPOT = auto()              # Vehicle depot (starting point)


@dataclass
class NodeAttributes:
    """Attributes stored on each graph node.

    Parameters
    ----------
    node_type : NodeType
        Functional role of the node.
    position : Tuple[float, float]
        (x, y) coordinates in a 2-D Euclidean plane.
    capacity : float
        Processing capacity per time step (tonnes).
        For WASTE_GENERATION nodes this represents storage limit.
    recycling_rates : Dict[str, float]
        Maps waste type → recovery proportion.  Only meaningful for
        SORTING_FACILITY nodes; empty dict for others.
    """
    node_type: NodeType
    position: Tuple[float, float]
    capacity: float = 0.0
    recycling_rates: Dict[str, float] = field(default_factory=dict)


@dataclass
class EdgeAttributes:
    """Attributes stored on each directed edge (road segment).

    Parameters
    ----------
    distance : float
        Length of the road segment (km).
    base_travel_time : float
        Free-flow travel time (hours) under full health.
    unit_cost : float
        Monetary cost per km of traversal ($/km).
    unit_emission : float
        Carbon emission per km (kg CO₂/km).
    health : float
        Current road health coefficient φ ∈ [0, 1].
        0 = completely blocked, 1 = fully operational.
    """
    distance: float
    base_travel_time: float
    unit_cost: float = 1.0
    unit_emission: float = 0.5
    health: float = 1.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DisasterNetwork:
    """Dynamic directed graph model for post-disaster road networks.

    The network is built on top of ``networkx.DiGraph`` and augmented with
    Poisson-driven road degradation and exponential repair mechanics.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility (default 42).

    Examples
    --------
    >>> net = DisasterNetwork(seed=0)
    >>> net.add_node(0, NodeAttributes(NodeType.DEPOT, (0, 0)))
    >>> net.add_node(1, NodeAttributes(NodeType.WASTE_GENERATION, (5, 5)))
    >>> net.add_edge(0, 1, EdgeAttributes(distance=7.07, base_travel_time=0.5))
    >>> net.get_travel_time(0, 1)
    0.5
    """

    # Default Poisson damage parameters
    DEFAULT_LAMBDA_DAMAGE: float = 0.05    # damage events per time step
    DEFAULT_DAMAGE_SEVERITY: float = 0.3   # Δφ drop on damage event
    DEFAULT_REPAIR_RATE: float = 0.02      # μ_repair recovery per time step
    DEFAULT_MIN_HEALTH: float = 0.05       # φ never drops below this (passable but slow)

    def __init__(self, seed: int = 42) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._current_time: int = 0

        # Poisson damage / repair configuration (can be overridden per edge)
        self.lambda_damage: float = self.DEFAULT_LAMBDA_DAMAGE
        self.damage_severity: float = self.DEFAULT_DAMAGE_SEVERITY
        self.repair_rate: float = self.DEFAULT_REPAIR_RATE
        self.min_health: float = self.DEFAULT_MIN_HEALTH

        # History log: list of (time, edge, old_health, new_health, event_type)
        self._damage_log: List[Tuple[int, Tuple[int, int], float, float, str]] = []

    # ------------------------------------------------------------------
    # Graph construction helpers
    # ------------------------------------------------------------------

    def add_node(self, node_id: int, attrs: NodeAttributes) -> None:
        """Add a node with its typed attributes to the network.

        Parameters
        ----------
        node_id : int
            Unique integer identifier for the node.
        attrs : NodeAttributes
            Data container holding the node's role, position, capacity, etc.
        """
        self._graph.add_node(
            node_id,
            node_type=attrs.node_type,
            position=attrs.position,
            capacity=attrs.capacity,
            recycling_rates=attrs.recycling_rates,
        )

    def add_edge(self, u: int, v: int, attrs: EdgeAttributes) -> None:
        """Add a directed edge (road segment) between two existing nodes.

        If both directions are needed, call this method twice with reversed
        arguments (the network is a DiGraph, not a Graph).

        Parameters
        ----------
        u, v : int
            Source and target node IDs.
        attrs : EdgeAttributes
            Data container with distance, cost, emission, and health values.
        """
        self._graph.add_edge(
            u, v,
            distance=attrs.distance,
            base_travel_time=attrs.base_travel_time,
            unit_cost=attrs.unit_cost,
            unit_emission=attrs.unit_emission,
            health=attrs.health,
        )

    def add_bidirectional_edge(self, u: int, v: int, attrs: EdgeAttributes) -> None:
        """Convenience method: add edges in both directions with identical attributes."""
        self.add_edge(u, v, attrs)
        self.add_edge(v, u, copy.copy(attrs))

    # ------------------------------------------------------------------
    # Bulk network generation
    # ------------------------------------------------------------------

    def generate_random_network(
        self,
        n_generation: int = 15,
        n_tcp: int = 5,
        n_sorting: int = 3,
        n_landfill: int = 2,
        n_depot: int = 2,
        area_size: float = 100.0,
        connectivity: float = 0.35,
        speed_kmh: float = 40.0,
    ) -> None:
        """Procedurally generate a random disaster logistics network.

        Nodes are placed uniformly in a ``[0, area_size]²`` square.  Edges are
        created for node pairs whose Euclidean distance falls below a
        connectivity-driven threshold, ensuring the graph is connected.

        Parameters
        ----------
        n_generation : int
            Number of waste generation / demolition sites.
        n_tcp : int
            Number of temporary collection points.
        n_sorting : int
            Number of sorting / recycling facilities.
        n_landfill : int
            Number of landfill / final disposal sites.
        n_depot : int
            Number of vehicle depots.
        area_size : float
            Side length of the square deployment area (km).
        connectivity : float
            Probability of edge creation between any pair with distance below
            the 60th-percentile of all pairwise distances.
        speed_kmh : float
            Free-flow average speed used to compute base travel times.

        Notes
        -----
        After generation, the graph is verified for strong-connectivity.
        If disconnected components exist, bridge edges are added automatically.
        """
        self._graph.clear()

        # --- 1. Create nodes ---
        node_counts = [
            (NodeType.WASTE_GENERATION, n_generation),
            (NodeType.TCP, n_tcp),
            (NodeType.SORTING_FACILITY, n_sorting),
            (NodeType.LANDFILL, n_landfill),
            (NodeType.DEPOT, n_depot),
        ]

        waste_types = ["concrete", "metal", "wood", "mixed", "hazardous"]
        node_id = 0
        for ntype, count in node_counts:
            for _ in range(count):
                pos = tuple(self._rng.uniform(0, area_size, size=2).tolist())

                # Assign capacity by node type
                if ntype == NodeType.WASTE_GENERATION:
                    cap = float(self._rng.uniform(50, 200))  # tonnes storage
                elif ntype == NodeType.TCP:
                    cap = float(self._rng.uniform(100, 400))
                elif ntype == NodeType.SORTING_FACILITY:
                    cap = float(self._rng.uniform(80, 250))
                elif ntype == NodeType.LANDFILL:
                    cap = float(self._rng.uniform(500, 2000))
                else:  # DEPOT
                    cap = 0.0

                # Recycling rates only for sorting facilities
                rec: Dict[str, float] = {}
                if ntype == NodeType.SORTING_FACILITY:
                    rec = {wt: float(self._rng.uniform(0.3, 0.9)) for wt in waste_types}

                self.add_node(node_id, NodeAttributes(
                    node_type=ntype,
                    position=pos,
                    capacity=cap,
                    recycling_rates=rec,
                ))
                node_id += 1

        # --- 2. Create edges based on distance threshold ---
        all_nodes = list(self._graph.nodes)
        positions = np.array([self._graph.nodes[n]["position"] for n in all_nodes])
        n_nodes = len(all_nodes)

        # Compute pairwise Euclidean distances
        # dist_matrix[i, j] = ||pos_i − pos_j||₂
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

        # Distance threshold: 60th percentile of non-zero distances
        non_zero_dists = dist_matrix[dist_matrix > 0]
        threshold = float(np.percentile(non_zero_dists, 60))

        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    continue
                d = dist_matrix[i, j]
                if d <= threshold and self._rng.random() < connectivity:
                    travel_time = d / speed_kmh
                    self.add_edge(all_nodes[i], all_nodes[j], EdgeAttributes(
                        distance=round(d, 2),
                        base_travel_time=round(travel_time, 4),
                        unit_cost=round(float(self._rng.uniform(0.5, 2.0)), 2),
                        unit_emission=round(float(self._rng.uniform(0.3, 1.2)), 2),
                        health=1.0,
                    ))

        # --- 3. Ensure strong connectivity ---
        self._ensure_connectivity(speed_kmh)

    def _ensure_connectivity(self, speed_kmh: float) -> None:
        """Add bridge edges if the graph is not strongly connected.

        Uses ``nx.strongly_connected_components`` to identify isolated
        subgraphs and connects them via their closest node pairs.
        """
        components = list(nx.strongly_connected_components(self._graph))
        if len(components) <= 1:
            return

        # Sort components by size (largest first) and iteratively bridge
        components.sort(key=len, reverse=True)
        main_component = components[0]

        for comp in components[1:]:
            min_dist = float("inf")
            best_pair: Optional[Tuple[int, int]] = None

            for u in main_component:
                pos_u = np.array(self._graph.nodes[u]["position"])
                for v in comp:
                    pos_v = np.array(self._graph.nodes[v]["position"])
                    d = float(np.linalg.norm(pos_u - pos_v))
                    if d < min_dist:
                        min_dist = d
                        best_pair = (u, v)

            if best_pair is not None:
                u, v = best_pair
                travel_time = min_dist / speed_kmh
                attrs = EdgeAttributes(
                    distance=round(min_dist, 2),
                    base_travel_time=round(travel_time, 4),
                    unit_cost=1.0,
                    unit_emission=0.5,
                    health=1.0,
                )
                self.add_bidirectional_edge(u, v, attrs)
                main_component = main_component.union(comp)

    # ------------------------------------------------------------------
    # Road damage & repair dynamics
    # ------------------------------------------------------------------

    def apply_damage_step(self, delta_t: float = 1.0) -> List[Tuple[int, int]]:
        """Advance the Poisson road-damage process by one time step.

        For every edge in the network, a damage event occurs with probability:

            P(damage | Δt) = 1 − exp(−λ_damage · Δt)

        If an event fires, the road health is reduced:

            φ_ij ← max(φ_min, φ_ij · (1 − Δφ))

        Parameters
        ----------
        delta_t : float
            Duration of the time step (default 1.0).

        Returns
        -------
        damaged_edges : List[Tuple[int, int]]
            List of (u, v) edges that sustained new damage in this step.
        """
        damage_prob = 1.0 - np.exp(-self.lambda_damage * delta_t)
        damaged_edges: List[Tuple[int, int]] = []

        for u, v, data in self._graph.edges(data=True):
            if self._rng.random() < damage_prob:
                old_health = data["health"]
                new_health = max(self.min_health, old_health * (1.0 - self.damage_severity))
                data["health"] = new_health
                damaged_edges.append((u, v))
                self._damage_log.append(
                    (self._current_time, (u, v), old_health, new_health, "damage")
                )

        return damaged_edges

    def apply_repair_step(self, delta_t: float = 1.0) -> List[Tuple[int, int]]:
        """Advance the exponential repair process by one time step.

        Every edge with health below 1.0 recovers linearly:

            φ_ij ← min(1.0, φ_ij + μ_repair · Δt)

        Parameters
        ----------
        delta_t : float
            Duration of the time step (default 1.0).

        Returns
        -------
        repaired_edges : List[Tuple[int, int]]
            List of edges whose health increased in this step.
        """
        repaired_edges: List[Tuple[int, int]] = []

        for u, v, data in self._graph.edges(data=True):
            if data["health"] < 1.0:
                old_health = data["health"]
                new_health = min(1.0, old_health + self.repair_rate * delta_t)
                data["health"] = new_health
                repaired_edges.append((u, v))
                self._damage_log.append(
                    (self._current_time, (u, v), old_health, new_health, "repair")
                )

        return repaired_edges

    def step_dynamics(self, delta_t: float = 1.0) -> Dict[str, List[Tuple[int, int]]]:
        """Execute one full time step of network dynamics (damage + repair).

        Parameters
        ----------
        delta_t : float
            Duration of the time step.

        Returns
        -------
        dict with keys ``"damaged"`` and ``"repaired"``, each mapping to a
        list of affected ``(u, v)`` edges.
        """
        damaged = self.apply_damage_step(delta_t)
        repaired = self.apply_repair_step(delta_t)
        self._current_time += 1
        return {"damaged": damaged, "repaired": repaired}

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_travel_time(self, u: int, v: int) -> float:
        """Effective travel time on edge (u, v) given current health.

        Computed as:
            τ_eff = base_travel_time / φ_ij

        If the edge is fully blocked (φ → 0), travel time approaches
        base / φ_min, acting as a soft penalty rather than hard infinity.

        Parameters
        ----------
        u, v : int
            Source and target node IDs.

        Returns
        -------
        float
            Effective travel time in hours.

        Raises
        ------
        nx.NetworkXError
            If the edge does not exist.
        """
        data = self._graph.edges[u, v]
        health = max(data["health"], self.min_health)
        return data["base_travel_time"] / health

    def get_traversal_cost(self, u: int, v: int) -> float:
        """Monetary traversal cost for edge (u, v).

        Returns
        -------
        float
            cost = unit_cost × distance
        """
        data = self._graph.edges[u, v]
        return data["unit_cost"] * data["distance"]

    def get_traversal_emission(self, u: int, v: int) -> float:
        """Carbon emission for traversing edge (u, v).

        Returns
        -------
        float
            emission = unit_emission × distance  (kg CO₂)
        """
        data = self._graph.edges[u, v]
        return data["unit_emission"] * data["distance"]

    def get_edge_health(self, u: int, v: int) -> float:
        """Return current health coefficient φ_ij ∈ [0, 1] for edge (u, v)."""
        return float(self._graph.edges[u, v]["health"])

    def get_node_type(self, node_id: int) -> NodeType:
        """Return the NodeType enum of the given node."""
        return self._graph.nodes[node_id]["node_type"]

    def get_node_position(self, node_id: int) -> Tuple[float, float]:
        """Return (x, y) coordinates of the node."""
        return tuple(self._graph.nodes[node_id]["position"])

    def get_neighbors(self, node_id: int) -> List[int]:
        """Return successor node IDs reachable from ``node_id``."""
        return list(self._graph.successors(node_id))

    def get_reachable_neighbors(
        self, node_id: int, health_threshold: float = 0.1
    ) -> List[int]:
        """Return neighbors reachable via edges with health above a threshold.

        Parameters
        ----------
        node_id : int
            Current node.
        health_threshold : float
            Minimum health required to consider the road passable.

        Returns
        -------
        List[int]
            Node IDs reachable from ``node_id`` through healthy-enough roads.
        """
        reachable: List[int] = []
        for v in self._graph.successors(node_id):
            if self._graph.edges[node_id, v]["health"] >= health_threshold:
                reachable.append(v)
        return reachable

    def get_nodes_by_type(self, node_type: NodeType) -> List[int]:
        """Return all node IDs matching a given NodeType."""
        return [
            n for n, d in self._graph.nodes(data=True)
            if d["node_type"] == node_type
        ]

    def shortest_path(
        self, source: int, target: int, weight: str = "effective_time"
    ) -> Tuple[List[int], float]:
        """Compute the shortest path using Dijkstra with dynamic weights.

        Parameters
        ----------
        source, target : int
            Start and end node IDs.
        weight : str
            If ``"effective_time"``, uses health-adjusted travel time.
            If ``"distance"``, uses raw edge distance.

        Returns
        -------
        (path, total_weight) : Tuple[List[int], float]
            Ordered list of node IDs and the total weight of the path.

        Raises
        ------
        nx.NetworkXNoPath
            If no path exists between source and target.
        """
        if weight == "effective_time":
            # Build a temporary weight attribute
            for u, v, data in self._graph.edges(data=True):
                health = max(data["health"], self.min_health)
                data["_eff_time"] = data["base_travel_time"] / health
            path = nx.shortest_path(self._graph, source, target, weight="_eff_time")
            total = sum(
                self._graph.edges[path[i], path[i + 1]]["_eff_time"]
                for i in range(len(path) - 1)
            )
        else:
            path = nx.shortest_path(self._graph, source, target, weight=weight)
            total = sum(
                self._graph.edges[path[i], path[i + 1]][weight]
                for i in range(len(path) - 1)
            )
        return path, round(total, 4)

    # ------------------------------------------------------------------
    # State vectors (for RL observations)
    # ------------------------------------------------------------------

    def get_edge_health_vector(self) -> np.ndarray:
        """Return a 1-D array of all edge health values.

        The ordering follows ``self._graph.edges`` iteration order and is
        deterministic for a fixed graph topology.

        Returns
        -------
        np.ndarray of shape (|E|,)
        """
        return np.array([d["health"] for _, _, d in self._graph.edges(data=True)])

    def get_adjacency_with_health(self) -> np.ndarray:
        """Return a weighted adjacency matrix where weights are health values.

        Returns
        -------
        np.ndarray of shape (|V|, |V|)
            Entry (i, j) = φ_ij if edge exists, else 0.
        """
        node_list = sorted(self._graph.nodes)
        n = len(node_list)
        idx_map = {node: i for i, node in enumerate(node_list)}
        adj = np.zeros((n, n), dtype=np.float32)
        for u, v, data in self._graph.edges(data=True):
            adj[idx_map[u], idx_map[v]] = data["health"]
        return adj

    def get_local_health_vector(self, node_id: int) -> np.ndarray:
        """Return health values of all outgoing edges from a single node.

        Used for constructing the local observation vector of an agent.

        Parameters
        ----------
        node_id : int
            The focal node.

        Returns
        -------
        np.ndarray of shape (degree_out,)
        """
        return np.array([
            self._graph.edges[node_id, v]["health"]
            for v in self._graph.successors(node_id)
        ])

    # ------------------------------------------------------------------
    # Snapshot & reset
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict:
        """Return a serialisable snapshot of the full network state.

        Useful for checkpointing and analysis.
        """
        return {
            "time": self._current_time,
            "nodes": dict(self._graph.nodes(data=True)),
            "edges": {
                (u, v): dict(data) for u, v, data in self._graph.edges(data=True)
            },
        }

    def reset_health(self) -> None:
        """Reset all edge health values to 1.0 (pristine state)."""
        for _, _, data in self._graph.edges(data=True):
            data["health"] = 1.0
        self._current_time = 0
        self._damage_log.clear()

    def reset(self, seed: Optional[int] = None) -> None:
        """Full reset: restore health and optionally re-seed the RNG."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.reset_health()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def graph(self) -> nx.DiGraph:
        """Direct access to the underlying NetworkX DiGraph."""
        return self._graph

    @property
    def num_nodes(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._graph.number_of_edges()

    @property
    def current_time(self) -> int:
        return self._current_time

    @property
    def damage_log(self) -> List[Tuple[int, Tuple[int, int], float, float, str]]:
        """Full history of damage and repair events."""
        return self._damage_log

    @property
    def average_health(self) -> float:
        """Mean health across all edges."""
        healths = self.get_edge_health_vector()
        return float(np.mean(healths)) if len(healths) > 0 else 1.0

    def __repr__(self) -> str:
        return (
            f"DisasterNetwork(nodes={self.num_nodes}, edges={self.num_edges}, "
            f"t={self._current_time}, avg_health={self.average_health:.3f})"
        )
