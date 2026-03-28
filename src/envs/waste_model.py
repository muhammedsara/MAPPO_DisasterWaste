"""
waste_model.py — Stochastic Time-Varying Waste Generation Engine
=================================================================

This module models post-disaster waste generation as a spatio-temporal
stochastic process.  Each waste generation node produces debris quantities
that follow a **Log-Normal distribution** whose mean decays exponentially
over time (modelling the initial surge of demolition waste that tapers off
as recovery progresses).

Mathematical foundation (see implementation_plan.md §3.1.2):

    Total waste at node i, time t:
        W_i(t) ~ LogNormal(μ_i(t), σ_i²)

    Time-decaying mean:
        μ_i(t) = μ_i⁰ · exp(−α_i · t) + μ_i^base

    Per waste-type breakdown:
        W_i^w(t) = ρ_i^w · W_i(t),   Σ_w ρ_i^w = 1

    Waste types  W = {concrete, metal, wood, mixed, hazardous}

Author  : Muhammed Şara
License : MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WASTE_TYPES: List[str] = ["concrete", "metal", "wood", "mixed", "hazardous"]
"""Canonical ordering of waste categories used throughout the system."""

# Default waste-type proportions for a typical earthquake demolition site
# Source: adapted from Brown et al. (2011), Disaster Waste Management Guidelines
DEFAULT_WASTE_PROPORTIONS: Dict[str, float] = {
    "concrete": 0.45,
    "metal":    0.15,
    "wood":     0.12,
    "mixed":    0.20,
    "hazardous": 0.08,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WasteNodeConfig:
    """Configuration for a single waste generation node.

    Parameters
    ----------
    node_id : int
        Identifier matching the corresponding node in ``DisasterNetwork``.
    mu_initial : float
        Initial log-mean of waste generation μ_i⁰ (tonnes).  This governs
        the magnitude of the peak waste output immediately after the disaster.
    mu_base : float
        Baseline log-mean μ_i^base that persists even after the exponential
        decay has plateaued (chronic low-level debris generation).
    sigma : float
        Standard deviation of the log-normal distribution σ_i.
        Higher values introduce more uncertainty in daily output.
    decay_rate : float
        Exponential decay coefficient α_i (per time step).
        Typical range: 0.01 (slow tapering) – 0.1 (rapid tapering).
    waste_proportions : Dict[str, float]
        Fraction of total waste belonging to each type ρ_i^w.
        Must sum to 1.0.
    storage : float
        Current accumulated (uncollected) waste at this node (tonnes).
    """
    node_id: int
    mu_initial: float = 5.0
    mu_base: float = 1.0
    sigma: float = 0.4
    decay_rate: float = 0.05
    waste_proportions: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_WASTE_PROPORTIONS)
    )
    storage: float = 0.0

    def __post_init__(self) -> None:
        """Validate that waste proportions sum to 1.0 (tolerance 1e-4)."""
        total = sum(self.waste_proportions.values())
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"Waste proportions for node {self.node_id} sum to {total:.4f}; "
                f"expected 1.0. Proportions: {self.waste_proportions}"
            )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class WasteGenerationModel:
    """Stochastic, time-varying waste generation engine.

    This class manages a collection of ``WasteNodeConfig`` objects and
    advances the waste generation process at each simulation time step.
    Each step:
        1. Computes the time-dependent log-mean μ_i(t).
        2. Draws total waste from LogNormal(μ_i(t), σ_i²).
        3. Breaks the total into per-type quantities via fixed proportions.
        4. Accumulates uncollected waste in each node's storage.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility (default 42).

    Examples
    --------
    >>> model = WasteGenerationModel(seed=0)
    >>> model.add_node(WasteNodeConfig(node_id=1, mu_initial=6.0, sigma=0.3))
    >>> generated = model.step(t=0)
    >>> print(generated[1])  # dict: waste_type -> quantity
    """

    def __init__(self, seed: int = 42) -> None:
        self._nodes: Dict[int, WasteNodeConfig] = {}
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._current_time: int = 0

        # History: list of (time, node_id, total_waste, per_type_dict)
        self._generation_log: List[Tuple[int, int, float, Dict[str, float]]] = []

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_node(self, config: WasteNodeConfig) -> None:
        """Register a waste generation node.

        Parameters
        ----------
        config : WasteNodeConfig
            Configuration object for the node.

        Raises
        ------
        ValueError
            If a node with the same ``node_id`` already exists.
        """
        if config.node_id in self._nodes:
            raise ValueError(
                f"Node {config.node_id} already registered in waste model."
            )
        self._nodes[config.node_id] = config

    def add_nodes_batch(self, configs: List[WasteNodeConfig]) -> None:
        """Register multiple nodes at once."""
        for cfg in configs:
            self.add_node(cfg)

    def remove_node(self, node_id: int) -> None:
        """Remove a waste generation node from the model."""
        self._nodes.pop(node_id, None)

    # ------------------------------------------------------------------
    # Core generation logic
    # ------------------------------------------------------------------

    def _compute_mu(self, config: WasteNodeConfig, t: int) -> float:
        """Compute the time-dependent log-mean μ_i(t).

        Formula:
            μ_i(t) = μ_i⁰ · exp(−α_i · t) + μ_i^base

        The exponential term captures the initial demolition surge that
        decays over time, while the base term ensures a minimum chronic
        generation rate persists indefinitely.

        Parameters
        ----------
        config : WasteNodeConfig
            Node-specific parameters.
        t : int
            Current time step.

        Returns
        -------
        float
            Log-mean for the Log-Normal draw at time t.
        """
        return config.mu_initial * np.exp(-config.decay_rate * t) + config.mu_base

    def _sample_waste(self, mu_t: float, sigma: float) -> float:
        """Draw a single total-waste sample from the Log-Normal distribution.

        Log-Normal parametrisation (NumPy convention):
            X = exp(N(μ, σ²))
        where μ and σ are the mean and std of the **underlying normal**
        distribution.  The expected value of X is exp(μ + σ²/2).

        Parameters
        ----------
        mu_t : float
            Time-dependent log-mean.
        sigma : float
            Log-standard deviation.

        Returns
        -------
        float
            Non-negative waste quantity (tonnes).
        """
        return float(self._rng.lognormal(mean=mu_t, sigma=sigma))

    def generate_for_node(
        self, node_id: int, t: int
    ) -> Dict[str, float]:
        """Generate waste for a single node at time step t.

        Steps:
            1. μ_i(t) = μ_i⁰ · exp(−α_i · t) + μ_i^base
            2. W_total ~ LogNormal(μ_i(t), σ_i²)
            3. W^w = ρ_i^w · W_total   for each waste type w
            4. Accumulate into node storage

        Parameters
        ----------
        node_id : int
            Target node identifier.
        t : int
            Current time step.

        Returns
        -------
        Dict[str, float]
            Mapping of waste type → generated quantity (tonnes).

        Raises
        ------
        KeyError
            If ``node_id`` is not registered.
        """
        config = self._nodes[node_id]
        mu_t = self._compute_mu(config, t)
        total = self._sample_waste(mu_t, config.sigma)

        per_type: Dict[str, float] = {
            wtype: config.waste_proportions.get(wtype, 0.0) * total
            for wtype in WASTE_TYPES
        }

        # Accumulate uncollected waste
        config.storage += total

        # Log the event
        self._generation_log.append((t, node_id, total, dict(per_type)))

        return per_type

    def step(
        self, t: Optional[int] = None
    ) -> Dict[int, Dict[str, float]]:
        """Advance all nodes by one time step and generate waste.

        Parameters
        ----------
        t : int, optional
            Explicit time step.  If ``None``, uses internal counter.

        Returns
        -------
        Dict[int, Dict[str, float]]
            Mapping of node_id → {waste_type: quantity} for every
            registered generation node.
        """
        if t is None:
            t = self._current_time

        results: Dict[int, Dict[str, float]] = {}
        for node_id in self._nodes:
            results[node_id] = self.generate_for_node(node_id, t)

        self._current_time = t + 1
        return results

    # ------------------------------------------------------------------
    # Collection / pickup interface
    # ------------------------------------------------------------------

    def collect_waste(
        self,
        node_id: int,
        amount: float,
        waste_type: Optional[str] = None,
    ) -> float:
        """Remove collected waste from a node's storage.

        This method is called by the vehicle agent after picking up waste
        from a generation node.

        Parameters
        ----------
        node_id : int
            Node from which waste is collected.
        amount : float
            Maximum quantity to collect (tonnes).  The actual collected
            quantity is ``min(amount, storage)``.
        waste_type : str, optional
            If specified, only collect this type.  Currently the storage
            is tracked as a scalar aggregate; per-type tracking can be
            extended in future iterations.

        Returns
        -------
        float
            Actually collected quantity (may be less than ``amount``
            if storage was insufficient).
        """
        config = self._nodes[node_id]
        collected = min(amount, config.storage)
        config.storage -= collected
        return collected

    # ------------------------------------------------------------------
    # Observation vectors (for RL state space)
    # ------------------------------------------------------------------

    def get_storage_vector(self) -> np.ndarray:
        """Return a 1-D array of current storage at each generation node.

        The ordering follows ``sorted(self._nodes.keys())``.

        Returns
        -------
        np.ndarray of shape (|N_gen|,)
        """
        return np.array(
            [self._nodes[nid].storage for nid in sorted(self._nodes)],
            dtype=np.float32,
        )

    def get_generation_rate_vector(self, t: Optional[int] = None) -> np.ndarray:
        """Return the **expected** waste generation rate at each node.

        This is the deterministic component E[W_i(t)] = exp(μ_i(t) + σ²/2)
        without any stochastic sampling.  Useful for feeding the agent
        a predictive signal about upcoming waste volumes.

        Parameters
        ----------
        t : int, optional
            Time step to evaluate.  Defaults to internal counter.

        Returns
        -------
        np.ndarray of shape (|N_gen|,)
        """
        if t is None:
            t = self._current_time

        rates: List[float] = []
        for nid in sorted(self._nodes):
            cfg = self._nodes[nid]
            mu_t = self._compute_mu(cfg, t)
            # E[LogNormal] = exp(μ + σ²/2)
            expected = float(np.exp(mu_t + (cfg.sigma ** 2) / 2.0))
            rates.append(expected)

        return np.array(rates, dtype=np.float32)

    def get_waste_proportion_matrix(self) -> np.ndarray:
        """Return node × waste-type proportion matrix.

        Returns
        -------
        np.ndarray of shape (|N_gen|, |W|)
            Row i corresponds to sorted node i, columns follow ``WASTE_TYPES``
            ordering.
        """
        matrix: List[List[float]] = []
        for nid in sorted(self._nodes):
            cfg = self._nodes[nid]
            row = [cfg.waste_proportions.get(wt, 0.0) for wt in WASTE_TYPES]
            matrix.append(row)
        return np.array(matrix, dtype=np.float32)

    # ------------------------------------------------------------------
    # Statistics and summary
    # ------------------------------------------------------------------

    def get_total_pending_waste(self) -> float:
        """Sum of uncollected waste across all generation nodes (tonnes)."""
        return sum(cfg.storage for cfg in self._nodes.values())

    def get_node_storage(self, node_id: int) -> float:
        """Return current uncollected waste at a specific node."""
        return self._nodes[node_id].storage

    def get_expected_total_at_time(self, t: int) -> float:
        """Total expected waste generation across all nodes at time t.

        Returns the sum of E[W_i(t)] = exp(μ_i(t) + σ²/2) over all nodes.
        """
        return float(np.sum(self.get_generation_rate_vector(t)))

    def get_generation_summary(self) -> Dict[str, float]:
        """Return aggregate statistics from the generation log.

        Returns
        -------
        dict with keys: ``total_generated``, ``total_pending``,
        ``mean_per_step``, ``max_single_step``.
        """
        if not self._generation_log:
            return {
                "total_generated": 0.0,
                "total_pending": self.get_total_pending_waste(),
                "mean_per_step": 0.0,
                "max_single_step": 0.0,
            }

        totals_per_step: Dict[int, float] = {}
        grand_total = 0.0
        for t, _, amount, _ in self._generation_log:
            totals_per_step[t] = totals_per_step.get(t, 0.0) + amount
            grand_total += amount

        step_totals = list(totals_per_step.values())
        return {
            "total_generated": grand_total,
            "total_pending": self.get_total_pending_waste(),
            "mean_per_step": float(np.mean(step_totals)),
            "max_single_step": float(np.max(step_totals)),
        }

    # ------------------------------------------------------------------
    # Reset and configuration helpers
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset all node storages, clear history, and optionally re-seed.

        Parameters
        ----------
        seed : int, optional
            New random seed.  If ``None``, re-seeds with current state.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        for cfg in self._nodes.values():
            cfg.storage = 0.0
        self._current_time = 0
        self._generation_log.clear()

    def configure_from_network(
        self,
        node_ids: List[int],
        mu_range: Tuple[float, float] = (3.0, 7.0),
        sigma_range: Tuple[float, float] = (0.2, 0.6),
        decay_range: Tuple[float, float] = (0.02, 0.08),
        mu_base_range: Tuple[float, float] = (0.5, 2.0),
    ) -> None:
        """Auto-configure waste generation nodes with random parameters.

        Convenient method to quickly populate the model from a list of
        waste generation node IDs obtained from ``DisasterNetwork``.

        Parameters
        ----------
        node_ids : List[int]
            Node IDs from the network that are waste generation sites.
        mu_range : Tuple[float, float]
            Uniform range for μ_i⁰.
        sigma_range : Tuple[float, float]
            Uniform range for σ_i.
        decay_range : Tuple[float, float]
            Uniform range for α_i.
        mu_base_range : Tuple[float, float]
            Uniform range for μ_i^base.
        """
        for nid in node_ids:
            # Randomise proportions with Dirichlet (adds natural variation)
            raw_proportions = self._rng.dirichlet(
                alpha=[4.5, 1.5, 1.2, 2.0, 0.8]  # prior weights per type
            )
            proportions = {
                wt: float(p) for wt, p in zip(WASTE_TYPES, raw_proportions)
            }

            config = WasteNodeConfig(
                node_id=nid,
                mu_initial=float(self._rng.uniform(*mu_range)),
                mu_base=float(self._rng.uniform(*mu_base_range)),
                sigma=float(self._rng.uniform(*sigma_range)),
                decay_rate=float(self._rng.uniform(*decay_range)),
                waste_proportions=proportions,
            )
            self.add_node(config)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def node_ids(self) -> List[int]:
        """Sorted list of registered waste generation node IDs."""
        return sorted(self._nodes.keys())

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def current_time(self) -> int:
        return self._current_time

    @property
    def generation_log(self) -> List[Tuple[int, int, float, Dict[str, float]]]:
        """Full history of (time, node_id, total, per_type_dict)."""
        return self._generation_log

    def __repr__(self) -> str:
        pending = self.get_total_pending_waste()
        return (
            f"WasteGenerationModel(nodes={self.num_nodes}, "
            f"t={self._current_time}, pending={pending:.1f}t)"
        )
