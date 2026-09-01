"""
scenario_generator.py — Parametric Disaster Scenario Factory
==============================================================

This module provides a ``ScenarioGenerator`` that constructs complete,
ready-to-simulate disaster waste logistics scenarios by orchestrating
the three foundation modules:

    1. ``DisasterNetwork``        →  road graph with damage parameters
    2. ``WasteGenerationModel``   →  stochastic waste sources
    3. ``Vehicle`` fleet          →  heterogeneous transport agents

Four standardised difficulty tiers are defined in the master plan:

    +-----------+----------+--------+---------+--------+-------------------+
    | Scenario  | Scale    | Nodes  | Vehicles| λ_dam  | Purpose           |
    +-----------+----------+--------+---------+--------+-------------------+
    | S1-Small  | Debug    | 20–30  | 3–5     | 0.03   | Unit testing      |
    | S2-Medium | Standard | 50–80  | 8–12    | 0.05   | Main experiments  |
    | S3-Large  | Stress   | 120–200| 15–25   | 0.07   | Scalability       |
    | S4-Severe | Crisis   | 50–80  | 8–12    | 0.12   | Resilience        |
    +-----------+----------+--------+---------+--------+-------------------+

All parameters are fully configurable via ``ScenarioConfig``; the tier
presets serve as sensible defaults.

Author  : Muhammed Şara
License : MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from .network import DisasterNetwork, NodeType
from .waste_model import WasteGenerationModel, WASTE_TYPES
from .vehicle import Vehicle, VehicleConfig


# ---------------------------------------------------------------------------
# Scenario tier enumeration
# ---------------------------------------------------------------------------

class ScenarioTier(Enum):
    """Pre-defined difficulty tiers for disaster scenarios."""
    S1_SMALL  = auto()
    S2_MEDIUM = auto()
    S3_LARGE  = auto()
    S4_SEVERE = auto()


# ---------------------------------------------------------------------------
# Configuration data class
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    """Full configuration for a disaster waste management scenario.

    All numeric fields carry sensible defaults corresponding to S2-Medium.
    Use ``ScenarioGenerator.from_tier()`` for pre-filled tier presets.

    Parameters
    ----------
    tier : ScenarioTier
        Difficulty tier label.
    n_generation : int
        Number of waste generation / demolition nodes.
    n_tcp : int
        Temporary collection points.
    n_sorting : int
        Sorting / recycling facilities.
    n_landfill : int
        Landfill / final disposal sites.
    n_depot : int
        Vehicle depots.
    n_vehicles : int
        Total fleet size.
    area_size : float
        Side length of the 2-D deployment area (km).
    connectivity : float
        Edge creation probability (see ``DisasterNetwork``).

    lambda_damage : float
        Poisson damage rate λ per time step.
    damage_severity : float
        Health drop Δφ on each damage event.
    repair_rate : float
        Linear health recovery μ_repair per step.

    vehicle_capacity_range : Tuple[float, float]
        (min, max) vehicle capacity in tonnes.
    vehicle_speed_range : Tuple[float, float]
        (min, max) average speed in km/h.
    vehicle_emission_range : Tuple[float, float]
        (min, max) emission factor in kg CO₂/km.

    mu_initial_range : Tuple[float, float]
        LogNormal μ₀ range for waste generation.
    sigma_range : Tuple[float, float]
        LogNormal σ range.
    decay_rate_range : Tuple[float, float]
        Exponential decay α range.
    mu_base_range : Tuple[float, float]
        Baseline log-mean range.

    max_time_steps : int
        Episode horizon T.
    hazmat_vehicle_fraction : float
        Fraction of vehicles designated as hazmat-only carriers.
    """
    tier: ScenarioTier = ScenarioTier.S2_MEDIUM

    # --- Network topology ---
    n_generation: int = 15
    n_tcp: int = 5
    n_sorting: int = 3
    n_landfill: int = 2
    n_depot: int = 2
    area_size: float = 80.0
    connectivity: float = 0.35

    # --- Road damage dynamics ---
    lambda_damage: float = 0.05
    damage_severity: float = 0.30
    repair_rate: float = 0.02

    # --- Fleet ---
    n_vehicles: int = 10
    vehicle_capacity_range: Tuple[float, float] = (15.0, 30.0)
    vehicle_speed_range: Tuple[float, float] = (30.0, 60.0)
    vehicle_emission_range: Tuple[float, float] = (0.5, 1.2)

    # --- Waste generation ---
    mu_initial_range: Tuple[float, float] = (3.0, 7.0)
    sigma_range: Tuple[float, float] = (0.2, 0.6)
    decay_rate_range: Tuple[float, float] = (0.02, 0.08)
    mu_base_range: Tuple[float, float] = (0.5, 2.0)

    # --- Episode ---
    max_time_steps: int = 200
    hazmat_vehicle_fraction: float = 0.15


# ---------------------------------------------------------------------------
# Pre-defined tier presets
# ---------------------------------------------------------------------------

_TIER_PRESETS: Dict[ScenarioTier, ScenarioConfig] = {
    ScenarioTier.S1_SMALL: ScenarioConfig(
        tier=ScenarioTier.S1_SMALL,
        n_generation=8,
        n_tcp=3,
        n_sorting=2,
        n_landfill=1,
        n_depot=1,
        n_vehicles=4,
        area_size=40.0,
        connectivity=0.45,
        lambda_damage=0.03,
        damage_severity=0.20,
        repair_rate=0.03,
        vehicle_capacity_range=(15.0, 25.0),
        vehicle_speed_range=(35.0, 55.0),
        vehicle_emission_range=(0.5, 1.0),
        mu_initial_range=(3.0, 5.0),
        sigma_range=(0.2, 0.4),
        decay_rate_range=(0.03, 0.07),
        mu_base_range=(0.5, 1.5),
        max_time_steps=100,
        hazmat_vehicle_fraction=0.0,  # no hazmat in small scenario
    ),
    ScenarioTier.S2_MEDIUM: ScenarioConfig(
        tier=ScenarioTier.S2_MEDIUM,
        # all defaults
    ),
    ScenarioTier.S3_LARGE: ScenarioConfig(
        tier=ScenarioTier.S3_LARGE,
        n_generation=40,
        n_tcp=12,
        n_sorting=6,
        n_landfill=4,
        n_depot=3,
        n_vehicles=20,
        area_size=150.0,
        connectivity=0.30,
        lambda_damage=0.07,
        damage_severity=0.30,
        repair_rate=0.015,
        vehicle_capacity_range=(15.0, 40.0),
        vehicle_speed_range=(25.0, 55.0),
        vehicle_emission_range=(0.6, 1.4),
        mu_initial_range=(4.0, 8.0),
        sigma_range=(0.3, 0.7),
        decay_rate_range=(0.01, 0.06),
        mu_base_range=(1.0, 3.0),
        max_time_steps=300,
        hazmat_vehicle_fraction=0.15,
    ),
    ScenarioTier.S4_SEVERE: ScenarioConfig(
        tier=ScenarioTier.S4_SEVERE,
        n_generation=20,
        n_tcp=6,
        n_sorting=3,
        n_landfill=2,
        n_depot=2,
        n_vehicles=10,
        area_size=80.0,
        connectivity=0.30,
        lambda_damage=0.12,       # very high damage rate
        damage_severity=0.45,     # severe health drops
        repair_rate=0.01,         # slow recovery
        vehicle_capacity_range=(12.0, 25.0),
        vehicle_speed_range=(20.0, 45.0),
        vehicle_emission_range=(0.7, 1.5),
        mu_initial_range=(5.0, 9.0),   # intense initial waste
        sigma_range=(0.4, 0.8),
        decay_rate_range=(0.01, 0.04),  # slow decay → prolonged crisis
        mu_base_range=(1.5, 3.5),
        max_time_steps=250,
        hazmat_vehicle_fraction=0.20,
    ),
}


# ---------------------------------------------------------------------------
# Scenario container (output of the generator)
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """Immutable container holding all instantiated components for one scenario.

    Attributes
    ----------
    config : ScenarioConfig
        The configuration that produced this scenario.
    network : DisasterNetwork
        The road network graph with nodes and edges.
    waste_model : WasteGenerationModel
        Stochastic waste generation engine (configured for gen-nodes).
    vehicles : List[Vehicle]
        The vehicle fleet, each assigned to a depot.
    seed : int
        Random seed used for reproducibility.
    """
    config: ScenarioConfig
    network: DisasterNetwork
    waste_model: WasteGenerationModel
    vehicles: List[Vehicle]
    seed: int

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def num_agents(self) -> int:
        return len(self.vehicles)

    @property
    def num_nodes(self) -> int:
        return self.network.num_nodes

    @property
    def generation_node_ids(self) -> List[int]:
        return self.network.get_nodes_by_type(NodeType.WASTE_GENERATION)

    @property
    def tcp_node_ids(self) -> List[int]:
        return self.network.get_nodes_by_type(NodeType.TCP)

    @property
    def sorting_node_ids(self) -> List[int]:
        return self.network.get_nodes_by_type(NodeType.SORTING_FACILITY)

    @property
    def landfill_node_ids(self) -> List[int]:
        return self.network.get_nodes_by_type(NodeType.LANDFILL)

    @property
    def depot_node_ids(self) -> List[int]:
        return self.network.get_nodes_by_type(NodeType.DEPOT)

    def summary(self) -> Dict[str, object]:
        """Return a human-readable summary dictionary."""
        return {
            "tier": self.config.tier.name,
            "seed": self.seed,
            "nodes": self.num_nodes,
            "edges": self.network.num_edges,
            "vehicles": self.num_agents,
            "generation_sites": len(self.generation_node_ids),
            "tcps": len(self.tcp_node_ids),
            "sorting_facilities": len(self.sorting_node_ids),
            "landfills": len(self.landfill_node_ids),
            "depots": len(self.depot_node_ids),
            "lambda_damage": self.config.lambda_damage,
            "max_time_steps": self.config.max_time_steps,
        }

    def __repr__(self) -> str:
        return (
            f"Scenario(tier={self.config.tier.name}, "
            f"nodes={self.num_nodes}, edges={self.network.num_edges}, "
            f"vehicles={self.num_agents}, seed={self.seed})"
        )


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class ScenarioGenerator:
    """Factory that builds complete, reproducible disaster scenarios.

    The generator wires together ``DisasterNetwork``, ``WasteGenerationModel``,
    and a ``Vehicle`` fleet according to a ``ScenarioConfig``.

    Parameters
    ----------
    seed : int
        Master random seed.  Sub-components derive their seeds from
        this to ensure full reproducibility.

    Examples
    --------
    >>> gen = ScenarioGenerator(seed=42)
    >>> scenario = gen.from_tier(ScenarioTier.S1_SMALL)
    >>> print(scenario)
    Scenario(tier=S1_SMALL, nodes=15, ...)
    """

    def __init__(self, seed: int = 42) -> None:
        self._master_seed: int = seed
        self._rng: np.random.Generator = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def from_tier(self, tier: ScenarioTier) -> Scenario:
        """Create a scenario from a pre-defined difficulty tier.

        Parameters
        ----------
        tier : ScenarioTier
            One of S1_SMALL, S2_MEDIUM, S3_LARGE, S4_SEVERE.

        Returns
        -------
        Scenario
            Fully configured and ready-to-simulate scenario.
        """
        config = _TIER_PRESETS[tier]
        return self.from_config(config)

    def from_config(self, config: ScenarioConfig) -> Scenario:
        """Create a scenario from an arbitrary configuration.

        Parameters
        ----------
        config : ScenarioConfig
            User-provided or tier-preset configuration.

        Returns
        -------
        Scenario
        """
        # Derive deterministic sub-seeds for each component
        sub_seeds = self._rng.integers(0, 2**31, size=3)
        net_seed = int(sub_seeds[0])
        waste_seed = int(sub_seeds[1])
        fleet_seed = int(sub_seeds[2])

        # --- 1. Build network ---
        network = self._build_network(config, net_seed)

        # --- 2. Build waste model ---
        waste_model = self._build_waste_model(config, network, waste_seed)

        # --- 3. Build vehicle fleet ---
        vehicles = self._build_fleet(config, network, fleet_seed)

        return Scenario(
            config=config,
            network=network,
            waste_model=waste_model,
            vehicles=vehicles,
            seed=self._master_seed,
        )

    def create_batch(
        self,
        tier: ScenarioTier,
        count: int = 5,
    ) -> List[Scenario]:
        """Generate multiple scenario instances with different seeds.

        Useful for statistical evaluation across random instances of the
        same difficulty tier.

        Parameters
        ----------
        tier : ScenarioTier
            Difficulty tier.
        count : int
            Number of independent scenarios to generate.

        Returns
        -------
        List[Scenario]
        """
        scenarios: List[Scenario] = []
        seeds = self._rng.integers(0, 2**31, size=count)
        for s in seeds:
            gen = ScenarioGenerator(seed=int(s))
            scenarios.append(gen.from_tier(tier))
        return scenarios

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build_network(
        self, config: ScenarioConfig, seed: int
    ) -> DisasterNetwork:
        """Construct and configure the road network.

        Steps:
            1. Generate random network topology.
            2. Override damage parameters with scenario config.
        """
        net = DisasterNetwork(seed=seed)
        net.generate_random_network(
            n_generation=config.n_generation,
            n_tcp=config.n_tcp,
            n_sorting=config.n_sorting,
            n_landfill=config.n_landfill,
            n_depot=config.n_depot,
            area_size=config.area_size,
            connectivity=config.connectivity,
        )

        # Override Poisson damage / repair parameters
        net.lambda_damage = config.lambda_damage
        net.damage_severity = config.damage_severity
        net.repair_rate = config.repair_rate

        return net

    def _build_waste_model(
        self,
        config: ScenarioConfig,
        network: DisasterNetwork,
        seed: int,
    ) -> WasteGenerationModel:
        """Attach waste generation nodes to the network's generation sites."""
        model = WasteGenerationModel(seed=seed)
        gen_ids = network.get_nodes_by_type(NodeType.WASTE_GENERATION)
        model.configure_from_network(
            node_ids=gen_ids,
            mu_range=config.mu_initial_range,
            sigma_range=config.sigma_range,
            decay_range=config.decay_rate_range,
            mu_base_range=config.mu_base_range,
        )
        return model

    def _build_fleet(
        self,
        config: ScenarioConfig,
        network: DisasterNetwork,
        seed: int,
    ) -> List[Vehicle]:
        """Create a heterogeneous vehicle fleet distributed across depots.

        Fleet composition:
            - ``hazmat_vehicle_fraction`` of the fleet is restricted to
              carrying only hazardous waste (specialised carriers).
            - The remaining vehicles carry all waste types.
            - Vehicles are distributed round-robin across available depots.
        """
        rng = np.random.default_rng(seed)
        depots = network.get_nodes_by_type(NodeType.DEPOT)
        if not depots:
            raise ValueError("No depot nodes found in the network.")

        n_hazmat = max(0, int(config.n_vehicles * config.hazmat_vehicle_fraction))
        vehicles: List[Vehicle] = []

        for i in range(config.n_vehicles):
            # Round-robin depot assignment
            home_depot = depots[i % len(depots)]

            capacity = float(rng.uniform(*config.vehicle_capacity_range))
            speed = float(rng.uniform(*config.vehicle_speed_range))
            emission = float(rng.uniform(*config.vehicle_emission_range))
            cost_km = float(rng.uniform(0.8, 2.0))

            # Hazmat specialisation: first n_hazmat vehicles
            if i < n_hazmat:
                compatible = ["hazardous"]
            else:
                compatible = list(WASTE_TYPES)

            vcfg = VehicleConfig(
                vehicle_id=i,
                capacity=round(capacity, 1),
                speed=round(speed, 1),
                emission_factor=round(emission, 3),
                cost_per_km=round(cost_km, 2),
                compatible_waste=compatible,
                home_depot=home_depot,
            )
            vehicles.append(Vehicle(vcfg))

        return vehicles

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def available_tiers() -> List[str]:
        """List names of all pre-defined scenario tiers."""
        return [t.name for t in ScenarioTier]

    @staticmethod
    def get_tier_config(tier: ScenarioTier) -> ScenarioConfig:
        """Return the default ScenarioConfig for a tier (read-only copy)."""
        import copy
        return copy.deepcopy(_TIER_PRESETS[tier])

    def __repr__(self) -> str:
        return f"ScenarioGenerator(master_seed={self._master_seed})"
