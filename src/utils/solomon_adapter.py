"""
solomon_adapter.py — Solomon VRP Benchmark → Disaster Waste Scenario Converter
================================================================================

This module reads classical Solomon VRPTW benchmark instances (.txt format)
and transforms them into disaster waste management scenarios compatible with
``DisasterNetwork``, ``WasteGenerationModel``, and ``Vehicle`` fleet.

Solomon .txt format (per line, after header):
    CUST_NO  XCOORD  YCOORD  DEMAND  READY_TIME  DUE_DATE  SERVICE_TIME

Conversion rules (see implementation_plan.md §3.2):
    1. Solomon DEPOT (cust 0)             → NodeType.DEPOT
    2. Solomon CUSTOMERs                  → NodeType.WASTE_GENERATION
    3. Synthetically injected nodes       → TCP, SORTING_FACILITY, LANDFILL
    4. Customer DEMAND                    → μ_i⁰ (LogNormal initial mean)
    5. Edges created from Euclidean dist  → travel time + cost + emission
    6. Poisson road damage added          → λ_damage, severity, repair rate

Supported series: R1, R2, C1, C2, RC1, RC2 (Solomon, 1987).

Author  : Muhammed Şara
License : MIT
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.environment.network import DisasterNetwork, EdgeAttributes, NodeAttributes, NodeType
from src.environment.vehicle import Vehicle, VehicleConfig
from src.environment.waste_model import WASTE_TYPES, WasteGenerationModel, WasteNodeConfig
from src.environment.scenario_generator import Scenario, ScenarioConfig, ScenarioTier


# ---------------------------------------------------------------------------
# Parsed data structures
# ---------------------------------------------------------------------------

@dataclass
class SolomonCustomer:
    """Parsed representation of one customer/depot line in Solomon format.

    Parameters
    ----------
    cust_no : int
        Customer number (0 = depot).
    x : float
        X-coordinate.
    y : float
        Y-coordinate.
    demand : float
        Customer demand (units).
    ready_time : float
        Start of time window.
    due_date : float
        End of time window.
    service_time : float
        Service duration at customer.
    """
    cust_no: int
    x: float
    y: float
    demand: float
    ready_time: float
    due_date: float
    service_time: float


@dataclass
class SolomonInstance:
    """Complete parsed Solomon benchmark instance.

    Parameters
    ----------
    name : str
        Instance name (e.g., "R101").
    num_vehicles : int
        Number of available vehicles.
    vehicle_capacity : float
        Uniform vehicle capacity.
    depot : SolomonCustomer
        The depot node (customer 0).
    customers : List[SolomonCustomer]
        All customer nodes (customer 1..N).
    """
    name: str = ""
    num_vehicles: int = 25
    vehicle_capacity: float = 200.0
    depot: Optional[SolomonCustomer] = None
    customers: List[SolomonCustomer] = field(default_factory=list)

    @property
    def num_customers(self) -> int:
        return len(self.customers)


# ---------------------------------------------------------------------------
# Adapter configuration
# ---------------------------------------------------------------------------

@dataclass
class AdapterConfig:
    """Configuration for the Solomon → disaster waste conversion.

    Parameters
    ----------
    n_tcp : int
        Number of TCP nodes to inject.
    n_sorting : int
        Number of sorting/recycling facilities to inject.
    n_landfill : int
        Number of landfill/disposal sites to inject.
    n_extra_depots : int
        Additional depots beyond Solomon's single depot (for multi-depot).
    lambda_damage : float
        Poisson road damage rate.
    damage_severity : float
        Health drop per damage event.
    repair_rate : float
        Health recovery per time step.
    demand_to_mu_scale : float
        Multiplier: μ_i⁰ = demand × scale.  Solomon demands are
        typically in [1, 50]; this maps them to log-means.
    sigma : float
        Log-normal σ for waste generation.
    decay_rate : float
        Waste decay α.
    mu_base_fraction : float
        μ_base = μ_i⁰ × fraction.
    connectivity_radius_pct : float
        Edges are created for node pairs within this percentile of
        all pairwise distances.
    speed : float
        Nominal vehicle speed (km/h), for travel time computation.
    max_time_steps : int
        Episode horizon.
    time_window_noise_std : float
        Gaussian noise σ added to Solomon time windows to simulate
        stochastic shifts (0.0 = deterministic windows).
    """
    n_tcp: int = 3
    n_sorting: int = 2
    n_landfill: int = 1
    n_extra_depots: int = 1
    lambda_damage: float = 0.05
    damage_severity: float = 0.30
    repair_rate: float = 0.02
    demand_to_mu_scale: float = 0.15
    sigma: float = 0.4
    decay_rate: float = 0.05
    mu_base_fraction: float = 0.2
    connectivity_radius_pct: float = 60.0
    speed: float = 40.0
    max_time_steps: int = 200
    time_window_noise_std: float = 0.0


# ---------------------------------------------------------------------------
# Main adapter class
# ---------------------------------------------------------------------------

class SolomonAdapter:
    """Converts Solomon VRP instances into disaster waste scenarios.

    The adapter:
        1. Parses the Solomon .txt file into a ``SolomonInstance``.
        2. Maps depot/customer nodes to disaster logistics node types.
        3. Injects synthetic TCP, sorting, and landfill nodes.
        4. Creates edges based on Euclidean distance thresholding.
        5. Configures Poisson road damage and stochastic waste generation.
        6. Builds a heterogeneous vehicle fleet.
        7. Returns a ``Scenario`` object ready for environment use.

    Parameters
    ----------
    config : AdapterConfig, optional
        Conversion parameters.  Defaults to sensible values.
    seed : int
        Random seed for synthetic node placement and parameter sampling.

    Examples
    --------
    >>> adapter = SolomonAdapter(seed=42)
    >>> scenario = adapter.from_file("data/solomon/R101.txt")
    >>> print(scenario.summary())
    """

    def __init__(
        self,
        config: Optional[AdapterConfig] = None,
        seed: int = 42,
    ) -> None:
        self._config = config or AdapterConfig()
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    # ==================================================================
    # Public API
    # ==================================================================

    def from_file(self, filepath: str) -> Scenario:
        """Parse a Solomon .txt file and convert to a disaster Scenario.

        Parameters
        ----------
        filepath : str
            Path to the Solomon benchmark file (e.g., ``R101.txt``).

        Returns
        -------
        Scenario
            Fully configured disaster waste scenario.
        """
        instance = self.parse_solomon_file(filepath)
        return self.convert(instance)

    def from_string(self, content: str, name: str = "custom") -> Scenario:
        """Parse Solomon-format text from a string and convert.

        Parameters
        ----------
        content : str
            Multi-line string in Solomon format.
        name : str
            Instance name label.

        Returns
        -------
        Scenario
        """
        instance = self._parse_content(content, name)
        return self.convert(instance)

    def convert(self, instance: SolomonInstance) -> Scenario:
        """Convert a parsed SolomonInstance into a disaster Scenario.

        Pipeline:
            1. Build DisasterNetwork (nodes + edges).
            2. Build WasteGenerationModel from customer demands.
            3. Build vehicle fleet from Solomon vehicle parameters.
            4. Wrap into Scenario container.

        Parameters
        ----------
        instance : SolomonInstance

        Returns
        -------
        Scenario
        """
        network = self._build_network(instance)
        waste_model = self._build_waste_model(instance, network)
        vehicles = self._build_fleet(instance, network)

        config = ScenarioConfig(
            tier=ScenarioTier.S2_MEDIUM,
            n_generation=instance.num_customers,
            n_tcp=self._config.n_tcp,
            n_sorting=self._config.n_sorting,
            n_landfill=self._config.n_landfill,
            n_depot=1 + self._config.n_extra_depots,
            n_vehicles=len(vehicles),
            lambda_damage=self._config.lambda_damage,
            damage_severity=self._config.damage_severity,
            repair_rate=self._config.repair_rate,
            max_time_steps=self._config.max_time_steps,
        )

        return Scenario(
            config=config,
            network=network,
            waste_model=waste_model,
            vehicles=vehicles,
            seed=self._seed,
        )

    # ==================================================================
    # Parsing
    # ==================================================================

    def parse_solomon_file(self, filepath: str) -> SolomonInstance:
        """Parse a Solomon benchmark .txt file.

        The standard Solomon format consists of:
            - Line 1: Instance name
            - Lines 2–4: (empty or header text)
            - Line 5: VEHICLE  NUMBER  CAPACITY
            - Line 6: (empty)
            - Line 7: header labels
            - Lines 8+: CUST_NO  XCOORD  YCOORD  DEMAND  READY  DUE  SERVICE

        The parser is resilient to minor format variations across
        different Solomon file distributions.

        Parameters
        ----------
        filepath : str
            Path to the Solomon .txt file.

        Returns
        -------
        SolomonInstance
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Solomon file not found: {filepath}")

        content = path.read_text(encoding="utf-8")
        name = path.stem  # e.g., "R101"
        return self._parse_content(content, name)

    def _parse_content(self, content: str, name: str) -> SolomonInstance:
        """Parse Solomon-format text content into a SolomonInstance.

        The parser uses a state-machine approach:
            1. Scan for VEHICLE header → extract num_vehicles, capacity.
            2. Scan for CUSTOMER header → parse remaining lines as data.
            3. Customer 0 = depot; customers 1..N = customers.

        Parameters
        ----------
        content : str
            Raw file content.
        name : str
            Instance label.

        Returns
        -------
        SolomonInstance
        """
        instance = SolomonInstance(name=name)
        lines = content.strip().split("\n")

        # ---- Phase 1: find vehicle info ----
        vehicle_section_found = False
        data_start_idx = 0

        for idx, line in enumerate(lines):
            stripped = line.strip().upper()
            # Look for the VEHICLE section header
            if "VEHICLE" in stripped and "NUMBER" in stripped:
                vehicle_section_found = True
                continue
            # After the vehicle header, the next numeric line has the values
            if vehicle_section_found and not stripped:
                continue
            if vehicle_section_found:
                nums = self._extract_numbers(line)
                if len(nums) >= 2:
                    instance.num_vehicles = int(nums[0])
                    instance.vehicle_capacity = float(nums[1])
                    vehicle_section_found = False
                continue
            # Look for CUSTOMER or CUST_NO header → data starts after it
            if "CUST_NO" in stripped or "CUST" in stripped and "COORD" in stripped:
                data_start_idx = idx + 1
                break

        # ---- Phase 2: parse customer/depot data ----
        for idx in range(data_start_idx, len(lines)):
            nums = self._extract_numbers(lines[idx])
            if len(nums) < 7:
                continue  # skip malformed or empty lines

            customer = SolomonCustomer(
                cust_no=int(nums[0]),
                x=float(nums[1]),
                y=float(nums[2]),
                demand=float(nums[3]),
                ready_time=float(nums[4]),
                due_date=float(nums[5]),
                service_time=float(nums[6]),
            )

            if customer.cust_no == 0:
                instance.depot = customer
            else:
                instance.customers.append(customer)

        if instance.depot is None and instance.customers:
            # Fallback: treat first customer as depot
            instance.depot = instance.customers.pop(0)

        return instance

    @staticmethod
    def _extract_numbers(line: str) -> List[float]:
        """Extract all numeric tokens from a line."""
        return [float(x) for x in re.findall(r"-?\d+\.?\d*", line)]

    # ==================================================================
    # Network construction
    # ==================================================================

    def _build_network(self, instance: SolomonInstance) -> DisasterNetwork:
        """Build a DisasterNetwork from a parsed Solomon instance.

        Node mapping scheme:
            ID 0            : Solomon depot  → NodeType.DEPOT
            ID 1..N         : Solomon custs  → NodeType.WASTE_GENERATION
            ID N+1..N+D     : Extra depots   → NodeType.DEPOT
            ID N+D+1..      : TCP, Sorting, Landfill (injected)
        """
        net = DisasterNetwork(seed=self._seed)
        cfg = self._config

        # --- Depot (Node 0) ---
        depot = instance.depot
        net.add_node(0, NodeAttributes(
            node_type=NodeType.DEPOT,
            position=(depot.x, depot.y),
            capacity=0.0,
        ))

        # --- Customer → Waste Generation nodes (1..N) ---
        for cust in instance.customers:
            net.add_node(cust.cust_no, NodeAttributes(
                node_type=NodeType.WASTE_GENERATION,
                position=(cust.x, cust.y),
                capacity=float(cust.demand * 3),  # storage = 3× demand
            ))

        # --- Compute bounding box for synthetic node placement ---
        all_x = [depot.x] + [c.x for c in instance.customers]
        all_y = [depot.y] + [c.y for c in instance.customers]
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        margin = 0.1 * max(x_max - x_min, y_max - y_min)

        next_id = max(c.cust_no for c in instance.customers) + 1

        # --- Extra depots ---
        for _ in range(cfg.n_extra_depots):
            pos = self._sample_position_in_bbox(
                x_min - margin, x_max + margin,
                y_min - margin, y_max + margin,
            )
            net.add_node(next_id, NodeAttributes(
                node_type=NodeType.DEPOT,
                position=pos,
                capacity=0.0,
            ))
            next_id += 1

        # --- TCP nodes (scattered among customers) ---
        for _ in range(cfg.n_tcp):
            pos = self._sample_position_near_centroid(
                instance.customers, x_min, x_max, y_min, y_max
            )
            net.add_node(next_id, NodeAttributes(
                node_type=NodeType.TCP,
                position=pos,
                capacity=float(self._rng.uniform(200, 600)),
            ))
            next_id += 1

        # --- Sorting / recycling facilities ---
        waste_types = ["concrete", "metal", "wood", "mixed", "hazardous"]
        for _ in range(cfg.n_sorting):
            pos = self._sample_position_in_bbox(
                x_min, x_max, y_min, y_max
            )
            rec_rates = {wt: float(self._rng.uniform(0.3, 0.85))
                         for wt in waste_types}
            net.add_node(next_id, NodeAttributes(
                node_type=NodeType.SORTING_FACILITY,
                position=pos,
                capacity=float(self._rng.uniform(150, 500)),
                recycling_rates=rec_rates,
            ))
            next_id += 1

        # --- Landfill nodes (placed at periphery) ---
        for _ in range(cfg.n_landfill):
            pos = self._sample_position_at_periphery(
                x_min, x_max, y_min, y_max, margin
            )
            net.add_node(next_id, NodeAttributes(
                node_type=NodeType.LANDFILL,
                position=pos,
                capacity=float(self._rng.uniform(1000, 5000)),
            ))
            next_id += 1

        # --- Create edges based on distance thresholding ---
        self._create_edges(net, cfg.speed)

        # --- Configure Poisson damage parameters ---
        net.lambda_damage = cfg.lambda_damage
        net.damage_severity = cfg.damage_severity
        net.repair_rate = cfg.repair_rate

        return net

    def _create_edges(self, net: DisasterNetwork, speed: float) -> None:
        """Create directed edges based on Euclidean distance thresholding.

        All node pairs within the connectivity_radius_pct percentile of
        pairwise distances get bidirectional edges.  Strong connectivity
        is then enforced.
        """
        node_list = sorted(net.graph.nodes)
        positions = np.array([net.get_node_position(n) for n in node_list])
        n = len(node_list)

        # Pairwise Euclidean distance matrix
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

        # Distance threshold from config percentile
        non_zero = dist_matrix[dist_matrix > 0]
        threshold = float(np.percentile(non_zero, self._config.connectivity_radius_pct))

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = dist_matrix[i, j]
                if d <= threshold:
                    travel_time = d / speed
                    attrs = EdgeAttributes(
                        distance=round(d, 2),
                        base_travel_time=round(travel_time, 4),
                        unit_cost=round(float(self._rng.uniform(0.5, 2.0)), 2),
                        unit_emission=round(float(self._rng.uniform(0.3, 1.2)), 2),
                        health=1.0,
                    )
                    net.add_edge(node_list[i], node_list[j], attrs)

        # Ensure strong connectivity
        net._ensure_connectivity(speed)

    # ==================================================================
    # Waste model construction
    # ==================================================================

    def _build_waste_model(
        self, instance: SolomonInstance, network: DisasterNetwork
    ) -> WasteGenerationModel:
        """Build WasteGenerationModel from Solomon customer demands.

        Demand → μ_i⁰ mapping:
            μ_i⁰ = demand × demand_to_mu_scale

        Each customer gets unique Dirichlet-sampled waste-type proportions
        and the configured σ and decay rate.
        """
        cfg = self._config
        model = WasteGenerationModel(seed=self._seed + 100)

        for cust in instance.customers:
            mu_0 = cust.demand * cfg.demand_to_mu_scale
            mu_base = mu_0 * cfg.mu_base_fraction

            # Dirichlet-sampled waste proportions
            raw_prop = self._rng.dirichlet(alpha=[4.5, 1.5, 1.2, 2.0, 0.8])
            proportions = {
                wt: float(p) for wt, p in zip(WASTE_TYPES, raw_prop)
            }

            node_cfg = WasteNodeConfig(
                node_id=cust.cust_no,
                mu_initial=max(mu_0, 0.5),
                mu_base=max(mu_base, 0.1),
                sigma=cfg.sigma,
                decay_rate=cfg.decay_rate,
                waste_proportions=proportions,
            )
            model.add_node(node_cfg)

        return model

    # ==================================================================
    # Fleet construction
    # ==================================================================

    def _build_fleet(
        self, instance: SolomonInstance, network: DisasterNetwork
    ) -> List[Vehicle]:
        """Build a vehicle fleet from Solomon vehicle parameters.

        Solomon specifies a homogeneous fleet; we introduce heterogeneity
        by sampling capacity and speed around the Solomon baseline:
            capacity ∈ [0.8 × C, 1.2 × C]
            speed    ∈ [30, 60] km/h
        """
        depots = network.get_nodes_by_type(NodeType.DEPOT)
        num_veh = min(instance.num_vehicles, max(instance.num_customers // 5, 3))
        base_cap = instance.vehicle_capacity

        vehicles: List[Vehicle] = []
        for i in range(num_veh):
            home = depots[i % len(depots)]

            vcfg = VehicleConfig(
                vehicle_id=i,
                capacity=round(float(self._rng.uniform(
                    0.8 * base_cap, 1.2 * base_cap)), 1),
                speed=round(float(self._rng.uniform(30.0, 60.0)), 1),
                emission_factor=round(float(self._rng.uniform(0.5, 1.2)), 3),
                cost_per_km=round(float(self._rng.uniform(0.8, 2.0)), 2),
                compatible_waste=list(WASTE_TYPES),
                home_depot=home,
            )
            vehicles.append(Vehicle(vcfg))

        return vehicles

    # ==================================================================
    # Synthetic node placement helpers
    # ==================================================================

    def _sample_position_in_bbox(
        self, x_min: float, x_max: float, y_min: float, y_max: float
    ) -> Tuple[float, float]:
        """Sample a random position within the bounding box."""
        return (
            float(self._rng.uniform(x_min, x_max)),
            float(self._rng.uniform(y_min, y_max)),
        )

    def _sample_position_near_centroid(
        self,
        customers: List[SolomonCustomer],
        x_min: float, x_max: float,
        y_min: float, y_max: float,
    ) -> Tuple[float, float]:
        """Sample a position near the centroid of customer locations.

        TCPs are best placed centrally, close to clusters of waste
        generation nodes, so we add Gaussian noise around the centroid.
        """
        if not customers:
            return self._sample_position_in_bbox(x_min, x_max, y_min, y_max)

        cx = np.mean([c.x for c in customers])
        cy = np.mean([c.y for c in customers])
        spread = max(x_max - x_min, y_max - y_min) * 0.15

        return (
            float(np.clip(cx + self._rng.normal(0, spread), x_min, x_max)),
            float(np.clip(cy + self._rng.normal(0, spread), y_min, y_max)),
        )

    def _sample_position_at_periphery(
        self,
        x_min: float, x_max: float,
        y_min: float, y_max: float,
        margin: float,
    ) -> Tuple[float, float]:
        """Sample a position at the periphery of the bounding box.

        Landfills are typically located on the outskirts of the service
        area, so we bias placement toward edges and corners.
        """
        side = int(self._rng.integers(0, 4))
        if side == 0:  # top
            x = float(self._rng.uniform(x_min, x_max))
            y = float(y_max + self._rng.uniform(0, margin))
        elif side == 1:  # bottom
            x = float(self._rng.uniform(x_min, x_max))
            y = float(y_min - self._rng.uniform(0, margin))
        elif side == 2:  # right
            x = float(x_max + self._rng.uniform(0, margin))
            y = float(self._rng.uniform(y_min, y_max))
        else:  # left
            x = float(x_min - self._rng.uniform(0, margin))
            y = float(self._rng.uniform(y_min, y_max))
        return (x, y)

    # ==================================================================
    # Utility: generate a synthetic Solomon-like .txt for testing
    # ==================================================================

    @staticmethod
    def generate_sample_solomon(
        n_customers: int = 25,
        area_size: float = 100.0,
        seed: int = 42,
    ) -> str:
        """Generate a synthetic Solomon-format string for testing.

        Produces a valid Solomon .txt content with uniformly distributed
        customers (R-series style), random demands, and time windows.

        Parameters
        ----------
        n_customers : int
            Number of customers (excluding depot).
        area_size : float
            Side length of the customer deployment area.
        seed : int
            Random seed.

        Returns
        -------
        str
            Multi-line string in Solomon .txt format.
        """
        rng = np.random.default_rng(seed)
        lines: List[str] = []

        # Header
        lines.append(f"SYNTHETIC_R1_{n_customers}")
        lines.append("")
        lines.append("VEHICLE")
        lines.append("NUMBER     CAPACITY")
        lines.append(f"  {n_customers // 4 + 2}          200")
        lines.append("")
        lines.append("CUSTOMER")
        lines.append(
            "CUST_NO.  XCOORD.  YCOORD.  DEMAND  READY_TIME  DUE_DATE  SERVICE_TIME"
        )
        lines.append("")

        # Depot at centre
        cx, cy = area_size / 2, area_size / 2
        lines.append(f"    0      {cx:.0f}      {cy:.0f}       0       0     999       0")

        # Customers
        for i in range(1, n_customers + 1):
            x = rng.uniform(0, area_size)
            y = rng.uniform(0, area_size)
            demand = rng.integers(5, 45)
            ready = rng.integers(0, 200)
            due = ready + rng.integers(50, 300)
            service = rng.integers(5, 20)
            lines.append(
                f"   {i:2d}      {x:5.1f}    {y:5.1f}      {demand:3d}"
                f"       {ready:3d}     {due:3d}      {service:3d}"
            )

        return "\n".join(lines)

    # ==================================================================
    # Representation
    # ==================================================================

    def __repr__(self) -> str:
        return (
            f"SolomonAdapter(seed={self._seed}, "
            f"tcp={self._config.n_tcp}, "
            f"sort={self._config.n_sorting}, "
            f"land={self._config.n_landfill})"
        )
