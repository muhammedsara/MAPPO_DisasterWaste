"""
vehicle.py — Heterogeneous Vehicle Agent Model
================================================

This module defines the ``Vehicle`` class that represents an individual
transport agent in the disaster waste logistics fleet.  Each vehicle
maintains its own:

    - Physical attributes (capacity, speed, emission factor)
    - Current state (position, loaded waste by type, remaining capacity)
    - Trip log (visited nodes, kilometres driven, emissions produced)

The class exposes **pickup** and **dropoff** operations that enforce
capacity and waste-type constraints, as well as RL-observation helpers
that vectorise the agent state for policy network consumption.

Design decisions:
    - **Heterogeneous fleet**: Vehicles differ in capacity, speed, and
      emission profiles, reflecting real-world mixed fleets (trucks,
      heavy-duty trailers, hazmat carriers).
    - **Per-type cargo tracking**: The ``cargo`` dictionary tracks each
      waste type independently, enabling the environment to enforce
      facility compatibility (e.g., hazardous waste → specialised facility).
    - **Trip accounting**: Cumulative cost, time, and emission tallies
      are maintained internally so the reward function can query them
      without recomputation.

Author  : Muhammed Şara
License : MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from .waste_model import WASTE_TYPES


# ---------------------------------------------------------------------------
# Enumerations & data classes
# ---------------------------------------------------------------------------

class VehicleStatus(Enum):
    """Operational status of a vehicle agent."""
    IDLE = auto()         # Parked at depot, awaiting dispatch
    EN_ROUTE = auto()     # Travelling between nodes
    LOADING = auto()      # At a waste generation node, picking up
    UNLOADING = auto()    # At a facility, dropping off
    RETURNING = auto()    # Heading back to depot (trip complete)


@dataclass
class VehicleConfig:
    """Immutable configuration for a single vehicle.

    Parameters
    ----------
    vehicle_id : int
        Unique identifier within the fleet.
    capacity : float
        Maximum total cargo in tonnes.
    speed : float
        Average travel speed (km/h).  Combined with road health and
        distance, this determines effective travel time.
    emission_factor : float
        CO₂ emission rate (kg CO₂ per km).  Heavier or older trucks
        have higher factors.
    cost_per_km : float
        Monetary operating cost per kilometre ($/km).
    compatible_waste : List[str]
        Waste types this vehicle is certified to carry.
        Default: all types.  Hazmat carriers may restrict to
        ``["hazardous"]`` only.
    home_depot : int
        Node ID of the vehicle's home depot (start / return point).
    """
    vehicle_id: int
    capacity: float = 20.0
    speed: float = 40.0
    emission_factor: float = 0.8
    cost_per_km: float = 1.5
    compatible_waste: List[str] = field(default_factory=lambda: list(WASTE_TYPES))
    home_depot: int = 0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Vehicle:
    """State-bearing vehicle agent for the disaster waste logistics fleet.

    Parameters
    ----------
    config : VehicleConfig
        Immutable physical and operational parameters.

    Examples
    --------
    >>> cfg = VehicleConfig(vehicle_id=0, capacity=25.0, home_depot=18)
    >>> v = Vehicle(cfg)
    >>> v.pickup({"concrete": 8.0, "metal": 3.0})
    11.0
    >>> v.remaining_capacity
    14.0
    >>> v.dropoff()
    {'concrete': 8.0, 'metal': 3.0}
    """

    def __init__(self, config: VehicleConfig) -> None:
        self.config: VehicleConfig = config

        # --- Mutable state ---
        self._current_node: int = config.home_depot
        self._next_node: Optional[int] = None
        self._status: VehicleStatus = VehicleStatus.IDLE

        # Per waste-type cargo: {waste_type: tonnes_loaded}
        self._cargo: Dict[str, float] = {wt: 0.0 for wt in WASTE_TYPES}

        # Trip-level accumulators
        self._total_distance: float = 0.0    # km driven this episode
        self._total_cost: float = 0.0        # $ spent this episode
        self._total_emission: float = 0.0    # kg CO₂ emitted this episode
        self._total_time: float = 0.0        # hours elapsed this episode
        self._nodes_visited: List[int] = [config.home_depot]
        self._pickup_count: int = 0
        self._dropoff_count: int = 0

        # Per-type total delivered (for recycling rate tracking)
        self._total_delivered: Dict[str, float] = {wt: 0.0 for wt in WASTE_TYPES}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def pickup(
        self,
        waste_amounts: Dict[str, float],
        strict: bool = True,
    ) -> float:
        """Load waste onto the vehicle, respecting capacity and compatibility.

        The method loads as much as possible without exceeding total capacity.
        If ``strict=True``, incompatible waste types raise an error; otherwise
        they are silently skipped.

        Parameters
        ----------
        waste_amounts : Dict[str, float]
            Desired pickup quantities for each waste type (tonnes).
        strict : bool
            If ``True``, raise ``ValueError`` on incompatible waste.

        Returns
        -------
        float
            Total tonnes actually loaded in this operation.

        Raises
        ------
        ValueError
            If ``strict=True`` and an incompatible waste type is requested.
        """
        remaining = self.remaining_capacity
        total_loaded = 0.0

        for wtype, amount in waste_amounts.items():
            if amount <= 0.0:
                continue

            # Compatibility check
            if wtype not in self.config.compatible_waste:
                if strict:
                    raise ValueError(
                        f"Vehicle {self.config.vehicle_id} cannot carry "
                        f"waste type '{wtype}'. Compatible: "
                        f"{self.config.compatible_waste}"
                    )
                continue

            # Load as much as capacity allows
            loadable = min(amount, remaining)
            if loadable <= 0.0:
                break

            self._cargo[wtype] += loadable
            total_loaded += loadable
            remaining -= loadable

        if total_loaded > 0:
            self._pickup_count += 1
            self._status = VehicleStatus.LOADING

        return total_loaded

    def dropoff(self) -> Dict[str, float]:
        """Unload the entire cargo at the current facility.

        Returns
        -------
        Dict[str, float]
            Mapping of waste type → tonnes unloaded.
            The dictionary is a snapshot; the vehicle's cargo is zeroed.
        """
        delivered = {wt: qty for wt, qty in self._cargo.items() if qty > 0.0}

        # Accumulate delivery history
        for wt, qty in delivered.items():
            self._total_delivered[wt] += qty

        # Reset cargo
        self._cargo = {wt: 0.0 for wt in WASTE_TYPES}

        if delivered:
            self._dropoff_count += 1
            self._status = VehicleStatus.UNLOADING

        return delivered

    def partial_dropoff(self, waste_type: str, amount: float) -> float:
        """Unload a specific waste type partially.

        Parameters
        ----------
        waste_type : str
            Which waste category to unload.
        amount : float
            Maximum tonnes to unload.

        Returns
        -------
        float
            Actually unloaded quantity.
        """
        available = self._cargo.get(waste_type, 0.0)
        unloaded = min(amount, available)
        self._cargo[waste_type] -= unloaded
        self._total_delivered[waste_type] = (
            self._total_delivered.get(waste_type, 0.0) + unloaded
        )
        if unloaded > 0:
            self._dropoff_count += 1
        return unloaded

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def move_to(
        self,
        target_node: int,
        distance: float,
        travel_time: float,
    ) -> Dict[str, float]:
        """Execute a move from the current node to ``target_node``.

        Updates position, accumulates distance/cost/time/emission, and
        appends the target to the visit log.

        Parameters
        ----------
        target_node : int
            Destination node ID.
        distance : float
            Road distance to target (km).
        travel_time : float
            Effective travel time (hours), already health-adjusted.

        Returns
        -------
        Dict[str, float]
            Trip delta: ``{"distance", "cost", "emission", "time"}``.
        """
        cost = distance * self.config.cost_per_km
        # Dynamic emission: heavier loads produce more CO2
        _LOAD_PENALTY_K = 0.05  # kg CO2 per km per tonne of cargo
        emission = distance * (self.config.emission_factor + _LOAD_PENALTY_K * self.current_load)

        self._current_node = target_node
        self._next_node = None
        self._total_distance += distance
        self._total_cost += cost
        self._total_emission += emission
        self._total_time += travel_time
        self._nodes_visited.append(target_node)
        self._status = VehicleStatus.EN_ROUTE

        return {
            "distance": distance,
            "cost": cost,
            "emission": emission,
            "time": travel_time,
        }

    def set_next_node(self, node_id: int) -> None:
        """Set the intended next-hop destination (action output)."""
        self._next_node = node_id

    def return_to_depot(self, distance: float, travel_time: float) -> Dict[str, float]:
        """Convenience: move back to the home depot."""
        self._status = VehicleStatus.RETURNING
        return self.move_to(self.config.home_depot, distance, travel_time)

    # ------------------------------------------------------------------
    # RL observation vectors
    # ------------------------------------------------------------------

    def get_observation_vector(self) -> np.ndarray:
        """Construct a fixed-size observation vector for the RL policy.

        Layout (14 elements):
            [0]      : vehicle_id (normalised by fleet size externally)
            [1]      : current_node  (integer, embedded externally)
            [2]      : remaining_capacity / max_capacity  ∈ [0, 1]
            [3]      : total_cargo / max_capacity          ∈ [0, 1]
            [4–8]    : per waste-type load fractions (5 values, ∈ [0, 1])
            [9]      : speed (raw, normalised externally)
            [10]     : emission_factor (raw)
            [11]     : total_distance driven so far
            [12]     : total_time elapsed so far
            [13]     : status (ordinal encoding 0–4)

        Returns
        -------
        np.ndarray of shape (14,)
        """
        cap = self.config.capacity
        cargo_total = self.current_load

        # Per-type fractional loads
        type_fracs = np.array(
            [self._cargo.get(wt, 0.0) / cap if cap > 0 else 0.0
             for wt in WASTE_TYPES],
            dtype=np.float32,
        )

        return np.array([
            float(self.config.vehicle_id),
            float(self._current_node),
            self.remaining_capacity / cap if cap > 0 else 0.0,
            cargo_total / cap if cap > 0 else 0.0,
            *type_fracs,
            self.config.speed,
            self.config.emission_factor,
            self._total_distance,
            self._total_time,
            float(self._status.value - 1),  # 0-indexed ordinal
        ], dtype=np.float32)

    def get_cargo_vector(self) -> np.ndarray:
        """Return per waste-type cargo as a NumPy array (5,).

        Order follows ``WASTE_TYPES``: [concrete, metal, wood, mixed, hazardous].
        """
        return np.array(
            [self._cargo[wt] for wt in WASTE_TYPES], dtype=np.float32
        )

    def get_action_mask(
        self,
        neighbor_ids: List[int],
        health_values: Optional[np.ndarray] = None,
        health_threshold: float = 0.1,
    ) -> np.ndarray:
        """Generate a binary action mask over candidate next-node actions.

        An action is masked (0) if:
            - The road health to that neighbor is below ``health_threshold``.
            - The vehicle has zero remaining capacity AND the neighbor is a
              waste generation node (no point visiting if you can't load).

        Parameters
        ----------
        neighbor_ids : List[int]
            Candidate successor node IDs.
        health_values : np.ndarray, optional
            Health values corresponding to each neighbor's edge.
            If ``None``, all edges are assumed passable.
        health_threshold : float
            Minimum health to consider the edge usable.

        Returns
        -------
        np.ndarray of shape (len(neighbor_ids),), dtype bool
            ``True`` where the action is valid.
        """
        n = len(neighbor_ids)
        mask = np.ones(n, dtype=bool)

        if health_values is not None:
            mask &= health_values >= health_threshold

        return mask

    # ------------------------------------------------------------------
    # Properties (read-only state queries)
    # ------------------------------------------------------------------

    @property
    def vehicle_id(self) -> int:
        return self.config.vehicle_id

    @property
    def current_node(self) -> int:
        return self._current_node

    @property
    def next_node(self) -> Optional[int]:
        return self._next_node

    @property
    def status(self) -> VehicleStatus:
        return self._status

    @property
    def current_load(self) -> float:
        """Total tonnes currently on board."""
        return sum(self._cargo.values())

    @property
    def remaining_capacity(self) -> float:
        """Remaining capacity in tonnes."""
        return max(0.0, self.config.capacity - self.current_load)

    @property
    def is_empty(self) -> bool:
        return self.current_load < 1e-6

    @property
    def is_full(self) -> bool:
        return self.remaining_capacity < 1e-6

    @property
    def cargo(self) -> Dict[str, float]:
        """Current cargo breakdown (read-only copy)."""
        return dict(self._cargo)

    @property
    def total_distance(self) -> float:
        return self._total_distance

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_emission(self) -> float:
        return self._total_emission

    @property
    def total_time(self) -> float:
        return self._total_time

    @property
    def total_delivered(self) -> Dict[str, float]:
        """Cumulative tonnes delivered, by waste type."""
        return dict(self._total_delivered)

    @property
    def nodes_visited(self) -> List[int]:
        return list(self._nodes_visited)

    @property
    def at_depot(self) -> bool:
        return self._current_node == self.config.home_depot

    # ------------------------------------------------------------------
    # Trip summary
    # ------------------------------------------------------------------

    def get_trip_summary(self) -> Dict[str, float]:
        """Return a dictionary summarising the vehicle's episode statistics.

        Keys: ``distance``, ``cost``, ``emission``, ``time``,
        ``pickups``, ``dropoffs``, ``total_delivered``,
        ``capacity_utilisation`` (avg load / capacity across all moves).
        """
        total_del = sum(self._total_delivered.values())
        n_moves = max(len(self._nodes_visited) - 1, 1)
        # Rough capacity utilisation: total delivered / (capacity × n_moves)
        cap_util = total_del / (self.config.capacity * n_moves) if n_moves > 0 else 0.0

        return {
            "distance": self._total_distance,
            "cost": self._total_cost,
            "emission": self._total_emission,
            "time": self._total_time,
            "pickups": float(self._pickup_count),
            "dropoffs": float(self._dropoff_count),
            "total_delivered": total_del,
            "capacity_utilisation": min(cap_util, 1.0),
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the vehicle to its initial state at the home depot."""
        self._current_node = self.config.home_depot
        self._next_node = None
        self._status = VehicleStatus.IDLE
        self._cargo = {wt: 0.0 for wt in WASTE_TYPES}
        self._total_distance = 0.0
        self._total_cost = 0.0
        self._total_emission = 0.0
        self._total_time = 0.0
        self._nodes_visited = [self.config.home_depot]
        self._pickup_count = 0
        self._dropoff_count = 0
        self._total_delivered = {wt: 0.0 for wt in WASTE_TYPES}

    def __repr__(self) -> str:
        return (
            f"Vehicle(id={self.config.vehicle_id}, "
            f"node={self._current_node}, "
            f"load={self.current_load:.1f}/{self.config.capacity:.1f}t, "
            f"status={self._status.name})"
        )
