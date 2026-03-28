"""
disaster_waste_env.py — PettingZoo ParallelEnv for Disaster Waste Logistics
============================================================================

This module implements the central multi-agent reinforcement learning
environment by wrapping the four infrastructure modules:

    - ``DisasterNetwork``      → dynamic road graph
    - ``WasteGenerationModel`` → stochastic waste sources
    - ``Vehicle``              → heterogeneous agent fleet
    - ``ScenarioGenerator``    → parametric scenario factory

The environment follows the **PettingZoo Parallel API** (v1.25), where
all agents act simultaneously in each time step.

Key design decisions
--------------------
1. **Observation**: Per-agent local observation (vehicle state 14-D +
   padded local road health + top-k nearest waste storages) and a global
   state vector for the centralised critic (CTDE).
2. **Action**: Discrete space with action masking.  Actions cover
   movement to neighbours, waste pickup, facility dropoff, and wait.
3. **Reward**: Multi-objective weighted sum (§4.4 of the plan):
   cost + time + emission penalty, recycling bonus, task rewards, and
   constraint-violation penalties.
4. **Dynamics**: Each ``step()`` advances the Poisson road-damage
   process and the stochastic waste generator.

Author  : Muhammed Şara
License : MIT
"""

from __future__ import annotations

import functools
from typing import Any, Dict, List, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from .network import DisasterNetwork, NodeType
from .vehicle import Vehicle, VehicleConfig, VehicleStatus
from .waste_model import WasteGenerationModel, WASTE_TYPES
from .scenario_generator import (
    Scenario,
    ScenarioConfig,
    ScenarioGenerator,
    ScenarioTier,
)


# ---------------------------------------------------------------------------
# Action encoding
# ---------------------------------------------------------------------------

class ActionType:
    """Semantic labels for the discrete action encoding.

    For a node with degree-out D, the action space has size D + 3:
        [0 .. D-1]  →  Move to the i-th neighbour
        D           →  PICKUP waste at current node
        D + 1       →  DROPOFF waste at current node
        D + 2       →  WAIT (do nothing)

    Because max degree varies per node and per time step, we fix the
    action space to ``max_degree + 3`` globally and mask invalid slots.
    """
    PICKUP_OFFSET: int = 0   # index = max_degree + 0
    DROPOFF_OFFSET: int = 1  # index = max_degree + 1
    WAIT_OFFSET: int = 2     # index = max_degree + 2


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class DisasterWasteEnv(ParallelEnv):
    """Multi-agent PettingZoo environment for disaster waste logistics.

    Parameters
    ----------
    scenario : Scenario, optional
        Pre-built scenario.  If ``None``, a default S2_MEDIUM scenario
        is generated with the provided ``seed``.
    scenario_tier : ScenarioTier, optional
        Tier to auto-generate if ``scenario`` is not provided.
    reward_weights : Dict[str, float], optional
        Keys: ``"cost"``, ``"time"``, ``"emission"``, ``"recycling"``.
        Default: uniform (0.25 each).
    top_k_waste : int
        Number of nearest waste-generation nodes included in each
        agent's observation.  Default 5.
    penalty_capacity : float
        Penalty for capacity violation (should not occur with masking).
    penalty_time_window : float
        Penalty per step when waste exceeds accumulation threshold.
    waste_threshold_factor : float
        Waste accumulation threshold = initial_expected × this factor.
    task_reward : float
        Bonus for a successful waste dropoff at a facility.
    seed : int
        Random seed for scenario generation and dynamics.

    Attributes
    ----------
    metadata : dict
        PettingZoo metadata.
    possible_agents : list[str]
        Agent string IDs (``"vehicle_0"``, ``"vehicle_1"``, …).
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "disaster_waste_v0",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        scenario: Optional[Scenario] = None,
        scenario_tier: ScenarioTier = ScenarioTier.S2_MEDIUM,
        reward_weights: Optional[Dict[str, float]] = None,
        top_k_waste: int = 5,
        penalty_capacity: float = 10.0,
        penalty_time_window: float = 0.5,
        waste_threshold_factor: float = 2.0,
        task_reward: float = 5.0,
        seed: int = 42,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        # --- Build or reuse scenario ---
        if scenario is not None:
            self._scenario = scenario
        else:
            gen = ScenarioGenerator(seed=seed)
            self._scenario = gen.from_tier(scenario_tier)

        self._seed = seed
        self.render_mode = render_mode

        # --- Reward configuration ---
        rw = reward_weights or {}
        self._w_cost: float = rw.get("cost", 0.25)
        self._w_time: float = rw.get("time", 0.25)
        self._w_emission: float = rw.get("emission", 0.25)
        self._w_recycling: float = rw.get("recycling", 0.25)

        self._penalty_capacity: float = penalty_capacity
        self._penalty_time_window: float = penalty_time_window
        self._waste_threshold_factor: float = waste_threshold_factor
        self._task_reward: float = task_reward

        # --- Top-k observation parameter ---
        self._top_k: int = top_k_waste

        # --- Internal references (populated on reset) ---
        self._network: DisasterNetwork = self._scenario.network
        self._waste_model: WasteGenerationModel = self._scenario.waste_model
        self._vehicles: List[Vehicle] = self._scenario.vehicles
        self._max_steps: int = self._scenario.config.max_time_steps
        self._current_step: int = 0

        # --- Agent naming ---
        self.possible_agents: List[str] = [
            f"vehicle_{v.config.vehicle_id}" for v in self._vehicles
        ]
        self.agents: List[str] = list(self.possible_agents)

        # Agent-name → Vehicle mapping
        self._agent_vehicle: Dict[str, Vehicle] = {
            name: veh for name, veh in zip(self.possible_agents, self._vehicles)
        }

        # --- Pre-compute topology constants ---
        self._max_degree: int = self._compute_max_degree()
        self._action_size: int = self._max_degree + 3  # +PICKUP +DROPOFF +WAIT
        self._node_list: List[int] = sorted(self._network.graph.nodes)
        self._node_to_idx: Dict[int, int] = {
            n: i for i, n in enumerate(self._node_list)
        }

        # --- Waste accumulation thresholds ---
        # Compute once: expected waste at t=0 per node × factor
        gen_ids = self._network.get_nodes_by_type(NodeType.WASTE_GENERATION)
        rates_t0 = self._waste_model.get_generation_rate_vector(t=0)
        self._waste_thresholds: Dict[int, float] = {}
        for idx, nid in enumerate(sorted(self._waste_model.node_ids)):
            self._waste_thresholds[nid] = float(rates_t0[idx]) * waste_threshold_factor

        # --- Observation / action space dimensions ---
        # Local obs: vehicle(14) + local_health(max_degree) + top_k_storage(top_k)
        self._local_obs_dim: int = 14 + self._max_degree + self._top_k
        self._global_state_dim: int = self._compute_global_state_dim()

        # --- Facility remaining capacities ---
        self._facility_remaining: Dict[int, float] = {}

    # ==================================================================
    # PettingZoo API: spaces
    # ==================================================================

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Dict:
        """Per-agent observation space.

        Returns a Dict space with:
            - ``"obs"``: Box of shape (local_obs_dim,)
            - ``"action_mask"``: MultiBinary of shape (action_size,)
            - ``"global_state"``: Box of shape (global_state_dim,)

        The ``"global_state"`` key is for the centralised critic (CTDE);
        during decentralised execution only ``"obs"`` and ``"action_mask"``
        are used by each actor.
        """
        return spaces.Dict({
            "obs": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self._local_obs_dim,), dtype=np.float32,
            ),
            "action_mask": spaces.MultiBinary(self._action_size),
            "global_state": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self._global_state_dim,), dtype=np.float32,
            ),
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Discrete:
        """Discrete action space: max_degree movement + pickup + dropoff + wait."""
        return spaces.Discrete(self._action_size)

    # ==================================================================
    # PettingZoo API: reset
    # ==================================================================

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, dict], Dict[str, dict]]:
        """Reset the environment to its initial state.

        Parameters
        ----------
        seed : int, optional
            If provided, re-seeds the RNG for a new episode.
        options : dict, optional
            Reserved for future use.

        Returns
        -------
        (observations, infos) : tuple
            Initial observations and info dicts for every agent.
        """
        effective_seed = seed if seed is not None else self._seed

        # Reset components
        self._network.reset(seed=effective_seed)
        self._waste_model.reset(seed=effective_seed + 1)
        for veh in self._vehicles:
            veh.reset()

        self._current_step = 0
        self.agents = list(self.possible_agents)

        # Initialise facility capacities from network node data
        self._facility_remaining = {}
        for ntype in (NodeType.SORTING_FACILITY, NodeType.LANDFILL, NodeType.TCP):
            for nid in self._network.get_nodes_by_type(ntype):
                cap = self._network.graph.nodes[nid].get("capacity", 500.0)
                self._facility_remaining[nid] = cap

        # Generate initial waste (t=0)
        self._waste_model.step(t=0)

        # Build observations
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}

        return observations, infos

    # ==================================================================
    # PettingZoo API: step
    # ==================================================================

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[
        Dict[str, dict],       # observations
        Dict[str, float],      # rewards
        Dict[str, bool],       # terminations
        Dict[str, bool],       # truncations
        Dict[str, dict],       # infos
    ]:
        """Execute one time step for all agents simultaneously.

        Pipeline per step
        -----------------
        1. Decode and execute each agent's action (move / pickup / dropoff / wait).
        2. Advance network dynamics (Poisson damage + repair).
        3. Advance waste generation (stochastic production).
        4. Compute multi-objective rewards.
        5. Check termination / truncation conditions.
        6. Build new observations.

        Parameters
        ----------
        actions : dict[str, int]
            Mapping of agent name → discrete action index.

        Returns
        -------
        (observations, rewards, terminations, truncations, infos)
        """
        rewards: Dict[str, float] = {agent: 0.0 for agent in self.agents}
        step_costs: Dict[str, float] = {agent: 0.0 for agent in self.agents}
        step_times: Dict[str, float] = {agent: 0.0 for agent in self.agents}
        step_emissions: Dict[str, float] = {agent: 0.0 for agent in self.agents}
        step_recycling: Dict[str, float] = {agent: 0.0 for agent in self.agents}

        # ---- 1. Execute actions ----
        for agent in self.agents:
            if agent not in actions:
                continue
            action = actions[agent]
            veh = self._agent_vehicle[agent]
            neighbours = self._network.get_neighbors(veh.current_node)

            if action < self._max_degree:
                # --- MOVE to the action-th neighbour ---
                if action < len(neighbours):
                    target = neighbours[action]
                    edge_data = self._network.graph.edges[veh.current_node, target]
                    distance = edge_data["distance"]
                    travel_time = self._network.get_travel_time(
                        veh.current_node, target
                    )
                    trip = veh.move_to(target, distance, travel_time)
                    step_costs[agent] += trip["cost"]
                    step_times[agent] += trip["time"]
                    step_emissions[agent] += trip["emission"]
                # else: invalid move (masked), treated as wait

            elif action == self._max_degree + ActionType.PICKUP_OFFSET:
                # --- PICKUP waste at current generation node ---
                self._execute_pickup(agent, veh)

            elif action == self._max_degree + ActionType.DROPOFF_OFFSET:
                # --- DROPOFF waste at current facility ---
                recycled = self._execute_dropoff(agent, veh)
                step_recycling[agent] += recycled
                if recycled > 0:
                    rewards[agent] += self._task_reward

            # else: WAIT — do nothing (action == max_degree + WAIT_OFFSET)

        # ---- 2. Advance network dynamics ----
        self._network.step_dynamics()

        # ---- 3. Advance waste generation ----
        self._current_step += 1
        if self._current_step < self._max_steps:
            self._waste_model.step(t=self._current_step)

        # ---- 4. Compute multi-objective rewards ----
        for agent in self.agents:
            rewards[agent] += self._compute_reward(
                agent,
                step_costs[agent],
                step_times[agent],
                step_emissions[agent],
                step_recycling[agent],
            )

        # ---- 5. Termination / truncation ----
        truncated = self._current_step >= self._max_steps
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}

        # ---- 6. Build observations and infos ----
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}

        # Add step-level metrics to infos
        for agent in self.agents:
            infos[agent].update({
                "step_cost": step_costs[agent],
                "step_time": step_times[agent],
                "step_emission": step_emissions[agent],
                "step_recycling": step_recycling[agent],
            })

        return observations, rewards, terminations, truncations, infos

    # ==================================================================
    # PettingZoo API: state (for centralised critic)
    # ==================================================================

    def state(self) -> np.ndarray:
        """Return the global state vector for the centralised critic.

        Composition (concatenated 1-D vector):
            1. All edge health values          (|E| floats)
            2. All vehicle observation vectors  (K × 14 floats)
            3. All waste-node storages          (|N_gen| floats)
            4. All facility remaining caps      (|N_fac| floats)
            5. Normalised time step             (1 float)

        Returns
        -------
        np.ndarray of shape (global_state_dim,)
        """
        parts: List[np.ndarray] = []

        # Edge healths
        parts.append(self._network.get_edge_health_vector().astype(np.float32))

        # Vehicle states
        for veh in self._vehicles:
            parts.append(veh.get_observation_vector())

        # Waste storages
        parts.append(self._waste_model.get_storage_vector())

        # Facility remaining capacities
        fac_caps = np.array(
            [self._facility_remaining.get(nid, 0.0)
             for nid in sorted(self._facility_remaining)],
            dtype=np.float32,
        )
        parts.append(fac_caps)

        # Normalised time
        t_norm = np.array(
            [self._current_step / max(self._max_steps, 1)], dtype=np.float32
        )
        parts.append(t_norm)

        return np.concatenate(parts)

    # ==================================================================
    # PettingZoo API: render
    # ==================================================================

    def render(self) -> Optional[str]:
        """Render the environment state (ANSI text mode)."""
        if self.render_mode == "ansi" or self.render_mode == "human":
            lines = [
                f"=== DisasterWasteEnv  t={self._current_step}/{self._max_steps} ===",
                f"Network: {self._network}",
                f"Pending waste: {self._waste_model.get_total_pending_waste():.1f}t",
            ]
            for veh in self._vehicles:
                lines.append(f"  {veh}")
            text = "\n".join(lines)
            if self.render_mode == "human":
                print(text)
            return text
        return None

    def close(self) -> None:
        """Clean up resources."""
        pass

    # ==================================================================
    # Internal: observation construction
    # ==================================================================

    def _get_obs(self, agent: str) -> Dict[str, np.ndarray]:
        """Build the observation dict for a single agent.

        Observation layout (``"obs"`` key):
        ┌──────────────────────────────────────────────┐
        │ [0..13]       Vehicle state (14-D)           │
        │ [14..14+D-1]  Local road health (padded)     │
        │ [14+D..end]   Top-k nearest waste storages   │
        └──────────────────────────────────────────────┘

        ``"action_mask"`` key:
            Binary mask of size (action_size,).

        ``"global_state"`` key:
            Full state vector (for centralised critic).
        """
        veh = self._agent_vehicle[agent]

        # --- 1. Vehicle observation ---
        veh_obs = veh.get_observation_vector()  # (14,)

        # --- 2. Local road health (padded to max_degree) ---
        local_health = self._get_padded_local_health(veh.current_node)

        # --- 3. Top-k nearest waste storages ---
        top_k_storage = self._get_topk_waste_storage(veh.current_node)

        # Concatenate into single obs vector
        obs = np.concatenate([veh_obs, local_health, top_k_storage])

        return {
            "obs": obs.astype(np.float32),
            "action_mask": self._get_action_mask(agent),
            "global_state": self.state(),
        }

    def _get_padded_local_health(self, node_id: int) -> np.ndarray:
        """Return outgoing edge healths padded to ``max_degree``.

        If the node has fewer outgoing edges than max_degree, remaining
        slots are filled with 0.0 (indicating no road).
        """
        neighbours = self._network.get_neighbors(node_id)
        health = np.zeros(self._max_degree, dtype=np.float32)
        for i, nbr in enumerate(neighbours):
            if i >= self._max_degree:
                break
            health[i] = self._network.get_edge_health(node_id, nbr)
        return health

    def _get_topk_waste_storage(self, node_id: int) -> np.ndarray:
        """Return storages of the k nearest waste-generation nodes.

        Sorted by Euclidean distance from ``node_id``.  If fewer than k
        generation nodes exist, remainder is zero-padded.
        """
        gen_ids = sorted(self._waste_model.node_ids)
        if not gen_ids:
            return np.zeros(self._top_k, dtype=np.float32)

        pos_cur = np.array(self._network.get_node_position(node_id))

        # Compute distances to all generation nodes
        dists_and_storage: List[Tuple[float, float]] = []
        for gid in gen_ids:
            pos_g = np.array(self._network.get_node_position(gid))
            dist = float(np.linalg.norm(pos_cur - pos_g))
            storage = self._waste_model.get_node_storage(gid)
            dists_and_storage.append((dist, storage))

        # Sort by distance, take top-k
        dists_and_storage.sort(key=lambda x: x[0])
        result = np.zeros(self._top_k, dtype=np.float32)
        for i in range(min(self._top_k, len(dists_and_storage))):
            result[i] = dists_and_storage[i][1]

        return result

    # ==================================================================
    # Internal: action masking
    # ==================================================================

    def _get_action_mask(self, agent: str) -> np.ndarray:
        """Compute binary action mask for an agent.

        Mask layout (size = max_degree + 3):
            [0 .. max_degree-1]  : 1 if neighbour exists AND road health ≥ threshold
            [max_degree]         : 1 if at a WASTE_GENERATION node AND not full
            [max_degree + 1]     : 1 if at a facility/TCP AND carrying cargo
            [max_degree + 2]     : 1 always (WAIT is always valid)
        """
        veh = self._agent_vehicle[agent]
        mask = np.zeros(self._action_size, dtype=np.int8)

        neighbours = self._network.get_neighbors(veh.current_node)

        # Movement actions
        for i, nbr in enumerate(neighbours):
            if i >= self._max_degree:
                break
            health = self._network.get_edge_health(veh.current_node, nbr)
            if health >= 0.1:  # passable threshold
                mask[i] = 1

        # PICKUP: valid if at generation node and vehicle not full
        node_type = self._network.get_node_type(veh.current_node)
        if node_type == NodeType.WASTE_GENERATION and not veh.is_full:
            # Also check there's actually waste to pick up
            if (veh.current_node in self._waste_model.node_ids and
                    self._waste_model.get_node_storage(veh.current_node) > 0.01):
                mask[self._max_degree + ActionType.PICKUP_OFFSET] = 1

        # DROPOFF: valid if at facility/TCP/landfill and carrying cargo
        if node_type in (NodeType.SORTING_FACILITY, NodeType.LANDFILL, NodeType.TCP):
            if not veh.is_empty:
                # Check facility has remaining capacity
                fac_cap = self._facility_remaining.get(veh.current_node, 0.0)
                if fac_cap > 0.01:
                    mask[self._max_degree + ActionType.DROPOFF_OFFSET] = 1

        # WAIT: always valid
        mask[self._max_degree + ActionType.WAIT_OFFSET] = 1

        return mask

    # ==================================================================
    # Internal: action execution
    # ==================================================================

    def _execute_pickup(self, agent: str, veh: Vehicle) -> float:
        """Execute a waste pickup action.

        The vehicle loads as much compatible waste as its remaining
        capacity allows from the current generation node's storage.

        Returns
        -------
        float
            Total tonnes loaded.
        """
        nid = veh.current_node
        if nid not in self._waste_model.node_ids:
            return 0.0

        available_storage = self._waste_model.get_node_storage(nid)
        if available_storage < 0.01:
            return 0.0

        remaining_cap = veh.remaining_capacity
        if remaining_cap < 0.01:
            return 0.0

        # Get per-type proportions at this node
        prop_matrix = self._waste_model.get_waste_proportion_matrix()
        sorted_ids = sorted(self._waste_model.node_ids)
        node_idx = sorted_ids.index(nid)
        proportions = prop_matrix[node_idx]

        # Amount to pick up (limited by vehicle capacity and available waste)
        total_pickup = min(remaining_cap, available_storage)

        # Break into per-type amounts
        waste_amounts: Dict[str, float] = {}
        for i, wtype in enumerate(WASTE_TYPES):
            waste_amounts[wtype] = total_pickup * float(proportions[i])

        # Execute pickup on vehicle (respects compatibility)
        loaded = veh.pickup(waste_amounts, strict=False)

        # Remove from node storage
        self._waste_model.collect_waste(nid, loaded)

        return loaded

    def _execute_dropoff(self, agent: str, veh: Vehicle) -> float:
        """Execute a waste dropoff action at a facility.

        The recycling reward depends on the facility type:
            - SORTING_FACILITY: applies per-type recycling rates
            - LANDFILL: zero recycling (disposal only)
            - TCP: partial recycling (intermediate storage)

        Returns
        -------
        float
            Recycled quantity (tonnes), used for the recycling reward term.
        """
        nid = veh.current_node
        node_type = self._network.get_node_type(nid)

        # Check facility capacity
        fac_cap = self._facility_remaining.get(nid, 0.0)
        if fac_cap < 0.01:
            return 0.0

        # Get cargo and limit by facility capacity
        cargo = veh.cargo
        total_cargo = sum(cargo.values())
        if total_cargo < 0.01:
            return 0.0

        # Scale down if facility can't absorb all cargo
        scale = min(1.0, fac_cap / total_cargo) if total_cargo > 0 else 0.0

        if scale < 1.0:
            # Partial dropoff: scale all types proportionally
            scaled_cargo: Dict[str, float] = {
                wt: qty * scale for wt, qty in cargo.items()
            }
            # Remove scaled amounts from vehicle
            for wt, qty in scaled_cargo.items():
                if qty > 0:
                    veh.partial_dropoff(wt, qty)
            delivered = scaled_cargo
        else:
            # Full dropoff
            delivered = veh.dropoff()

        actual_delivered = sum(delivered.values())
        self._facility_remaining[nid] = max(0.0, fac_cap - actual_delivered)

        # Compute recycling based on facility type
        # Hazardous waste priority: 2x recycling reward
        recycled = 0.0
        if node_type == NodeType.SORTING_FACILITY:
            rec_rates = self._network.graph.nodes[nid].get("recycling_rates", {})
            for wtype, qty in delivered.items():
                rate = rec_rates.get(wtype, 0.0)
                hazard_mult = 2.0 if wtype == "hazardous" else 1.0
                recycled += qty * rate * hazard_mult
        elif node_type == NodeType.TCP:
            # TCPs provide modest interim recycling (20%)
            for wtype, qty in delivered.items():
                hazard_mult = 2.0 if wtype == "hazardous" else 1.0
                recycled += qty * 0.2 * hazard_mult
        # LANDFILL: recycled stays 0.0

        return recycled

    # ==================================================================
    # Internal: reward computation
    # ==================================================================

    def _compute_reward(
        self,
        agent: str,
        cost: float,
        time: float,
        emission: float,
        recycling: float,
    ) -> float:
        """Compute the multi-objective reward for a single agent.

        The reward function implements the formulation from implementation
        plan §4.4:

            r(t) = − ω_c · cost − ω_τ · time − ω_e · emission
                   + ω_r · recycling
                   + r_penalty(t)

        The penalty term penalises:
            - (implicit) Capacity violations — prevented by action masking
            - Waste accumulation above threshold at generation nodes

        Normalisation
        -------------
        Raw cost/time/emission values are divided by scenario-specific
        reference magnitudes to keep the reward components on similar
        scales.  This avoids one component dominating simply because
        its raw unit is larger.

        Parameters
        ----------
        agent : str
            Agent name.
        cost, time, emission, recycling : float
            Step-level magnitudes.

        Returns
        -------
        float
            Scalar reward (negative for costs, positive for recycling).
        """
        # Reference magnitudes for normalisation (approximate per-step scale)
        # These prevent reward-component scale imbalance.
        cost_ref = max(self._scenario.config.area_size * 0.5, 1.0)
        time_ref = max(self._scenario.config.area_size /
                       np.mean([v.config.speed for v in self._vehicles]), 0.1)

        # Carbon Tax: $50 per tonne CO2 → $0.05 per kg CO2
        _CARBON_TAX_RATE = 0.05  # $/kg
        emission_cost = emission * _CARBON_TAX_RATE

        # Weighted sum (emission penalised as economic cost)
        reward = (
            - self._w_cost     * (cost / cost_ref)
            - self._w_time     * (time / time_ref)
            - self._w_emission * (emission_cost / cost_ref)
            + self._w_recycling * (recycling / max(cost_ref, 1.0))
        )

        # --- Penalty: excess waste accumulation ---
        # Penalise if any generation node's storage exceeds threshold
        penalty = 0.0
        for nid, threshold in self._waste_thresholds.items():
            storage = self._waste_model.get_node_storage(nid)
            excess = max(0.0, storage - threshold)
            if excess > 0:
                penalty += self._penalty_time_window * (excess / max(threshold, 1.0))

        # Distribute penalty equally among all agents
        n_agents = max(len(self.agents), 1)
        reward -= penalty / n_agents

        return reward

    # ==================================================================
    # Internal: info dict
    # ==================================================================

    def _get_info(self, agent: str) -> Dict[str, Any]:
        """Build the info dict for an agent."""
        veh = self._agent_vehicle[agent]
        return {
            "vehicle_id": veh.config.vehicle_id,
            "current_node": veh.current_node,
            "current_load": veh.current_load,
            "remaining_capacity": veh.remaining_capacity,
            "total_distance": veh.total_distance,
            "total_emission": veh.total_emission,
            "status": veh.status.name,
            "step": self._current_step,
            "avg_network_health": self._network.average_health,
            "total_pending_waste": self._waste_model.get_total_pending_waste(),
        }

    # ==================================================================
    # Internal: helpers
    # ==================================================================

    def _compute_max_degree(self) -> int:
        """Compute the maximum out-degree across all nodes in the network."""
        return max(
            (self._network.graph.out_degree(n) for n in self._network.graph.nodes),
            default=1,
        )

    def _compute_global_state_dim(self) -> int:
        """Compute the dimension of the global state vector.

        Composition:
            edges health    : |E|
            vehicle states  : K × 14
            waste storages  : |N_gen|
            facility caps   : |N_fac|
            time normalised : 1
        """
        n_edges = self._network.num_edges
        n_vehicles = len(self._vehicles)
        n_gen = self._waste_model.num_nodes

        # Count facility nodes (sorting + landfill + TCP)
        n_fac = (
            len(self._network.get_nodes_by_type(NodeType.SORTING_FACILITY))
            + len(self._network.get_nodes_by_type(NodeType.LANDFILL))
            + len(self._network.get_nodes_by_type(NodeType.TCP))
        )

        return n_edges + (n_vehicles * 14) + n_gen + n_fac + 1

    # ==================================================================
    # Public convenience methods
    # ==================================================================

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def scenario(self) -> Scenario:
        return self._scenario

    def get_fleet_summary(self) -> Dict[str, Dict[str, float]]:
        """Return trip summaries for all vehicles."""
        return {
            f"vehicle_{v.config.vehicle_id}": v.get_trip_summary()
            for v in self._vehicles
        }

    def get_episode_metrics(self) -> Dict[str, float]:
        """Compute aggregate episode-level KPIs.

        Returns
        -------
        dict with keys: ``total_cost``, ``total_time``, ``total_emission``,
        ``total_delivered``, ``avg_capacity_utilisation``,
        ``remaining_waste``, ``service_level``.
        """
        summaries = [v.get_trip_summary() for v in self._vehicles]

        total_delivered = sum(s["total_delivered"] for s in summaries)
        total_generated = self._waste_model.get_generation_summary()["total_generated"]
        remaining = self._waste_model.get_total_pending_waste()

        service_level = (
            (total_generated - remaining) / total_generated
            if total_generated > 0 else 0.0
        )

        return {
            "total_cost": sum(s["cost"] for s in summaries),
            "total_time": sum(s["time"] for s in summaries),
            "total_emission": sum(s["emission"] for s in summaries),
            "total_delivered": total_delivered,
            "avg_capacity_utilisation": float(
                np.mean([s["capacity_utilisation"] for s in summaries])
            ),
            "remaining_waste": remaining,
            "service_level": service_level,
        }

    def __repr__(self) -> str:
        return (
            f"DisasterWasteEnv(agents={len(self.agents)}, "
            f"step={self._current_step}/{self._max_steps}, "
            f"tier={self._scenario.config.tier.name})"
        )
