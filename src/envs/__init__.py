"""
Disaster Waste Management — Simulation Environment Package

This package provides the core simulation components:
    - network: Dynamic disaster road network model
    - waste_model: Stochastic time-varying waste generation engine
    - vehicle: Heterogeneous vehicle agent model
    - scenario_generator: Parametric disaster scenario factory
    - disaster_waste_env: PettingZoo-compatible multi-agent environment
"""

from .network import DisasterNetwork, NodeType, NodeAttributes, EdgeAttributes
from .waste_model import WasteGenerationModel, WasteNodeConfig, WASTE_TYPES
from .vehicle import Vehicle, VehicleConfig, VehicleStatus
from .scenario_generator import ScenarioGenerator, ScenarioTier, ScenarioConfig, Scenario
from .disaster_waste_env import DisasterWasteEnv
