"""
Canopy SDK — Collaborative Spatial Awareness for Multi-Robot Fleets.

Each robot publishes structured observations into a shared WorldModel.
Every robot's effective perception = the fleet's combined FOV.

Quick start:
    from canopy.world_model import WorldModel
    from canopy.queries import QueryEngine
"""

from canopy.models import (
    Pose2D,
    RobotState,
    DetectedObject,
    FreeSpaceCell,
    ObservationReport,
    OccupancyCellState,
    WorldState,
    QueryRequest,
    QueryResult,
)
from canopy.world_model import WorldModel
from canopy.queries import QueryEngine

__all__ = [
    "Pose2D",
    "RobotState",
    "DetectedObject",
    "FreeSpaceCell",
    "ObservationReport",
    "OccupancyCellState",
    "WorldState",
    "QueryRequest",
    "QueryResult",
    "WorldModel",
    "QueryEngine",
]
