"""
SkyNet SDK — Collaborative Spatial Awareness for Multi-Robot Fleets.

Each robot publishes structured observations into a shared WorldModel.
Every robot's effective perception = the fleet's combined FOV.

Quick start:
    from skynet.world_model import WorldModel
    from skynet.simulation import Robot, Obstacle, WorldObject, SimWorld
    from skynet.queries import QueryEngine

    See skynet/main.py for a full demo.
"""

from skynet.models import (
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
from skynet.world_model import WorldModel
from skynet.queries import QueryEngine

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
