"""
Data contracts for the SkyNet SDK.

All models use Pydantic v2. These are the canonical types passed between
robots, the WorldModel, the fusion engine, and the query layer.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator


# ---------------------------------------------------------------------------
# Spatial primitives
# ---------------------------------------------------------------------------


class Pose2D(BaseModel):
    """2D position + heading in world coordinates (meters, radians)."""

    model_config = ConfigDict(frozen=False)

    x: float
    y: float
    heading_rad: float = 0.0  # 0 = east (positive X), CCW positive

    def to_grid(self, cell_size: float = 0.5) -> tuple[int, int]:
        """Convert world coords to grid cell indices."""
        return (int(self.x / cell_size), int(self.y / cell_size))

    def distance_to(self, other: Pose2D) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


# ---------------------------------------------------------------------------
# Grid state
# ---------------------------------------------------------------------------


class OccupancyCellState(str, Enum):
    UNKNOWN = "UNKNOWN"
    FREE = "FREE"
    OCCUPIED = "OCCUPIED"


# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------


class RobotState(BaseModel):
    """Last known state of a robot as seen by the WorldModel."""

    model_config = ConfigDict(frozen=False)

    robot_id: str
    pose: Pose2D
    timestamp: float  # Unix epoch seconds
    is_active: bool = True


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


class DetectedObject(BaseModel):
    """A single object detection reported by a robot."""

    model_config = ConfigDict(frozen=False)

    object_id: str          # Stable ID; for POC: hash of class+grid_cell
    class_label: str        # e.g. "box", "person", "forklift"
    position: Pose2D        # Center of object in world coords
    confidence: float       # [0, 1]
    observed_by: str        # robot_id that made this detection
    timestamp: float        # Unix epoch seconds
    width: float = 0.5      # meters
    height: float = 0.5     # meters

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {v}")
        return v


class FreeSpaceCell(BaseModel):
    """A grid cell confirmed to be free (no obstacles, no objects)."""

    model_config = ConfigDict(frozen=False)

    grid_x: int
    grid_y: int
    confidence: float       # [0, 1]
    reported_by: str        # robot_id
    timestamp: float        # Unix epoch seconds

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {v}")
        return v


class ObservationReport(BaseModel):
    """
    A robot's complete observation for one sensing tick.

    Contains all detected objects and all cells confirmed free within
    the robot's current FOV. Published to the WorldModel via ingest().
    """

    model_config = ConfigDict(frozen=False)

    robot_id: str
    timestamp: float
    robot_pose: Pose2D
    detected_objects: list[DetectedObject] = []
    free_cells: list[FreeSpaceCell] = []
    fov_range: float = 6.0          # meters
    fov_angle_rad: float = 2.094    # radians (~120°)


# ---------------------------------------------------------------------------
# World state snapshot
# ---------------------------------------------------------------------------


class WorldState(BaseModel):
    """
    Serializable snapshot of the entire fused world model.

    The grid is stored as a nested list so it serializes cleanly to JSON.
    Use WorldModel.get_grid_snapshot() for the raw numpy arrays.
    """

    model_config = ConfigDict(frozen=False)

    timestamp: float
    objects: dict[str, DetectedObject] = {}
    robot_states: dict[str, RobotState] = {}
    grid_width: int = 30
    grid_height: int = 30
    # 0=UNKNOWN, 1=FREE, 2=OCCUPIED
    grid_data: list[list[int]] = []
    confidence_data: list[list[float]] = []


# ---------------------------------------------------------------------------
# Query interface
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Parameters for querying the shared world model."""

    model_config = ConfigDict(frozen=False)

    robot_id: str
    # Optional spatial filter: (x_min, y_min, x_max, y_max) in world meters
    bbox: Optional[tuple[float, float, float, float]] = None
    object_class: Optional[str] = None
    max_age_seconds: float = 60.0
    min_confidence: float = 0.1


class QueryResult(BaseModel):
    """Result returned by QueryEngine.query()."""

    model_config = ConfigDict(frozen=False)

    objects: list[DetectedObject] = []
    free_cells: list[FreeSpaceCell] = []
    occupied_cells: list[tuple[int, int]] = []
    robot_positions: dict[str, Pose2D] = {}
    query_timestamp: float
    source_robots: list[str] = []      # which robots contributed detections
    querying_robot_id: str
