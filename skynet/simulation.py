"""
2D simulation engine.

Components:
  - Obstacle       — rectangular solid block in the grid
  - WorldObject    — detectable object (box, person, etc.)
  - Robot          — senses the environment, produces ObservationReports
  - SimWorld       — orchestrates the scene and drives tick()

FOV + Occlusion algorithm:
  For each candidate grid cell within the FOV cone:
    1. Check it falls within range and angle bounds.
    2. Cast a Bresenham ray from robot cell to target cell.
    3. If any intermediate cell is an obstacle → target is OCCLUDED.
    4. Otherwise → cell is VISIBLE (FREE or OCCUPIED depending on objects).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from skynet.models import (
    DetectedObject,
    FreeSpaceCell,
    ObservationReport,
    Pose2D,
)

CELL_SIZE: float = 0.5  # meters per grid cell


# ---------------------------------------------------------------------------
# Ray casting
# ---------------------------------------------------------------------------


def _bresenham_ray(
    x0: int, y0: int, x1: int, y1: int
) -> list[tuple[int, int]]:
    """
    Return all grid cells along the line from (x0, y0) to (x1, y1).
    Uses Bresenham's line algorithm. Start and end cells are included.
    """
    cells: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0

    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return cells


def compute_fov_mask(
    robot_pose: Pose2D,
    fov_angle_rad: float,
    fov_range: float,
    obstacle_cells: set[tuple[int, int]],
    grid_width: int,
    grid_height: int,
    cell_size: float = CELL_SIZE,
) -> set[tuple[int, int]]:
    """
    Return all grid cells visible from robot_pose given FOV cone and obstacles.

    A cell is visible if:
      - It lies within fov_range meters of the robot.
      - The angle from the robot's heading to the cell is within ±fov_angle_rad/2.
      - No obstacle cell lies strictly between the robot and the target cell
        along the Bresenham ray.
    """
    robot_gx_f = robot_pose.x / cell_size
    robot_gy_f = robot_pose.y / cell_size
    robot_gx = int(robot_gx_f)
    robot_gy = int(robot_gy_f)

    half_fov = fov_angle_rad / 2.0
    range_cells = fov_range / cell_size

    visible: set[tuple[int, int]] = set()

    for gx in range(grid_width):
        for gy in range(grid_height):
            dx = gx - robot_gx_f
            dy = gy - robot_gy_f
            dist = math.sqrt(dx * dx + dy * dy)

            if dist > range_cells or dist == 0.0:
                continue

            # Angle check
            angle_to_cell = math.atan2(dy, dx)
            angle_diff = angle_to_cell - robot_pose.heading_rad
            # Normalise to (−π, π]
            angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

            if abs(angle_diff) > half_fov:
                continue

            # Occlusion check via Bresenham ray
            ray = _bresenham_ray(robot_gx, robot_gy, gx, gy)
            occluded = False
            # Skip first cell (robot position) and last cell (target itself)
            for rx, ry in ray[1:-1]:
                if (rx, ry) in obstacle_cells:
                    occluded = True
                    break

            if not occluded:
                visible.add((gx, gy))

    return visible


# ---------------------------------------------------------------------------
# Scene entities
# ---------------------------------------------------------------------------


@dataclass
class Obstacle:
    """Rectangular solid obstacle (e.g. shipping container) in grid coordinates."""

    gx_min: int
    gy_min: int
    gx_max: int
    gy_max: int
    label: str = "obstacle"

    def get_cells(self) -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for gx in range(self.gx_min, self.gx_max + 1):
            for gy in range(self.gy_min, self.gy_max + 1):
                cells.add((gx, gy))
        return cells


@dataclass
class WorldObject:
    """A detectable object placed in the simulation world."""

    object_id: str
    class_label: str
    pose: Pose2D
    width: float = 0.5                              # meters
    height: float = 0.5                             # meters
    path: list = field(default_factory=list)        # list[Pose2D] waypoints
    loop: bool = True                               # loop path vs stop at end

    def __post_init__(self) -> None:
        self._path_idx: int = 0

    def step(self) -> None:
        """Advance one waypoint along the path (no-op if no path set)."""
        if not self.path:
            return
        if self.loop:
            self._path_idx = (self._path_idx + 1) % len(self.path)
        else:
            self._path_idx = min(self._path_idx + 1, len(self.path) - 1)
        self.pose = self.path[self._path_idx]

    def to_grid(self, cell_size: float = CELL_SIZE) -> tuple[int, int]:
        return (int(self.pose.x / cell_size), int(self.pose.y / cell_size))


# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------


class Robot:
    """
    A simulated robot that senses its local environment.

    Call sense() each tick to get an ObservationReport ready for ingest
    into the WorldModel. After sense() returns, _direct_visible_cells holds
    the raw visibility mask (useful for per-robot visualisation).
    """

    def __init__(
        self,
        robot_id: str,
        pose: Pose2D,
        fov_range: float = 6.0,         # meters
        fov_angle_rad: float = 2.094,   # ~120°
        detection_confidence: float = 0.9,
        cell_size: float = CELL_SIZE,
        path: list | None = None,       # list[Pose2D] waypoints
        loop: bool = True,              # loop path vs stop at end
    ) -> None:
        self.robot_id = robot_id
        self.pose = pose
        self.fov_range = fov_range
        self.fov_angle_rad = fov_angle_rad
        self.detection_confidence = detection_confidence
        self.cell_size = cell_size
        self.path: list[Pose2D] = path or []
        self.loop = loop
        self._path_idx: int = 0

        # Populated after each sense() call — used by visualisation
        self._direct_visible_cells: set[tuple[int, int]] = set()

    def step(self) -> None:
        """Advance one waypoint along the path (no-op if no path set)."""
        if not self.path:
            return
        if self.loop:
            self._path_idx = (self._path_idx + 1) % len(self.path)
        else:
            self._path_idx = min(self._path_idx + 1, len(self.path) - 1)
        self.pose = self.path[self._path_idx]

    def sense(
        self,
        world_objects: list[WorldObject],
        obstacles: list[Obstacle],
        grid_width: int,
        grid_height: int,
    ) -> ObservationReport:
        """
        Compute FOV + occlusion, detect objects, and return an ObservationReport.

        For every visible cell:
          - If a WorldObject occupies it → emit a DetectedObject.
          - Otherwise (and not an obstacle) → emit a FreeSpaceCell.
        """
        obstacle_cells: set[tuple[int, int]] = set()
        for obs in obstacles:
            obstacle_cells.update(obs.get_cells())

        visible_cells = compute_fov_mask(
            self.pose,
            self.fov_angle_rad,
            self.fov_range,
            obstacle_cells,
            grid_width,
            grid_height,
            self.cell_size,
        )
        self._direct_visible_cells = visible_cells

        now = time.time()
        detected: list[DetectedObject] = []
        free: list[FreeSpaceCell] = []

        for gx, gy in visible_cells:
            cell_occupied = False

            for obj in world_objects:
                ogx, ogy = obj.to_grid(self.cell_size)
                if ogx == gx and ogy == gy:
                    detected.append(
                        DetectedObject(
                            object_id=obj.object_id,
                            class_label=obj.class_label,
                            position=obj.pose,
                            confidence=self.detection_confidence,
                            observed_by=self.robot_id,
                            timestamp=now,
                            width=obj.width,
                            height=obj.height,
                        )
                    )
                    cell_occupied = True

            if not cell_occupied and (gx, gy) not in obstacle_cells:
                free.append(
                    FreeSpaceCell(
                        grid_x=gx,
                        grid_y=gy,
                        confidence=0.85,
                        reported_by=self.robot_id,
                        timestamp=now,
                    )
                )

        return ObservationReport(
            robot_id=self.robot_id,
            timestamp=now,
            robot_pose=self.pose,
            detected_objects=detected,
            free_cells=free,
            fov_range=self.fov_range,
            fov_angle_rad=self.fov_angle_rad,
        )


# ---------------------------------------------------------------------------
# SimWorld
# ---------------------------------------------------------------------------


class SimWorld:
    """
    Orchestrates the full simulation scene.

    Usage:
        world = SimWorld()
        world.add_robot(robot_a)
        world.add_obstacle(container)
        world.add_world_object(hidden_box)
        reports = world.tick(world_model)
    """

    def __init__(
        self,
        grid_width: int = 30,
        grid_height: int = 30,
        cell_size: float = CELL_SIZE,
    ) -> None:
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.robots: list[Robot] = []
        self.obstacles: list[Obstacle] = []
        self.world_objects: list[WorldObject] = []

    def add_robot(self, robot: Robot) -> None:
        self.robots.append(robot)

    def add_obstacle(self, obstacle: Obstacle) -> None:
        self.obstacles.append(obstacle)

    def add_world_object(self, obj: WorldObject) -> None:
        self.world_objects.append(obj)

    def tick(self, world_model) -> list[ObservationReport]:
        """
        Run one simulation tick: step all entities, then sense and ingest.

        Returns the list of ObservationReports (one per robot) for inspection.
        """
        # Step all moving entities before sensing so robots observe the
        # updated positions, not where things were last tick.
        for robot in self.robots:
            robot.step()
        for obj in self.world_objects:
            obj.step()

        reports: list[ObservationReport] = []
        for robot in self.robots:
            report = robot.sense(
                self.world_objects,
                self.obstacles,
                self.grid_width,
                self.grid_height,
            )
            world_model.ingest(report)
            reports.append(report)
        return reports

    def get_obstacle_cells(self) -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for obs in self.obstacles:
            cells.update(obs.get_cells())
        return cells
