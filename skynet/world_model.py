"""
WorldModel — the central shared state store.

Robots call ingest() to publish observations; anything can call
get_world_state() or get_grid_snapshot() to read the fused state.

Thread-safe via a single reentrant lock (adequate for v1 in-process use).
For a distributed deployment, replace this class with a gRPC service client
that proxies the same interface.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from skynet.fusion import CELL_OCCUPIED, ObservationFuser
from skynet.models import (
    DetectedObject,
    ObservationReport,
    RobotState,
    WorldState,
)

CELL_SIZE: float = 0.5  # meters per grid cell (world default)


class WorldModel:
    """
    Central fused world model.

    Internal state:
      _objects        — dict[object_id, DetectedObject]
      _grid           — np.ndarray[W, H] uint8 (0=UNKNOWN, 1=FREE, 2=OCCUPIED)
      _confidence     — np.ndarray[W, H] float32
      _robot_states   — dict[robot_id, RobotState]
    """

    def __init__(
        self,
        grid_width: int = 30,
        grid_height: int = 30,
        cell_size: float = CELL_SIZE,
        confidence_decay_rate: float = 0.02,
    ) -> None:
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.confidence_decay_rate = confidence_decay_rate

        self._objects: dict[str, DetectedObject] = {}
        self._grid = np.zeros((grid_width, grid_height), dtype=np.uint8)
        self._confidence = np.zeros((grid_width, grid_height), dtype=np.float32)
        self._robot_states: dict[str, RobotState] = {}
        self._last_decay_time: float = time.time()

        self._lock = threading.Lock()
        self._fuser = ObservationFuser()

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def ingest(self, report: ObservationReport) -> None:
        """
        Ingest an ObservationReport from a robot.

        Updates robot state, fuses detected objects and free cells into
        the grid, then applies temporal confidence decay.
        """
        with self._lock:
            # 1. Update robot state
            self._robot_states[report.robot_id] = RobotState(
                robot_id=report.robot_id,
                pose=report.robot_pose,
                timestamp=report.timestamp,
                is_active=True,
            )

            # 2. Fuse detected objects + mark their cells occupied
            for obj in report.detected_objects:
                # Proximity deduplication: if a nearby object from a different
                # camera already exists within 0.5m, merge into it instead of
                # creating a duplicate entry.
                merge_id = None
                for oid, existing_obj in self._objects.items():
                    if existing_obj.observed_by == obj.observed_by:
                        continue
                    dx = existing_obj.position.x - obj.position.x
                    dy = existing_obj.position.y - obj.position.y
                    if (dx * dx + dy * dy) <= 0.25:  # 0.5m radius
                        merge_id = oid
                        break

                target_id = merge_id if merge_id else obj.object_id
                existing = self._objects.get(target_id)
                self._objects[target_id] = self._fuser.fuse_object(existing, obj)

                gx, gy = obj.position.to_grid(self.cell_size)
                self._fuser.fuse_occupied_cell(
                    self._grid, self._confidence,
                    gx, gy, obj.confidence,
                    self.grid_width, self.grid_height,
                )

            # 3. Fuse free-space cells
            self._fuser.fuse_free_cells(
                self._grid, self._confidence,
                report.free_cells,
                self.grid_width, self.grid_height,
            )

            # 4. Confidence decay
            now = time.time()
            elapsed = now - self._last_decay_time
            self._fuser.apply_confidence_decay(
                self._confidence, elapsed, self.confidence_decay_rate
            )
            self._last_decay_time = now

            # 5. Prune stale objects (not seen in the last 2 seconds)
            cutoff = now - 2.0
            self._objects = {
                oid: obj for oid, obj in self._objects.items()
                if obj.timestamp >= cutoff
            }

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get_world_state(self) -> WorldState:
        """Return a JSON-serializable snapshot of the full world state."""
        with self._lock:
            return WorldState(
                timestamp=time.time(),
                objects=dict(self._objects),
                robot_states=dict(self._robot_states),
                grid_width=self.grid_width,
                grid_height=self.grid_height,
                grid_data=self._grid.tolist(),
                confidence_data=self._confidence.tolist(),
            )

    def get_grid_snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of (grid, confidence) numpy arrays for read-only use."""
        with self._lock:
            return self._grid.copy(), self._confidence.copy()

    def get_objects(self) -> dict[str, DetectedObject]:
        """Return a shallow copy of the object registry."""
        with self._lock:
            return dict(self._objects)

    def get_robot_states(self) -> dict[str, RobotState]:
        """Return a shallow copy of all known robot states."""
        with self._lock:
            return dict(self._robot_states)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all state (useful for test setup)."""
        with self._lock:
            self._objects.clear()
            self._robot_states.clear()
            self._grid[:] = 0
            self._confidence[:] = 0.0
            self._last_decay_time = time.time()
