"""
Query layer for the shared WorldModel.

QueryEngine is a thin read-only interface that filters the fused state
by spatial region, object class, observation age, and confidence.
It is intentionally separate from WorldModel so the two can evolve
independently (e.g. swap in a vector store for Phase 3).
"""

from __future__ import annotations

import time

from canopy.models import (
    DetectedObject,
    FreeSpaceCell,
    QueryRequest,
    QueryResult,
)
from canopy.world_model import WorldModel

CELL_FREE: int = 1
CELL_OCCUPIED: int = 2


class QueryEngine:
    """Read-only query interface over a WorldModel instance."""

    def __init__(self, world_model: WorldModel) -> None:
        self._wm = world_model

    def query(self, request: QueryRequest) -> QueryResult:
        """
        Query the shared world model.

        Filters applied (all optional):
          bbox            — world-coordinate bounding box (x_min, y_min, x_max, y_max)
          object_class    — exact match on DetectedObject.class_label
          max_age_seconds — exclude detections older than this
          min_confidence  — exclude detections below this confidence
        """
        now = time.time()
        cutoff_time = now - request.max_age_seconds

        all_objects = self._wm.get_objects()
        robot_states = self._wm.get_robot_states()
        grid, confidence = self._wm.get_grid_snapshot()

        # --- Filter objects ---------------------------------------------------
        matched_objects: list[DetectedObject] = []
        source_robots: set[str] = set()

        for obj in all_objects.values():
            if obj.timestamp < cutoff_time:
                continue
            if obj.confidence < request.min_confidence:
                continue
            if request.object_class and obj.class_label != request.object_class:
                continue
            if request.bbox is not None:
                x_min, y_min, x_max, y_max = request.bbox
                if not (x_min <= obj.position.x <= x_max and y_min <= obj.position.y <= y_max):
                    continue

            matched_objects.append(obj)
            source_robots.add(obj.observed_by)

        # --- Collect grid cells inside bbox (if given) -------------------------
        occupied_cells: list[tuple[int, int]] = []
        free_cells: list[FreeSpaceCell] = []

        if request.bbox is not None:
            x_min, y_min, x_max, y_max = request.bbox
            cs = self._wm.cell_size
            gx_min = max(0, int(x_min / cs))
            gy_min = max(0, int(y_min / cs))
            gx_max = min(self._wm.grid_width - 1, int(x_max / cs))
            gy_max = min(self._wm.grid_height - 1, int(y_max / cs))

            for gx in range(gx_min, gx_max + 1):
                for gy in range(gy_min, gy_max + 1):
                    state = grid[gx, gy]
                    if state == CELL_OCCUPIED:
                        occupied_cells.append((gx, gy))
                    elif state == CELL_FREE:
                        free_cells.append(
                            FreeSpaceCell(
                                grid_x=gx,
                                grid_y=gy,
                                confidence=float(confidence[gx, gy]),
                                reported_by="world_model",
                                timestamp=now,
                            )
                        )

        return QueryResult(
            objects=matched_objects,
            free_cells=free_cells,
            occupied_cells=occupied_cells,
            robot_positions={rid: rs.pose for rid, rs in robot_states.items()},
            query_timestamp=now,
            source_robots=list(source_robots),
            querying_robot_id=request.robot_id,
        )

    def get_objects_near(
        self,
        x: float,
        y: float,
        radius: float,
        max_age_seconds: float = 60.0,
        min_confidence: float = 0.1,
    ) -> list[DetectedObject]:
        """Convenience: return all objects within `radius` meters of (x, y)."""
        bbox = (x - radius, y - radius, x + radius, y + radius)
        result = self.query(
            QueryRequest(
                robot_id="__internal__",
                bbox=bbox,
                max_age_seconds=max_age_seconds,
                min_confidence=min_confidence,
            )
        )
        # Refine with true Euclidean distance
        return [
            obj for obj in result.objects
            if (obj.position.x - x) ** 2 + (obj.position.y - y) ** 2 <= radius ** 2
        ]
