"""
Observation fusion logic.

ObservationFuser is a stateless helper that the WorldModel uses to merge
incoming robot observations into its internal grid and object registry.

Fusion strategy (v1):
- Objects: keep the detection with higher confidence; on tie, prefer newer.
- Free cells: mark FREE if the cell is UNKNOWN or was FREE with lower confidence.
  Never downgrade an OCCUPIED cell to FREE (robot must explicitly re-observe).
- Confidence decay: vectorized linear decay applied every ingest cycle.
"""

from __future__ import annotations

import numpy as np

from skynet.models import DetectedObject, FreeSpaceCell

# Grid cell values — mirrors OccupancyCellState enum for fast numpy ops
CELL_UNKNOWN: int = 0
CELL_FREE: int = 1
CELL_OCCUPIED: int = 2


class ObservationFuser:
    """Stateless helper that merges observations into grid/object state."""

    # ------------------------------------------------------------------
    # Object fusion
    # ------------------------------------------------------------------

    def fuse_object(
        self,
        existing: DetectedObject | None,
        incoming: DetectedObject,
    ) -> DetectedObject:
        """
        Merge an incoming detection with the existing entry for the same ID.

        Returns the winning detection (does not mutate either input).
        """
        if existing is None:
            return incoming.model_copy(deep=True)

        if incoming.confidence > existing.confidence:
            return incoming.model_copy(deep=True)

        if incoming.confidence == existing.confidence and incoming.timestamp > existing.timestamp:
            return incoming.model_copy(deep=True)

        return existing

    # ------------------------------------------------------------------
    # Grid fusion
    # ------------------------------------------------------------------

    def fuse_free_cells(
        self,
        grid: np.ndarray,
        confidence_grid: np.ndarray,
        free_cells: list[FreeSpaceCell],
        grid_width: int,
        grid_height: int,
    ) -> None:
        """
        Update the occupancy grid with free-space observations (in-place).

        A cell is marked FREE only if it is currently UNKNOWN, or if it is
        already FREE and the incoming confidence is higher. OCCUPIED cells
        are never downgraded here — that requires explicit re-observation.
        """
        for cell in free_cells:
            gx, gy = cell.grid_x, cell.grid_y
            if not (0 <= gx < grid_width and 0 <= gy < grid_height):
                continue

            current_state = grid[gx, gy]
            current_conf = confidence_grid[gx, gy]

            if current_state == CELL_UNKNOWN:
                grid[gx, gy] = CELL_FREE
                confidence_grid[gx, gy] = cell.confidence
            elif current_state == CELL_FREE and cell.confidence > current_conf:
                confidence_grid[gx, gy] = cell.confidence

    def fuse_occupied_cell(
        self,
        grid: np.ndarray,
        confidence_grid: np.ndarray,
        gx: int,
        gy: int,
        confidence: float,
        grid_width: int,
        grid_height: int,
    ) -> None:
        """Mark a single grid cell as occupied (in-place)."""
        if not (0 <= gx < grid_width and 0 <= gy < grid_height):
            return
        grid[gx, gy] = CELL_OCCUPIED
        if confidence > confidence_grid[gx, gy]:
            confidence_grid[gx, gy] = confidence

    # ------------------------------------------------------------------
    # Temporal decay
    # ------------------------------------------------------------------

    def apply_confidence_decay(
        self,
        confidence_grid: np.ndarray,
        elapsed_sec: float,
        decay_rate: float = 0.02,
    ) -> None:
        """
        Linearly decay all confidence values over time (vectorized, in-place).

        decay_rate: confidence units lost per second (default: 0.02/s → full
        decay in 50 seconds of no new observations).
        """
        if elapsed_sec <= 0:
            return
        decay = decay_rate * elapsed_sec
        np.subtract(confidence_grid, decay, out=confidence_grid)
        np.clip(confidence_grid, 0.0, 1.0, out=confidence_grid)
