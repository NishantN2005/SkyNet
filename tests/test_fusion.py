"""Tests for skynet/fusion.py — ObservationFuser."""

import numpy as np
import pytest

from skynet.fusion import CELL_FREE, CELL_OCCUPIED, CELL_UNKNOWN, ObservationFuser
from skynet.models import DetectedObject, FreeSpaceCell, Pose2D


def _make_obj(
    object_id="obj_1",
    confidence=0.8,
    timestamp=1000.0,
    observed_by="RobotA",
    x=1.0,
    y=2.0,
) -> DetectedObject:
    return DetectedObject(
        object_id=object_id,
        class_label="box",
        position=Pose2D(x=x, y=y),
        confidence=confidence,
        observed_by=observed_by,
        timestamp=timestamp,
    )


def _make_free_cell(gx=5, gy=5, confidence=0.7, timestamp=1000.0) -> FreeSpaceCell:
    return FreeSpaceCell(
        grid_x=gx, grid_y=gy,
        confidence=confidence,
        reported_by="RobotA",
        timestamp=timestamp,
    )


class TestFuseObject:
    def setup_method(self):
        self.fuser = ObservationFuser()

    def test_none_existing_returns_incoming(self):
        incoming = _make_obj(confidence=0.7)
        result = self.fuser.fuse_object(None, incoming)
        assert result.confidence == 0.7
        assert result.object_id == "obj_1"

    def test_higher_confidence_wins(self):
        existing = _make_obj(confidence=0.5, timestamp=900.0)
        incoming = _make_obj(confidence=0.9, timestamp=800.0)
        result = self.fuser.fuse_object(existing, incoming)
        assert result.confidence == 0.9

    def test_lower_confidence_incoming_loses(self):
        existing = _make_obj(confidence=0.9, timestamp=900.0)
        incoming = _make_obj(confidence=0.5, timestamp=1000.0)
        result = self.fuser.fuse_object(existing, incoming)
        assert result.confidence == 0.9

    def test_equal_confidence_newer_wins(self):
        existing = _make_obj(confidence=0.8, timestamp=900.0)
        incoming = _make_obj(confidence=0.8, timestamp=1100.0)
        result = self.fuser.fuse_object(existing, incoming)
        assert result.timestamp == 1100.0

    def test_equal_confidence_older_incoming_keeps_existing(self):
        existing = _make_obj(confidence=0.8, timestamp=1100.0)
        incoming = _make_obj(confidence=0.8, timestamp=900.0)
        result = self.fuser.fuse_object(existing, incoming)
        assert result.timestamp == 1100.0

    def test_does_not_mutate_inputs(self):
        existing = _make_obj(confidence=0.5)
        incoming = _make_obj(confidence=0.9)
        self.fuser.fuse_object(existing, incoming)
        assert existing.confidence == 0.5
        assert incoming.confidence == 0.9


class TestFuseFreeCells:
    def setup_method(self):
        self.fuser = ObservationFuser()
        self.grid = np.zeros((10, 10), dtype=np.uint8)
        self.conf = np.zeros((10, 10), dtype=np.float32)

    def test_marks_unknown_as_free(self):
        cells = [_make_free_cell(gx=3, gy=4, confidence=0.7)]
        self.fuser.fuse_free_cells(self.grid, self.conf, cells, 10, 10)
        assert self.grid[3, 4] == CELL_FREE
        assert abs(self.conf[3, 4] - 0.7) < 1e-6

    def test_higher_confidence_updates_free_cell(self):
        self.grid[3, 4] = CELL_FREE
        self.conf[3, 4] = 0.5
        cells = [_make_free_cell(gx=3, gy=4, confidence=0.9)]
        self.fuser.fuse_free_cells(self.grid, self.conf, cells, 10, 10)
        assert abs(self.conf[3, 4] - 0.9) < 1e-6

    def test_does_not_downgrade_occupied_to_free(self):
        self.grid[3, 4] = CELL_OCCUPIED
        self.conf[3, 4] = 0.9
        cells = [_make_free_cell(gx=3, gy=4, confidence=1.0)]
        self.fuser.fuse_free_cells(self.grid, self.conf, cells, 10, 10)
        assert self.grid[3, 4] == CELL_OCCUPIED

    def test_out_of_bounds_ignored(self):
        cells = [_make_free_cell(gx=99, gy=99)]
        self.fuser.fuse_free_cells(self.grid, self.conf, cells, 10, 10)
        # No exception, no grid mutation
        assert np.all(self.grid == 0)


class TestFuseOccupiedCell:
    def setup_method(self):
        self.fuser = ObservationFuser()
        self.grid = np.zeros((10, 10), dtype=np.uint8)
        self.conf = np.zeros((10, 10), dtype=np.float32)

    def test_marks_cell_occupied(self):
        self.fuser.fuse_occupied_cell(self.grid, self.conf, 2, 3, 0.8, 10, 10)
        assert self.grid[2, 3] == CELL_OCCUPIED
        assert abs(self.conf[2, 3] - 0.8) < 1e-6

    def test_out_of_bounds_ignored(self):
        self.fuser.fuse_occupied_cell(self.grid, self.conf, -1, 0, 0.8, 10, 10)
        assert np.all(self.grid == 0)


class TestConfidenceDecay:
    def setup_method(self):
        self.fuser = ObservationFuser()

    def test_decays_over_time(self):
        conf = np.full((5, 5), 1.0, dtype=np.float32)
        self.fuser.apply_confidence_decay(conf, elapsed_sec=10.0, decay_rate=0.02)
        expected = 1.0 - 0.02 * 10.0
        assert np.allclose(conf, expected)

    def test_clamps_to_zero(self):
        conf = np.full((3, 3), 0.1, dtype=np.float32)
        self.fuser.apply_confidence_decay(conf, elapsed_sec=100.0, decay_rate=0.1)
        assert np.all(conf == 0.0)

    def test_zero_elapsed_no_change(self):
        conf = np.full((3, 3), 0.5, dtype=np.float32)
        self.fuser.apply_confidence_decay(conf, elapsed_sec=0.0)
        assert np.all(conf == 0.5)
