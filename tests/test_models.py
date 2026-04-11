"""Tests for skynet/models.py."""

import pytest
from pydantic import ValidationError

from skynet.models import (
    DetectedObject,
    FreeSpaceCell,
    ObservationReport,
    OccupancyCellState,
    Pose2D,
    QueryRequest,
    QueryResult,
    RobotState,
)


class TestPose2D:
    def test_to_grid_basic(self):
        pose = Pose2D(x=3.0, y=7.5)
        gx, gy = pose.to_grid(cell_size=0.5)
        assert gx == 6
        assert gy == 15

    def test_to_grid_zero(self):
        pose = Pose2D(x=0.0, y=0.0)
        assert pose.to_grid() == (0, 0)

    def test_distance_to(self):
        a = Pose2D(x=0.0, y=0.0)
        b = Pose2D(x=3.0, y=4.0)
        assert abs(a.distance_to(b) - 5.0) < 1e-9

    def test_heading_default(self):
        pose = Pose2D(x=1.0, y=2.0)
        assert pose.heading_rad == 0.0


class TestDetectedObject:
    def _make(self, confidence=0.8):
        return DetectedObject(
            object_id="obj_1",
            class_label="box",
            position=Pose2D(x=1.0, y=2.0),
            confidence=confidence,
            observed_by="RobotA",
            timestamp=1000.0,
        )

    def test_valid(self):
        obj = self._make(0.9)
        assert obj.confidence == 0.9

    def test_confidence_zero(self):
        obj = self._make(0.0)
        assert obj.confidence == 0.0

    def test_confidence_one(self):
        obj = self._make(1.0)
        assert obj.confidence == 1.0

    def test_confidence_too_low(self):
        with pytest.raises(ValidationError):
            self._make(-0.01)

    def test_confidence_too_high(self):
        with pytest.raises(ValidationError):
            self._make(1.01)


class TestFreeSpaceCell:
    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            FreeSpaceCell(grid_x=0, grid_y=0, confidence=1.5, reported_by="A", timestamp=0.0)

    def test_valid(self):
        cell = FreeSpaceCell(grid_x=5, grid_y=3, confidence=0.7, reported_by="B", timestamp=100.0)
        assert cell.grid_x == 5


class TestObservationReport:
    def test_defaults(self):
        report = ObservationReport(
            robot_id="R1",
            timestamp=500.0,
            robot_pose=Pose2D(x=0.0, y=0.0),
        )
        assert report.detected_objects == []
        assert report.free_cells == []
        assert report.fov_angle_rad > 0


class TestOccupancyCellState:
    def test_values(self):
        assert OccupancyCellState.FREE == "FREE"
        assert OccupancyCellState.OCCUPIED == "OCCUPIED"
        assert OccupancyCellState.UNKNOWN == "UNKNOWN"


class TestQueryRequest:
    def test_defaults(self):
        req = QueryRequest(robot_id="R1")
        assert req.bbox is None
        assert req.object_class is None
        assert req.max_age_seconds == 60.0
        assert req.min_confidence == 0.1
