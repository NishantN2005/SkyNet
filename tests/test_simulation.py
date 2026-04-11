"""Tests for skynet/simulation.py — FOV, occlusion, Robot, SimWorld."""

import math
import time

import pytest

from skynet.models import Pose2D
from skynet.simulation import (
    Obstacle,
    Robot,
    SimWorld,
    WorldObject,
    _bresenham_ray,
    compute_fov_mask,
)
from skynet.world_model import WorldModel


class TestBresenhamRay:
    def test_horizontal(self):
        ray = _bresenham_ray(0, 0, 4, 0)
        assert ray == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]

    def test_vertical(self):
        ray = _bresenham_ray(0, 0, 0, 3)
        assert ray == [(0, 0), (0, 1), (0, 2), (0, 3)]

    def test_diagonal(self):
        ray = _bresenham_ray(0, 0, 2, 2)
        assert (0, 0) in ray
        assert (2, 2) in ray
        assert len(ray) >= 3  # at least start, middle, end

    def test_single_cell(self):
        ray = _bresenham_ray(3, 3, 3, 3)
        assert ray == [(3, 3)]

    def test_start_and_end_included(self):
        ray = _bresenham_ray(1, 1, 5, 3)
        assert ray[0] == (1, 1)
        assert ray[-1] == (5, 3)


class TestFOVMask:
    """Tests for compute_fov_mask."""

    def test_no_obstacles_sees_cone(self):
        # Robot at (5, 5), facing east (0 rad), 90° FOV, range 5 cells
        pose = Pose2D(x=5 * 0.5, y=5 * 0.5, heading_rad=0.0)
        visible = compute_fov_mask(
            robot_pose=pose,
            fov_angle_rad=math.radians(90),
            fov_range=5 * 0.5,
            obstacle_cells=set(),
            grid_width=20,
            grid_height=20,
        )
        # Cell directly ahead should be visible
        assert (6, 5) in visible
        assert (7, 5) in visible
        # Cell directly behind should NOT be visible
        assert (3, 5) not in visible

    def test_obstacle_causes_occlusion(self):
        # Robot at (3, 5), obstacle at column 6, target at (8, 5)
        pose = Pose2D(x=3 * 0.5, y=5 * 0.5, heading_rad=0.0)
        obstacle_cells = {(6, 5)}
        visible = compute_fov_mask(
            robot_pose=pose,
            fov_angle_rad=math.radians(180),
            fov_range=10 * 0.5,
            obstacle_cells=obstacle_cells,
            grid_width=20,
            grid_height=20,
        )
        # Cell directly in front of obstacle should be visible
        assert (5, 5) in visible
        # Cell behind obstacle should be occluded
        assert (8, 5) not in visible

    def test_out_of_range_not_visible(self):
        pose = Pose2D(x=0.5, y=0.5, heading_rad=0.0)
        visible = compute_fov_mask(
            robot_pose=pose,
            fov_angle_rad=math.radians(360),
            fov_range=2 * 0.5,  # only 2 cells range
            obstacle_cells=set(),
            grid_width=30,
            grid_height=30,
        )
        # Far cell should not be visible
        assert (15, 15) not in visible


class TestObstacle:
    def test_cells(self):
        obs = Obstacle(gx_min=2, gy_min=3, gx_max=4, gy_max=5)
        cells = obs.get_cells()
        assert (2, 3) in cells
        assert (4, 5) in cells
        assert len(cells) == 3 * 3  # 3×3 block


class TestRobotSense:
    def _make_robot(self, heading=0.0):
        return Robot(
            robot_id="TestRobot",
            pose=Pose2D(x=2.5, y=2.5, heading_rad=heading),  # grid (5,5)
            fov_range=5 * 0.5,
            fov_angle_rad=math.radians(120),
        )

    def test_empty_world_no_detections(self):
        robot = self._make_robot()
        report = robot.sense([], [], 20, 20)
        assert report.detected_objects == []
        assert len(report.free_cells) > 0

    def test_detects_visible_object(self):
        robot = self._make_robot(heading=0.0)  # facing east
        obj = WorldObject(
            object_id="box_1",
            class_label="box",
            pose=Pose2D(x=4 * 0.5, y=5 * 0.5),  # grid (4, 5) — one cell east, same row
        )
        report = robot.sense([obj], [], 20, 20)
        # Robot is at grid(5,5), object is at grid(4,5) — that's WEST, not in FOV facing east
        # So let's put object at grid (7, 5):
        obj2 = WorldObject(
            object_id="box_2",
            class_label="box",
            pose=Pose2D(x=7 * 0.5, y=5 * 0.5),
        )
        report2 = robot.sense([obj2], [], 20, 20)
        assert any(o.object_id == "box_2" for o in report2.detected_objects)

    def test_does_not_detect_occluded_object(self):
        # Robot faces east, obstacle in the way, object behind it
        robot = Robot(
            robot_id="R",
            pose=Pose2D(x=3 * 0.5, y=5 * 0.5, heading_rad=0.0),
            fov_range=10 * 0.5,
            fov_angle_rad=math.radians(120),
        )
        obstacle = Obstacle(gx_min=6, gy_min=4, gx_max=6, gy_max=6)
        hidden = WorldObject(
            object_id="hidden",
            class_label="box",
            pose=Pose2D(x=9 * 0.5, y=5 * 0.5),
        )
        report = robot.sense([hidden], [obstacle], 20, 20)
        assert not any(o.object_id == "hidden" for o in report.detected_objects)

    def test_report_robot_id_matches(self):
        robot = self._make_robot()
        report = robot.sense([], [], 20, 20)
        assert report.robot_id == "TestRobot"


class TestSimWorld:
    def test_tick_ingests_to_world_model(self):
        sim = SimWorld(grid_width=30, grid_height=30)
        robot = Robot(
            robot_id="R",
            pose=Pose2D(x=5 * 0.5, y=15 * 0.5, heading_rad=0.0),
            fov_range=6 * 0.5,
            fov_angle_rad=math.radians(120),
        )
        sim.add_robot(robot)
        wm = WorldModel(grid_width=30, grid_height=30)
        reports = sim.tick(wm)
        assert len(reports) == 1
        states = wm.get_robot_states()
        assert "R" in states

    def test_full_demo_scenario(self):
        """Integration test: Robot A cannot see the hidden box; Robot B can.
        After tick(), WorldModel knows about the box; Robot A can query it."""
        sim = SimWorld(grid_width=30, grid_height=30)

        robot_a = Robot(
            robot_id="RobotA",
            pose=Pose2D(x=5 * 0.5, y=15 * 0.5, heading_rad=0.0),
            fov_range=12 * 0.5,
            fov_angle_rad=math.radians(120),
        )
        robot_b = Robot(
            robot_id="RobotB",
            pose=Pose2D(x=25 * 0.5, y=15 * 0.5, heading_rad=math.pi),
            fov_range=12 * 0.5,
            fov_angle_rad=math.radians(120),
        )
        sim.add_robot(robot_a)
        sim.add_robot(robot_b)

        container = Obstacle(gx_min=14, gy_min=12, gx_max=16, gy_max=18)
        sim.add_obstacle(container)

        hidden_box = WorldObject(
            object_id="box_001",
            class_label="box",
            pose=Pose2D(x=18 * 0.5, y=15 * 0.5),
        )
        sim.add_world_object(hidden_box)

        wm = WorldModel(grid_width=30, grid_height=30)
        reports = sim.tick(wm)

        report_a = next(r for r in reports if r.robot_id == "RobotA")
        report_b = next(r for r in reports if r.robot_id == "RobotB")

        # Robot A should NOT have detected the hidden box directly
        a_detected_ids = [o.object_id for o in report_a.detected_objects]
        assert "box_001" not in a_detected_ids, "Robot A should not see the hidden box directly"

        # Robot B SHOULD have detected it
        b_detected_ids = [o.object_id for o in report_b.detected_objects]
        assert "box_001" in b_detected_ids, "Robot B should see the hidden box"

        # After tick, the shared WorldModel should know about the box
        objects = wm.get_objects()
        assert "box_001" in objects, "WorldModel should contain the hidden box after fusion"
        assert objects["box_001"].observed_by == "RobotB"
