"""
SkyNet POC Demo — with movement
================================
Dynamic scenario over 40 ticks:

  - Robot A patrols vertically on the left (x=5, y=8↔22), always facing east.
  - Robot B patrols vertically on the right (x=25, y=22↔8), always facing west.
  - A person walks horizontally behind the shipping container (x=17↔20, y=15).

The person is always occluded from Robot A by the container.
Robot B sees the person whenever their patrol paths bring B within range.
The fused WorldModel tracks the person's position across all ticks.
Robot A queries the model and finds the person — observed_by=RobotB.
Confidence decay means detections age out when Robot B loses sight, keeping
the model live rather than accumulating stale state.

Run with:
    python -m skynet.main
"""

from __future__ import annotations

import math

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from skynet.models import Pose2D, QueryRequest
from skynet.queries import QueryEngine
from skynet.simulation import Obstacle, Robot, SimWorld, WorldObject
from skynet.world_model import WorldModel

# ---------------------------------------------------------------------------
# Scene constants
# ---------------------------------------------------------------------------

GRID_W = 30
GRID_H = 30
CELL_SIZE = 0.5   # meters per cell
N_TICKS = 40
PAUSE_SEC = 0.12  # seconds between animation frames

# ---------------------------------------------------------------------------
# Colour map
# ---------------------------------------------------------------------------

_UNKNOWN = 0
_FREE = 1
_OCCUPIED = 2
_OBSTACLE = 3

_CMAP = ListedColormap(["#2d2d2d", "#90ee90", "#e74c3c", "#1a1a1a"])
_LEGEND = [
    mpatches.Patch(color="#2d2d2d", label="Unknown"),
    mpatches.Patch(color="#90ee90", label="Free space"),
    mpatches.Patch(color="#e74c3c", label="Detected object"),
    mpatches.Patch(color="#1a1a1a", label="Obstacle"),
]


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------


def _patrol_path(x_cell: int, y_start: int, y_end: int, heading_rad: float) -> list[Pose2D]:
    """Build a back-and-forth vertical patrol path at fixed x."""
    step = 1 if y_end > y_start else -1
    forward = list(range(y_start, y_end + step, step))
    backward = list(range(y_end - step, y_start - step, -step))
    return [
        Pose2D(x=x_cell * CELL_SIZE, y=y * CELL_SIZE, heading_rad=heading_rad)
        for y in (forward + backward)
    ]


def build_demo_scene() -> tuple[SimWorld, WorldModel]:
    """
    30×30 grid layout (grid cells, not to scale):

      A patrols x=5, y=8..22   [CONTAINER x=14-16, y=12-18]   B patrols x=25, y=22..8
                                          person walks x=17..20, y=15

    Container blocks A's LOS to anything at x≥17.
    """
    sim = SimWorld(grid_width=GRID_W, grid_height=GRID_H, cell_size=CELL_SIZE)

    # Robot A — vertical patrol on the left, always facing east
    path_a = _patrol_path(x_cell=5, y_start=8, y_end=22, heading_rad=0.0)
    robot_a = Robot(
        robot_id="RobotA",
        pose=path_a[0],
        fov_range=12 * CELL_SIZE,
        fov_angle_rad=math.radians(120),
        detection_confidence=0.9,
        path=path_a,
        loop=True,
    )

    # Robot B — vertical patrol on the right, always facing west
    # Start at the opposite end so the two robots are out of phase
    path_b = _patrol_path(x_cell=25, y_start=22, y_end=8, heading_rad=math.pi)
    robot_b = Robot(
        robot_id="RobotB",
        pose=path_b[0],
        fov_range=12 * CELL_SIZE,
        fov_angle_rad=math.radians(120),
        detection_confidence=0.9,
        path=path_b,
        loop=True,
    )

    sim.add_robot(robot_a)
    sim.add_robot(robot_b)

    # Obstacle: shipping container
    container = Obstacle(gx_min=14, gy_min=12, gx_max=16, gy_max=18, label="container")
    sim.add_obstacle(container)

    # Stationary box (always behind container, always hidden from A)
    static_box = WorldObject(
        object_id="box_001",
        class_label="box",
        pose=Pose2D(x=18 * CELL_SIZE, y=15 * CELL_SIZE),
    )
    sim.add_world_object(static_box)

    # Moving person: walks horizontally behind the container
    person_xs = list(range(17, 21)) + list(range(19, 16, -1))  # 17→20→17
    person_path = [
        Pose2D(x=x * CELL_SIZE, y=15 * CELL_SIZE) for x in person_xs
    ]
    person = WorldObject(
        object_id="person_001",
        class_label="person",
        pose=person_path[0],
        path=person_path,
        loop=True,
    )
    sim.add_world_object(person)

    wm = WorldModel(grid_width=GRID_W, grid_height=GRID_H, cell_size=CELL_SIZE)
    return sim, wm


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def _build_single_robot_grid(
    robot: Robot,
    report,          # ObservationReport already computed this tick
    obstacle_cells: set[tuple[int, int]],
) -> np.ndarray:
    """
    Build a display grid for one robot using the tick's pre-computed report.

    Uses robot._direct_visible_cells (set by sense() during tick) and
    report.detected_objects — no re-sensing needed or wanted.
    """
    display = np.zeros((GRID_W, GRID_H), dtype=np.uint8)

    # Obstacles
    for gx, gy in obstacle_cells:
        if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
            display[gx, gy] = _OBSTACLE

    # Detected object cells from this tick's report
    detected_cells: set[tuple[int, int]] = set()
    for obj in report.detected_objects:
        gx = int(obj.position.x / CELL_SIZE)
        gy = int(obj.position.y / CELL_SIZE)
        detected_cells.add((gx, gy))

    # Overlay visible FOV
    for gx, gy in robot._direct_visible_cells:
        if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
            if display[gx, gy] == _OBSTACLE:
                continue
            if (gx, gy) in detected_cells:
                display[gx, gy] = _OCCUPIED
            else:
                display[gx, gy] = _FREE

    return display


def _build_fused_grid(
    wm: WorldModel,
    obstacle_cells: set[tuple[int, int]],
) -> np.ndarray:
    """Build display grid from the shared (fused) WorldModel state."""
    raw_grid, _ = wm.get_grid_snapshot()
    display = np.zeros((GRID_W, GRID_H), dtype=np.uint8)

    for gx in range(GRID_W):
        for gy in range(GRID_H):
            if (gx, gy) in obstacle_cells:
                display[gx, gy] = _OBSTACLE
            elif raw_grid[gx, gy] == 2:
                display[gx, gy] = _OCCUPIED
            elif raw_grid[gx, gy] == 1:
                display[gx, gy] = _FREE

    return display


def _draw_robot(ax, x_cell: float, y_cell: float, label: str, color: str) -> None:
    ax.plot(
        x_cell, y_cell, "o",
        color=color, markersize=11,
        markeredgecolor="white", markeredgewidth=1.5, zorder=5,
    )
    ax.annotate(
        label, (x_cell, y_cell),
        textcoords="offset points", xytext=(6, 6),
        fontweight="bold", color=color, fontsize=9, zorder=6,
    )


def _render_frame(
    axes,
    sim: SimWorld,
    wm: WorldModel,
    report_map: dict,
    tick: int,
    obstacle_cells: set[tuple[int, int]],
) -> None:
    """Clear and redraw all three panels for the current tick."""
    robot_a, robot_b = sim.robots[0], sim.robots[1]

    grid_a = _build_single_robot_grid(robot_a, report_map["RobotA"], obstacle_cells)
    grid_b = _build_single_robot_grid(robot_b, report_map["RobotB"], obstacle_cells)
    grid_fused = _build_fused_grid(wm, obstacle_cells)

    panel_titles = [
        f"Robot A  ({robot_a.pose.x / CELL_SIZE:.0f}, {robot_a.pose.y / CELL_SIZE:.0f})\n"
        "Direct view — person never visible",
        f"Robot B  ({robot_b.pose.x / CELL_SIZE:.0f}, {robot_b.pose.y / CELL_SIZE:.0f})\n"
        "Direct view — tracks person when in range",
        "Fused World Model\nRobot A gains awareness via Robot B",
    ]
    grids = [grid_a, grid_b, grid_fused]

    for ax, title, g in zip(axes, panel_titles, grids):
        ax.cla()
        ax.imshow(
            g.T, cmap=_CMAP, vmin=0, vmax=3,
            origin="lower", interpolation="nearest",
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")
        ax.grid(True, alpha=0.2, linewidth=0.5)

    robot_info = [
        (robot_a.pose.x / CELL_SIZE, robot_a.pose.y / CELL_SIZE, "A", "#3498db"),
        (robot_b.pose.x / CELL_SIZE, robot_b.pose.y / CELL_SIZE, "B", "#9b59b6"),
    ]
    for ax in axes:
        for rx, ry, lbl, col in robot_info:
            _draw_robot(ax, rx, ry, lbl, col)

    axes[2].legend(handles=_LEGEND, loc="upper right", fontsize=8, framealpha=0.8)


# ---------------------------------------------------------------------------
# Demo entrypoint
# ---------------------------------------------------------------------------


def run_demo() -> None:
    print("=" * 62)
    print("  SkyNet — Collaborative Spatial Awareness (with movement)")
    print("=" * 62)
    print(f"  {N_TICKS} ticks  |  Robot A & B patrol vertically")
    print("  Person walks behind container — always hidden from A")
    print()

    sim, wm = build_demo_scene()
    qe = QueryEngine(wm)
    obstacle_cells = sim.get_obstacle_cells()

    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("SkyNet: Collaborative Spatial Awareness POC", fontsize=13, fontweight="bold")

    for tick in range(N_TICKS):
        reports = sim.tick(wm)
        report_map = {r.robot_id: r for r in reports}

        # Terminal output
        person_b = [o for o in report_map["RobotB"].detected_objects if o.object_id == "person_001"]
        person_a = [o for o in report_map["RobotA"].detected_objects if o.object_id == "person_001"]
        b_sees = f"pos=({person_b[0].position.x / CELL_SIZE:.0f},{person_b[0].position.y / CELL_SIZE:.0f})" if person_b else "—"
        a_sees = "DIRECT" if person_a else "—"
        result = qe.query(QueryRequest(robot_id="RobotA", object_class="person", max_age_seconds=5.0, min_confidence=0.3))
        a_via_model = f"via model ✓  src={result.source_robots}" if result.objects else "not in model"
        print(f"  tick {tick + 1:02d} | B sees person: {b_sees:<20} | A direct: {a_sees:<8} | A query: {a_via_model}")

        # Animation frame
        _render_frame(axes, sim, wm, report_map, tick, obstacle_cells)
        fig.suptitle(
            f"SkyNet — Tick {tick + 1}/{N_TICKS}  |  "
            f"A@({sim.robots[0].pose.x / CELL_SIZE:.0f},{sim.robots[0].pose.y / CELL_SIZE:.0f})  "
            f"B@({sim.robots[1].pose.x / CELL_SIZE:.0f},{sim.robots[1].pose.y / CELL_SIZE:.0f})",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout()
        plt.pause(PAUSE_SEC)

    plt.ioff()
    save_path = "skynet_demo.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"\nFinal frame saved → {save_path}")
    plt.show()


if __name__ == "__main__":
    run_demo()
