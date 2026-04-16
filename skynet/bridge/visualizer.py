"""
SkyNet Rerun Visualizer — live 3D world model visualization.

Opens a Rerun viewer showing:
  - Floor plane representing the tracked area
  - Camera positions as 3D frustums
  - Detected persons as 3D cylinders, colored by which camera saw them
  - Ghost objects (seen by Camera B, not Camera A) in cyan with transparency
  - Confidence decay shown via opacity
  - Live timeline as the WorldModel updates

Usage:
    python -m skynet.bridge.visualizer --api-url http://localhost:8000 \\
        --world-width 6.0 --world-height 5.0 \\
        --cameras camera_0 camera_1
"""

from __future__ import annotations

import argparse
import time

import httpx
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skynet.models import WorldState


# ---------------------------------------------------------------------------
# Colours per camera
# ---------------------------------------------------------------------------

CAMERA_COLORS = {
    "camera_0": [52, 152, 219],    # blue
    "camera_1": [231, 76, 60],     # red
    "camera_2": [46, 204, 113],    # green
    "camera_3": [155, 89, 182],    # purple
}
GHOST_COLOR = [0, 255, 204]        # cyan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_world(api_url: str) -> WorldState | None:
    try:
        resp = httpx.get(f"{api_url}/world", timeout=2.0)
        resp.raise_for_status()
        return WorldState(**resp.json())
    except Exception:
        return None


def _camera_color(camera_id: str) -> list[int]:
    return CAMERA_COLORS.get(camera_id, [200, 200, 200])


# ---------------------------------------------------------------------------
# Scene setup (logged once)
# ---------------------------------------------------------------------------

def _log_static_scene(world_w: float, world_h: float, camera_ids: list[str]) -> None:
    """Log the floor plane and camera markers (static, logged once)."""

    # Floor grid
    grid_lines = []
    for x in np.arange(0, world_w + 0.01, 0.5):
        grid_lines.append([[x, 0, 0], [x, world_h, 0]])
    for y in np.arange(0, world_h + 0.01, 0.5):
        grid_lines.append([[0, y, 0], [world_w, y, 0]])

    for i, (p0, p1) in enumerate(grid_lines):
        rr.log(f"world/grid/line_{i}", rr.LineStrips3D(
            [np.array([p0, p1])],
            colors=[[60, 60, 80]],
        ))

    # Floor outline
    corners = np.array([
        [0, 0, 0], [world_w, 0, 0],
        [world_w, world_h, 0], [0, world_h, 0], [0, 0, 0]
    ], dtype=float)
    rr.log("world/floor_outline", rr.LineStrips3D(
        [corners], colors=[[100, 100, 120]], radii=0.01
    ))

    # Camera markers — simple box at approximate positions
    cam_positions = {
        "camera_0": [0.0, world_h / 2, 0.5],
        "camera_1": [world_w, world_h / 2, 0.5],
        "camera_2": [world_w / 2, 0.0, 0.5],
        "camera_3": [world_w / 2, world_h, 0.5],
    }

    for cam_id in camera_ids:
        pos = cam_positions.get(cam_id, [world_w / 2, world_h / 2, 0.5])
        color = _camera_color(cam_id)
        rr.log(f"world/cameras/{cam_id}", rr.Points3D(
            positions=[pos],
            colors=[color],
            radii=0.15,
            labels=[cam_id],
        ))
        # Log downward-pointing arrow to show camera direction
        rr.log(f"world/cameras/{cam_id}/arrow", rr.Arrows3D(
            origins=[pos],
            vectors=[[0, 0, -0.4]],
            colors=[color],
        ))


# ---------------------------------------------------------------------------
# Per-frame update
# ---------------------------------------------------------------------------

_prev_obj_ids: set[str] = set()


def _log_world_state(
    world: WorldState,
    primary_camera: str,
    world_w: float,
    world_h: float,
) -> None:
    """Log current WorldModel state to Rerun (static — no timeline stacking)."""
    global _prev_obj_ids

    current_ids = set(world.objects.keys())

    # Hide disappeared objects by moving them far off-screen
    for obj_id in _prev_obj_ids - current_ids:
        rr.log(f"world/objects/{obj_id}", rr.Boxes3D(
            centers=[[9999, 9999, 9999]],
            half_sizes=[[0, 0, 0]],
        ), static=True)
        rr.log(f"world/objects/{obj_id}/foot", rr.Points3D(
            positions=[[9999, 9999, 9999]],
            radii=0.0,
        ), static=True)

    _prev_obj_ids = current_ids

    person_height = 1.75  # metres

    for obj_id, obj in world.objects.items():
        x = obj.position.x
        y = obj.position.y
        conf = obj.confidence

        is_ghost = obj.observed_by != primary_camera

        if is_ghost:
            color = GHOST_COLOR + [int(180 * conf)]    # cyan, semi-transparent
            label = f"GHOST: {obj.class_label}\n({x:.1f},{y:.1f})m\nvia {obj.observed_by}"
            radius = 0.2
            z_center = person_height / 2
        else:
            base_color = _camera_color(obj.observed_by)
            color = base_color + [int(220 * conf)]
            label = f"{obj.class_label}\n({x:.1f},{y:.1f})m\nconf={conf:.2f}"
            radius = 0.25
            z_center = person_height / 2

        # Person cylinder approximated as a tall thin box
        rr.log(f"world/objects/{obj_id}", rr.Boxes3D(
            centers=[[x, y, z_center]],
            half_sizes=[[radius, radius, person_height / 2]],
            colors=[color],
            labels=[label],
        ), static=True)

        # Foot dot on the floor
        rr.log(f"world/objects/{obj_id}/foot", rr.Points3D(
            positions=[[x, y, 0.02]],
            colors=[color],
            radii=0.12,
        ), static=True)

    # Robot/camera states as small spheres showing last known pose
    for robot_id, robot in world.robot_states.items():
        rx = robot.pose.x
        ry = robot.pose.y
        color = _camera_color(robot_id)
        rr.log(f"world/robots/{robot_id}", rr.Points3D(
            positions=[[rx, ry, 0.1]],
            colors=[color],
            radii=0.08,
        ), static=True)

    # Summary text
    n_total = len(world.objects)
    n_ghost = sum(1 for o in world.objects.values() if o.observed_by != primary_camera)
    rr.log("world/status", rr.TextLog(
        f"Objects: {n_total} total | {n_total - n_ghost} seen by {primary_camera} | {n_ghost} ghosts",
        level=rr.TextLogLevel.INFO,
    ))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_visualizer(
    api_url: str = "http://localhost:8000",
    world_w: float = 6.0,
    world_h: float = 5.0,
    primary_camera: str = "camera_0",
    camera_ids: list[str] | None = None,
    refresh_hz: float = 5.0,
) -> None:
    if camera_ids is None:
        camera_ids = ["camera_0", "camera_1"]

    # Init Rerun
    rr.init("SkyNet WorldModel", spawn=True)

    # Set up blueprint — 3D view + text log panel, timeline follows latest frame
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="World Model", origin="world"),
            rrb.TextLogView(name="Status Log", origin="world/status"),
            column_shares=[4, 1],
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed"),
    )
    rr.send_blueprint(blueprint)

    # Log static scene elements
    _log_static_scene(world_w, world_h, camera_ids)

    interval = 1.0 / refresh_hz
    print(f"SkyNet Rerun visualizer running at {refresh_hz}Hz")
    print("Rerun viewer should open automatically.")
    print("Ctrl+C to stop.\n")

    try:
        while True:
            t0 = time.time()
            world = _fetch_world(api_url)
            if world is not None:
                _log_world_state(world, primary_camera, world_w, world_h)
            else:
                print("WARNING: Cannot reach WorldModel server")

            elapsed = time.time() - t0
            sleep = max(0.0, interval - elapsed)
            time.sleep(sleep)

    except KeyboardInterrupt:
        print("\nVisualizer stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live 3D Rerun visualization of the SkyNet WorldModel."
    )
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--world-width", type=float, default=6.0)
    parser.add_argument("--world-height", type=float, default=5.0)
    parser.add_argument("--primary-camera", default="camera_0",
                        help="Camera whose detections are shown in blue (not ghosts)")
    parser.add_argument("--cameras", nargs="+",
                        default=["camera_0", "camera_1"])
    parser.add_argument("--refresh-hz", type=float, default=5.0)
    args = parser.parse_args()

    run_visualizer(
        api_url=args.api_url,
        world_w=args.world_width,
        world_h=args.world_height,
        primary_camera=args.primary_camera,
        camera_ids=args.cameras,
        refresh_hz=args.refresh_hz,
    )


if __name__ == "__main__":
    main()
