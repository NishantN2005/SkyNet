"""
Canopy Rerun Visualizer — live 3D world model visualization.

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

from canopy.models import WorldState


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

def _log_static_scene(world_w: float, world_h: float) -> None:
    """Log the floor plane grid (static, logged once)."""

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
        ), static=True)

    # Floor outline
    corners = np.array([
        [0, 0, 0], [world_w, 0, 0],
        [world_w, world_h, 0], [0, world_h, 0], [0, 0, 0]
    ], dtype=float)
    rr.log("world/floor_outline", rr.LineStrips3D(
        [corners], colors=[[100, 100, 120]], radii=0.01
    ), static=True)


def _log_cameras(world: "WorldState", world_w: float, world_h: float) -> None:
    """
    Log camera positions for the POM lab sequence.

    The homography maps image pixels to floor coords — it does NOT tell us
    where the camera physically sits. For the POM 6p lab sequence the cameras
    are corner-mounted on the walls at ~2m height, angled slightly downward
    toward the room centre.

    Positions (metres, from dataset layout):
      camera_0 — top-left corner  (0, 0)
      camera_1 — top-right corner (world_w, 0)
      camera_2 — bottom-right     (world_w, world_h)
      camera_3 — bottom-left      (0, world_h)
    """
    cam_height = 2.0   # wall mount height in POM lab (metres)
    tilt = -0.6        # downward Z component of look direction

    wall_positions = {
        "camera_0": [0.0,     world_h,     cam_height],   # bottom-left corner, points right
        "camera_1": [0.0,     world_h / 2, cam_height],   # left wall
        "camera_2": [world_w, world_h,     cam_height],   # bottom-right corner, points left
        "camera_3": [world_w, world_h / 2, cam_height],   # right wall
    }

    room_cx = world_w / 2
    room_cy = world_h / 2

    for robot_id in world.robot_states:
        pos = wall_positions.get(robot_id, [room_cx, room_cy, cam_height])
        color = _camera_color(robot_id)

        rr.log(f"world/cameras/{robot_id}", rr.Points3D(
            positions=[pos],
            colors=[color],
            radii=0.15,
            labels=[robot_id],
        ), static=True)

        # Arrow pointing toward room centre, angled slightly downward
        dx = room_cx - pos[0]
        dy = room_cy - pos[1]
        length = (dx**2 + dy**2) ** 0.5
        if length > 0:
            dx, dy = dx / length * 1.5, dy / length * 1.5

        rr.log(f"world/cameras/{robot_id}/arrow", rr.Arrows3D(
            origins=[pos],
            vectors=[[dx, dy, tilt]],
            colors=[color],
        ), static=True)


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
    refresh_hz: float = 5.0,
    camera_ids: list[str] | None = None,
    video_paths: dict[str, str] | None = None,
) -> None:

    # Init Rerun with a fixed recording ID so pom_ingest can join the same session
    rr.init("Canopy WorldModel", recording_id="canopy-live", spawn=True)

    # Build blueprint — 3D world on top, camera feeds on bottom
    cam_panels = [
        rrb.Spatial2DView(name=cam_id, origin=f"cameras/{cam_id}")
        for cam_id in (camera_ids or [])
    ]

    top_row = rrb.Horizontal(
        rrb.Spatial3DView(name="World Model", origin="world"),
        rrb.TextLogView(name="Status Log", origin="world/status"),
        column_shares=[4, 1],
    )

    if cam_panels:
        layout = rrb.Vertical(
            top_row,
            rrb.Horizontal(*cam_panels),
            row_shares=[3, 2],
        )
    else:
        layout = top_row

    blueprint = rrb.Blueprint(
        layout,
        rrb.BlueprintPanel(state="collapsed"),
        rrb.TimePanel(state="expanded"),
    )
    rr.send_blueprint(blueprint)

    # Log static scene elements
    _log_static_scene(world_w, world_h)

    # Open video captures for camera feed display
    import cv2
    caps: dict[str, cv2.VideoCapture] = {}
    if video_paths:
        for cam_id, path in video_paths.items():
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                caps[cam_id] = cap
                print(f"Opened video for {cam_id}: {path}")

    interval = 1.0 / refresh_hz
    logged_cameras: set[str] = set()
    print(f"Canopy Rerun visualizer running at {refresh_hz}Hz")
    print("Rerun viewer should open automatically.")
    print("Ctrl+C to stop.\n")

    try:
        while True:
            t0 = time.time()

            # Fetch current frame indices from ingest and seek video caps to them
            try:
                frame_resp = httpx.get(f"{api_url}/frames", timeout=1.0)
                frame_indices = frame_resp.json() if frame_resp.status_code == 200 else {}
            except Exception:
                frame_indices = {}

            for cam_id, cap in caps.items():
                target = frame_indices.get(cam_id)
                if target is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rr.log(f"cameras/{cam_id}/image", rr.Image(frame_rgb), static=True)

            world = _fetch_world(api_url)
            if world is not None:
                # Log camera positions for any newly seen robot
                new_cameras = set(world.robot_states) - logged_cameras
                if new_cameras:
                    _log_cameras(world, world_w, world_h)
                    logged_cameras.update(world.robot_states)
                _log_world_state(world, primary_camera, world_w, world_h)
            else:
                print("WARNING: Cannot reach WorldModel server")

            elapsed = time.time() - t0
            sleep = max(0.0, interval - elapsed)
            time.sleep(sleep)

    except KeyboardInterrupt:
        print("\nVisualizer stopped.")
    finally:
        for cap in caps.values():
            cap.release()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live 3D Rerun visualization of the Canopy WorldModel."
    )
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--world-width", type=float, default=6.0)
    parser.add_argument("--world-height", type=float, default=5.0)
    parser.add_argument("--primary-camera", default="camera_0",
                        help="Camera whose detections are shown in blue (not ghosts)")
    parser.add_argument("--cameras", nargs="*", default=[],
                        help="Camera IDs to show as 2D feeds (e.g. camera_0 camera_2)")
    parser.add_argument("--videos", nargs="*", default=[],
                        help="Video paths paired with --cameras (e.g. path/c0.avi path/c2.avi)")
    parser.add_argument("--refresh-hz", type=float, default=30.0)
    args = parser.parse_args()

    video_paths = dict(zip(args.cameras, args.videos)) if args.videos else None

    run_visualizer(
        api_url=args.api_url,
        world_w=args.world_width,
        world_h=args.world_height,
        primary_camera=args.primary_camera,
        refresh_hz=args.refresh_hz,
        camera_ids=args.cameras,
        video_paths=video_paths,
    )


if __name__ == "__main__":
    main()
