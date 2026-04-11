"""
Ghost Stream — live Camera A feed with real-time ghost outlines of people
that Camera B can see but Camera A cannot.

Camera A's view is displayed in a window. Any person detected by Camera B
but not by Camera A is projected back into Camera A's pixel space and drawn
as a glowing ghost silhouette — updated every frame.

Usage:
    # Terminal 1 — server
    uvicorn skynet.api:app --host 0.0.0.0 --port 8000

    # Terminal 2 — Camera B bridge (watches the other zone)
    python -m skynet.bridge.camera_bridge --config calibration_b_live.json

    # Terminal 3 — Ghost stream (Camera A's view + ghosts)
    python -m skynet.bridge.ghost_stream --config-a calibration_a_live.json

Press 'q' to quit.
"""

from __future__ import annotations

import argparse
import time
from copy import deepcopy

import cv2
import httpx
import numpy as np

from skynet.bridge.camera_bridge import _extract_detections
from skynet.bridge.config import load_config
from skynet.bridge.reveal import _draw_ghost_object, _world_to_pixel
from skynet.models import WorldState, Pose2D, ObservationReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_world(api_url: str) -> WorldState | None:
    try:
        resp = httpx.get(f"{api_url}/world", timeout=1.0)
        resp.raise_for_status()
        return WorldState(**resp.json())
    except Exception:
        return None


def _post_ingest(api_url: str, report: ObservationReport) -> None:
    try:
        httpx.post(
            f"{api_url}/ingest",
            content=report.model_dump_json(),
            headers={"Content-Type": "application/json"},
            timeout=1.0,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_ghost_stream(
    config_a: str,
    api_url: str = "http://localhost:8000",
    track_classes: list[str] | None = None,
    ingest: bool = True,
) -> None:
    from ultralytics import YOLO

    if track_classes is None:
        track_classes = ["person"]  # default: only ghost people

    cfg_a = load_config(config_a)
    H_a = np.array(cfg_a.homography, dtype=np.float64)
    world_w = cfg_a.world_width_m
    world_h = cfg_a.world_height_m
    camera_pose = Pose2D(**cfg_a.camera_pose)

    print(f"Loading {cfg_a.yolo_model}...")
    model = YOLO(cfg_a.yolo_model)

    cap = cv2.VideoCapture(cfg_a.camera_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera source: {cfg_a.camera_source!r}")

    print(f"Ghost stream running — camera_id={cfg_a.camera_id!r}")
    print(f"Tracking classes: {track_classes}")
    print("Press 'q' to quit.\n")

    win_name = f"SkyNet Ghost Stream — {cfg_a.camera_id}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    min_ingest_interval = 1.0 / cfg_a.ingest_fps
    last_ingest = 0.0
    fps_timer = time.time()
    fps_counter = 0
    display_fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Failed to read frame, retrying...")
            time.sleep(0.05)
            continue

        now = time.time()

        # --- Run YOLO on Camera A's frame ---
        results_a = model(frame, conf=cfg_a.conf_threshold, verbose=False)
        detected_a = _extract_detections(results_a, H_a, cfg_a, cfg_a.camera_id)

        # --- Ingest Camera A's observations ---
        if ingest and now - last_ingest >= min_ingest_interval:
            report = ObservationReport(
                robot_id=cfg_a.camera_id,
                timestamp=now,
                robot_pose=camera_pose,
                detected_objects=detected_a,
                free_cells=[],
                fov_range=999.0,
                fov_angle_rad=3.14159,
            )
            _post_ingest(api_url, report)
            last_ingest = now

        # --- Fetch WorldModel state ---
        world = _fetch_world(api_url)

        # --- Annotate Camera A's own detections (plain boxes) ---
        output = frame.copy()
        boxes_a = results_a[0].boxes
        names_a = results_a[0].names
        if boxes_a is not None:
            for box in boxes_a:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                cls_id = int(box.cls[0])
                lbl = names_a.get(cls_id, str(cls_id))
                conf = float(box.conf[0])
                cv2.rectangle(output, (x1, y1), (x2, y2), (52, 152, 219), 3)
                cv2.putText(output, f"{lbl} {conf:.2f}",
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (52, 152, 219), 2,
                            lineType=cv2.LINE_AA)

        # --- Draw ghost outlines for hidden objects ---
        if world is not None:
            ids_seen_by_a = {d.object_id for d in detected_a}

            for obj in world.objects.values():
                # Only ghost objects from Camera B, not seen by Camera A
                if obj.observed_by == cfg_a.camera_id:
                    continue
                if obj.object_id in ids_seen_by_a:
                    continue
                if obj.class_label not in track_classes:
                    continue

                # Mirror world coords (Camera B is at opposite end)
                mirrored = deepcopy(obj)
                mirrored.position.x = world_w - obj.position.x
                mirrored.position.y = world_h - obj.position.y

                # Bounds check
                if not (0.0 <= mirrored.position.x <= world_w and
                        0.0 <= mirrored.position.y <= world_h):
                    continue

                output = _draw_ghost_object(output, H_a, mirrored, obj.class_label)

        # --- FPS overlay ---
        fps_counter += 1
        if now - fps_timer >= 1.0:
            display_fps = fps_counter / (now - fps_timer)
            fps_counter = 0
            fps_timer = now

        ghost_count = 0
        if world is not None:
            ids_seen_by_a = {d.object_id for d in detected_a}
            ghost_count = sum(
                1 for obj in world.objects.values()
                if obj.observed_by != cfg_a.camera_id
                and obj.object_id not in ids_seen_by_a
                and obj.class_label in track_classes
            )

        status = (f"SkyNet Ghost Stream  |  {cfg_a.camera_id}  |  "
                  f"{len(detected_a)} detected  |  {ghost_count} ghosted  |  "
                  f"{display_fps:.1f} fps  |  q=quit")

        # Status bar background
        cv2.rectangle(output, (0, output.shape[0] - 35),
                      (output.shape[1], output.shape[0]), (0, 0, 0), -1)
        cv2.putText(output, status,
                    (10, output.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
                    lineType=cv2.LINE_AA)

        cv2.imshow(win_name, output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Ghost stream stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live ghost stream — Camera A view with ghosts from Camera B."
    )
    parser.add_argument("--config-a", required=True, help="Camera A calibration JSON")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--track", nargs="+", default=["person"],
                        help="Classes to ghost (default: person)")
    parser.add_argument("--no-ingest", action="store_true",
                        help="Don't POST Camera A observations (if bridge is already running)")
    args = parser.parse_args()

    run_ghost_stream(args.config_a, args.api_url, args.track, not args.no_ingest)


if __name__ == "__main__":
    main()
