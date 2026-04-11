"""
Camera bridge — connects one camera to the SkyNet WorldModel HTTP API.

Each detected object is mapped from pixel space to world space via the
perspective homography computed by calibrate.py, then posted to POST /ingest
as an ObservationReport.

Run one instance per camera:
    python -m skynet.bridge.camera_bridge --config calibration_laptop.json
    python -m skynet.bridge.camera_bridge --config calibration_phone.json

Press 'q' in the preview window to stop.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import httpx
import numpy as np

from skynet.bridge.config import CameraConfig, load_config
from skynet.models import DetectedObject, ObservationReport, Pose2D

# Colour for bounding box overlay in preview window
_BOX_COLOUR = (0, 255, 0)
_TEXT_COLOUR = (0, 255, 0)


# ---------------------------------------------------------------------------
# Homography
# ---------------------------------------------------------------------------

def _load_homography(cfg: CameraConfig) -> np.ndarray:
    if cfg.homography is None:
        print("ERROR: No homography in config. Run calibrate.py first.", file=sys.stderr)
        sys.exit(1)
    return np.array(cfg.homography, dtype=np.float64)


def _pixel_to_world(H: np.ndarray, cx: float, cy: float) -> tuple[float, float]:
    """Map a single pixel point to world coordinates via homography H."""
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H.astype(np.float32))
    return float(result[0, 0, 0]), float(result[0, 0, 1])


def _in_bounds(wx: float, wy: float, cfg: CameraConfig) -> bool:
    """Return True if the world point is within the calibrated table surface."""
    return 0.0 <= wx <= cfg.world_width_m and 0.0 <= wy <= cfg.world_height_m


# ---------------------------------------------------------------------------
# Object ID
# ---------------------------------------------------------------------------

def _make_object_id(class_label: str, wx: float, wy: float, cell_m: float = 0.05) -> str:
    """
    Stable object ID based on class + grid cell (5 cm cells by default).

    Two detections of the same class in the same 5 cm cell get the same ID,
    so the WorldModel treats them as updates to the same object rather than
    creating duplicates. Good enough for a tabletop demo.
    """
    gx = int(wx / cell_m)
    gy = int(wy / cell_m)
    return f"{class_label}_{gx}_{gy}"


# ---------------------------------------------------------------------------
# YOLO inference
# ---------------------------------------------------------------------------

def _run_yolo(model, frame: np.ndarray, conf: float):
    """Run YOLOv8 inference and return the Results object."""
    return model(frame, conf=conf, verbose=False)


def _extract_detections(
    results,
    H: np.ndarray,
    cfg: CameraConfig,
    camera_id: str,
) -> list[DetectedObject]:
    """
    Convert YOLO results to DetectedObject list.

    Only includes detections whose bounding-box centre maps to a point
    within the calibrated table region.
    """
    detected: list[DetectedObject] = []
    now = time.time()

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return detected

    names = results[0].names  # int → class label string

    for box in boxes:
        conf_val = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_label = names.get(cls_id, str(cls_id))

        # Bounding box centre in pixels
        xyxy = box.xyxy[0].cpu().numpy()
        cx_px = (xyxy[0] + xyxy[2]) / 2.0
        cy_px = (xyxy[1] + xyxy[3]) / 2.0

        wx, wy = _pixel_to_world(H, cx_px, cy_px)

        if not _in_bounds(wx, wy, cfg):
            continue

        # Approximate real-world size from pixel bbox and homography scale
        # (rough: map corners of bbox, compute diagonal in world space)
        tl_w = _pixel_to_world(H, xyxy[0], xyxy[1])
        br_w = _pixel_to_world(H, xyxy[2], xyxy[3])
        obj_w = abs(br_w[0] - tl_w[0])
        obj_h = abs(br_w[1] - tl_w[1])

        object_id = _make_object_id(class_label, wx, wy)

        detected.append(DetectedObject(
            object_id=object_id,
            class_label=class_label,
            position=Pose2D(x=wx, y=wy),
            confidence=conf_val,
            observed_by=camera_id,
            timestamp=now,
            width=max(obj_w, 0.01),
            height=max(obj_h, 0.01),
        ))

    return detected


# ---------------------------------------------------------------------------
# Preview window
# ---------------------------------------------------------------------------

def _draw_preview(
    frame: np.ndarray,
    results,
    detected: list[DetectedObject],
    camera_id: str,
    fps: float,
) -> np.ndarray:
    """Annotate frame with YOLO boxes and world coordinate labels."""
    vis = frame.copy()

    boxes = results[0].boxes
    names = results[0].names

    if boxes is not None:
        for box, det in zip(boxes, detected):
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(vis, (x1, y1), (x2, y2), _BOX_COLOUR, 2)
            label = (f"{det.class_label} "
                     f"({det.position.x:.2f},{det.position.y:.2f})m "
                     f"{det.confidence:.2f}")
            cv2.putText(vis, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT_COLOUR, 1)

    # Status bar
    status = f"{camera_id}  |  {len(detected)} obj  |  {fps:.1f} fps  |  q=quit"
    cv2.putText(vis, status, (10, vis.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    return vis


# ---------------------------------------------------------------------------
# Main bridge loop
# ---------------------------------------------------------------------------

def run_bridge(cfg: CameraConfig, show_preview: bool = True) -> None:
    """
    Main loop: capture → YOLO → map to world → POST to WorldModel.

    Runs until the user presses 'q' in the preview window or Ctrl-C.
    """
    H = _load_homography(cfg)

    # Load YOLO model (downloads on first run, ~6 MB for yolov8n)
    from ultralytics import YOLO
    print(f"Loading {cfg.yolo_model}...")
    model = YOLO(cfg.yolo_model)
    print("Model ready.")

    cap = cv2.VideoCapture(cfg.camera_source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera: {cfg.camera_source!r}", file=sys.stderr)
        sys.exit(1)

    # Determine whether to use live SLAM pose or static calibration pose
    use_slam_pose = cfg.use_slam_pose
    static_camera_pose = Pose2D(**cfg.camera_pose)
    min_interval = 1.0 / cfg.ingest_fps
    last_post_time = 0.0
    fps_counter = 0
    fps_timer = time.time()
    display_fps = 0.0

    print(f"\nBridge running — camera_id={cfg.camera_id!r}  api={cfg.api_url}")
    print("Press 'q' in preview window to stop.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: Failed to read frame, retrying...", file=sys.stderr)
                time.sleep(0.05)
                continue

            results = _run_yolo(model, frame, cfg.conf_threshold)
            detected = _extract_detections(results, H, cfg, cfg.camera_id)

            now = time.time()

            # Rate-limit POST calls to ingest_fps
            if now - last_post_time >= min_interval:
                # Use live SLAM pose if enabled, fall back to static pose
                camera_pose = static_camera_pose
                if use_slam_pose:
                    try:
                        pose_resp = httpx.get(
                            f"{cfg.api_url}/slam_pose/{cfg.camera_id}",
                            timeout=0.5,
                        )
                        if pose_resp.status_code == 200:
                            pd = pose_resp.json()
                            camera_pose = Pose2D(
                                x=pd["x"],
                                y=pd["y"],
                                heading_rad=pd["heading_rad"],
                            )
                    except httpx.RequestError:
                        pass  # keep static pose if SLAM endpoint unreachable

                report = ObservationReport(
                    robot_id=cfg.camera_id,
                    timestamp=now,
                    robot_pose=camera_pose,
                    detected_objects=detected,
                    free_cells=[],          # no depth → skip free space
                    fov_range=999.0,        # camera covers full calibrated region
                    fov_angle_rad=3.14159,
                )
                try:
                    httpx.post(
                        f"{cfg.api_url}/ingest",
                        content=report.model_dump_json(),
                        headers={"Content-Type": "application/json"},
                        timeout=1.0,
                    )
                except httpx.RequestError as e:
                    print(f"  POST failed: {e}")

                last_post_time = now

                if detected:
                    obj_summary = ", ".join(
                        f"{d.class_label}@({d.position.x:.2f},{d.position.y:.2f})m"
                        for d in detected
                    )
                    print(f"  [{cfg.camera_id}] {obj_summary}")

            # FPS counter
            fps_counter += 1
            if now - fps_timer >= 1.0:
                display_fps = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            if show_preview:
                vis = _draw_preview(frame, results, detected, cfg.camera_id, display_fps)
                cv2.imshow(f"SkyNet Bridge — {cfg.camera_id}", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Bridge stopped.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a SkyNet camera bridge (capture → YOLO → POST /ingest).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True,
                        help="Path to calibration JSON (output of calibrate.py)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable OpenCV preview window (headless mode)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_bridge(cfg, show_preview=not args.no_preview)


if __name__ == "__main__":
    main()
