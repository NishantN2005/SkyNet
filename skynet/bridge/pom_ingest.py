"""
EPFL POM Dataset ingestion pipeline for SkyNet.

Reads the POM lab sequence (6p-c*.avi) and feeds frames into the SkyNet
WorldModel as if each camera were a separate robot with a known pose.

Ground truth positions from gt_lab_6p.txt are used as robot_pose for each
camera — this is the key difference from the homography-only demo. Each
camera has a fixed real-world position derived from its homography origin.

Usage:
    # Terminal 1 — server
    uvicorn skynet.api:app --host 0.0.0.0 --port 8000

    # Terminal 2 — run ingestion
    python -m skynet.bridge.pom_ingest \\
        --dataset-dir datasets/pom_lab \\
        --cameras 0 1 \\
        --api-url http://localhost:8000

    # Terminal 3 — ghost stream (Camera 0's view + ghosts from Camera 1)
    python -m skynet.bridge.ghost_stream --config-a datasets/pom_lab/config_cam0.json
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import cv2
import httpx
import numpy as np

from skynet.models import DetectedObject, ObservationReport, Pose2D


# ---------------------------------------------------------------------------
# Calibration parsing
# ---------------------------------------------------------------------------

def parse_calibration(cal_path: Path) -> dict[int, np.ndarray]:
    """
    Parse EPFL POM calibration file.
    Returns dict mapping camera_id → 3×3 ground-plane homography (numpy).

    The homography maps image (pixel) coordinates to top-view world coords.
    We use only the ground-plane homography — head-plane is ignored.
    """
    homographies: dict[int, np.ndarray] = {}
    text = cal_path.read_text()

    # Split on camera headers
    camera_blocks = re.split(r"#{5,}\s*\n#\s*Camera\s+(\d+)", text)

    # camera_blocks: [preamble, cam_id, block, cam_id, block, ...]
    i = 1
    while i < len(camera_blocks) - 1:
        cam_id = int(camera_blocks[i])
        block = camera_blocks[i + 1]

        # Extract the ground plane homography (first 3×3 after "Ground plane")
        gp_match = re.search(
            r"Ground plane homography\s*\n"
            r"([-\d.eE+\t ]+)\n([-\d.eE+\t ]+)\n([-\d.eE+\t ]+)",
            block,
        )
        if gp_match:
            rows = []
            for row_str in gp_match.groups():
                rows.append([float(v) for v in row_str.split()])
            homographies[cam_id] = np.array(rows, dtype=np.float64)

        i += 2

    return homographies


# ---------------------------------------------------------------------------
# Ground truth parsing
# ---------------------------------------------------------------------------

def parse_ground_truth(gt_path: Path) -> tuple[int, int, list[list[int]]]:
    """
    Parse POM ground truth file.

    Returns:
        grid_w, grid_h  — grid dimensions
        positions       — list of frames, each frame is list of grid positions
                          per person (-1=undefined, -2=out of bounds, >=0=grid pos)
    """
    lines = gt_path.read_text().strip().splitlines()
    # Line 0: "1" (number of sequences)
    # Line 1: n_frames n_people grid_w grid_h step first_frame last_frame
    header = lines[1].split()
    n_frames = int(header[0])
    n_people = int(header[1])
    grid_w = int(header[2])
    grid_h = int(header[3])

    positions: list[list[int]] = []
    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue
        vals = [int(v) for v in line.split()]
        positions.append(vals)

    return grid_w, grid_h, positions


def grid_pos_to_world(pos: int, grid_w: int, grid_h: int,
                      world_w_m: float = 12.0, world_h_m: float = 8.0
                      ) -> tuple[float, float] | None:
    """Convert a POM grid position integer to world coordinates in metres."""
    if pos < 0:
        return None
    gx = pos % grid_w
    gy = pos // grid_w
    wx = (gx + 0.5) * (world_w_m / grid_w)
    wy = (gy + 0.5) * (world_h_m / grid_h)
    return wx, wy


# ---------------------------------------------------------------------------
# Camera pose estimation from homography
# ---------------------------------------------------------------------------

def estimate_camera_world_pose(
    H: np.ndarray,
    img_w: int, img_h: int,
    tv_w: float, tv_h: float,
    world_w_m: float, world_h_m: float,
) -> Pose2D:
    """
    Estimate camera world pose from homography.
    Maps image centre → top-view coords → metres.
    """
    cx, cy = img_w / 2.0, img_h / 2.0
    p = np.array([cx, cy, 1.0])
    wp = H @ p
    wp /= wp[2]
    wx = (wp[0] / tv_w) * world_w_m
    wy = (wp[1] / tv_h) * world_h_m
    return Pose2D(x=float(wx), y=float(wy), heading_rad=0.0)


# ---------------------------------------------------------------------------
# Pixel → world via homography
# ---------------------------------------------------------------------------

def pixel_to_world(
    px: float, py: float, H: np.ndarray,
    tv_w: float, tv_h: float,
    world_w_m: float, world_h_m: float,
) -> tuple[float, float]:
    """
    Map a pixel coordinate to world metres via the POM homography.

    The POM homography maps image pixels → top-view pixel coordinates
    (range ~0-500). We normalise by the top-view canvas size to get metres.
    """
    p = np.array([px, py, 1.0], dtype=np.float64)
    wp = H @ p
    wp /= wp[2]
    wx = (wp[0] / tv_w) * world_w_m
    wy = (wp[1] / tv_h) * world_h_m
    return float(wx), float(wy)


# ---------------------------------------------------------------------------
# Ingestion loop
# ---------------------------------------------------------------------------

def run_pom_ingest(
    dataset_dir: str,
    camera_ids: list[int],
    api_url: str = "http://localhost:8000",
    fps: float = 5.0,
    world_w_m: float = 6.0,
    world_h_m: float = 5.0,
    # POM top-view canvas dimensions in pixels (from dataset documentation)
    tv_w: float = 480.0,
    tv_h: float = 360.0,
    conf_threshold: float = 0.25,
    yolo_model: str = "yolov8n.pt",
    start_frame: int = 400,
    max_frames: int | None = None,
    save_configs: bool = True,
) -> None:
    from ultralytics import YOLO

    base = Path(dataset_dir)
    cal_path = base / "calibration-6p.txt"
    gt_path = base / "gt_lab_6p.txt"

    print("Parsing calibration...")
    homographies = parse_calibration(cal_path)

    print("Parsing ground truth...")
    grid_w, grid_h, gt_frames = parse_ground_truth(gt_path)
    print(f"  Grid: {grid_w}×{grid_h}, {len(gt_frames)} frames, "
          f"world size: {world_w_m}m × {world_h_m}m")

    print(f"Loading {yolo_model}...")
    model = YOLO(yolo_model)

    # Open video captures for each camera
    caps: dict[int, cv2.VideoCapture] = {}
    camera_poses: dict[int, Pose2D] = {}

    for cam_id in camera_ids:
        video_path = base / f"6p-c{cam_id}.avi"
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        caps[cam_id] = cap

        H = homographies[cam_id]
        cap_tmp = cv2.VideoCapture(str(base / f"6p-c{cam_id}.avi"))
        img_w = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_h = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_tmp.release()
        camera_poses[cam_id] = estimate_camera_world_pose(
            H, img_w, img_h, tv_w, tv_h, world_w_m, world_h_m
        )
        print(f"  Camera {cam_id}: {img_w}x{img_h}  pose≈"
              f"({camera_poses[cam_id].x:.2f}, {camera_poses[cam_id].y:.2f})m")

        # Save a calibration JSON for ghost_stream to use
        if save_configs:
            import json
            from skynet.bridge.config import CameraConfig, save_config
            cfg = CameraConfig(
                camera_id=f"camera_{cam_id}",
                camera_source=str(video_path),
                camera_pose={
                    "x": camera_poses[cam_id].x,
                    "y": camera_poses[cam_id].y,
                    "heading_rad": 0.0,
                },
                world_width_m=world_w_m,
                world_height_m=world_h_m,
                api_url=api_url,
                homography=H.tolist(),
                conf_threshold=conf_threshold,
            )
            cfg_path = base / f"config_cam{cam_id}.json"
            save_config(cfg, cfg_path)

    frame_interval = 1.0 / fps
    frame_idx = 0
    client = httpx.Client(timeout=2.0)

    print(f"\nIngesting {len(camera_ids)} cameras at {fps} fps "
          f"(press Ctrl+C to stop)...\n")

    try:
        while True:
            frames: dict[int, np.ndarray] = {}
            any_valid = False

            for cam_id, cap in caps.items():
                ret, frame = cap.read()
                if ret:
                    frames[cam_id] = frame
                    any_valid = True

            if not any_valid:
                print("All videos exhausted.")
                break

            if max_frames and frame_idx >= max_frames:
                break

            now = time.time()

            for cam_id, frame in frames.items():
                H = homographies[cam_id]
                robot_id = f"camera_{cam_id}"
                pose = camera_poses[cam_id]

                # YOLO detection
                results = model(frame, conf=conf_threshold, verbose=False)
                boxes = results[0].boxes
                names = results[0].names
                detected: list[DetectedObject] = []

                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        label = names.get(cls_id, str(cls_id))
                        if label != "person":
                            continue  # only track people for this demo

                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy()
                        # Use bottom-centre of bounding box as foot position
                        foot_px = float((xyxy[0] + xyxy[2]) / 2)
                        foot_py = float(xyxy[3])

                        wx, wy = pixel_to_world(
                            foot_px, foot_py, H,
                            tv_w, tv_h, world_w_m, world_h_m,
                        )

                        # Bounds check with small margin
                        margin = 0.5
                        if not (-margin <= wx <= world_w_m + margin and
                                -margin <= wy <= world_h_m + margin):
                            continue
                        wx = max(0.0, min(world_w_m, wx))
                        wy = max(0.0, min(world_h_m, wy))

                        obj_id = f"person_cam{cam_id}_{len(detected)}"
                        detected.append(DetectedObject(
                            object_id=obj_id,
                            class_label="person",
                            position=Pose2D(x=wx, y=wy, heading_rad=0.0),
                            confidence=conf,
                            observed_by=robot_id,
                            timestamp=now,
                            width=0.5,
                            height=1.75,
                        ))

                report = ObservationReport(
                    robot_id=robot_id,
                    timestamp=now,
                    robot_pose=pose,
                    detected_objects=detected,
                    free_cells=[],
                    fov_range=999.0,
                    fov_angle_rad=3.14159,
                )

                try:
                    resp = client.post(
                        f"{api_url}/ingest",
                        content=report.model_dump_json(),
                        headers={"Content-Type": "application/json"},
                    )
                    status = resp.status_code
                except httpx.RequestError as e:
                    status = f"ERR:{e}"

                print(f"  frame={frame_idx:04d}  cam={cam_id}  "
                      f"detections={len(detected)}  POST={status}")

            # Publish current frame index so the visualizer can seek to it
            try:
                client.post(
                    f"{api_url}/frames",
                    json={f"camera_{cam_id}": start_frame + frame_idx
                          for cam_id in camera_ids},
                    headers={"Content-Type": "application/json"},
                )
            except httpx.RequestError:
                pass

            frame_idx += 1
            time.sleep(frame_interval)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        for cap in caps.values():
            cap.release()
        client.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest EPFL POM lab sequence into SkyNet WorldModel."
    )
    parser.add_argument("--dataset-dir", default="datasets/pom_lab",
                        help="Directory containing 6p-c*.avi and calibration files")
    parser.add_argument("--cameras", nargs="+", type=int, default=[0, 1],
                        help="Camera IDs to ingest (default: 0 1)")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="Ingest rate in frames per second")
    parser.add_argument("--world-width", type=float, default=6.0,
                        help="World width in metres")
    parser.add_argument("--world-height", type=float, default=5.0,
                        help="World height in metres")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="YOLO confidence threshold")
    parser.add_argument("--start-frame", type=int, default=400,
                        help="Skip to this frame before ingesting (default: 400)")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--yolo-model", default="yolov8n.pt",
                        help="YOLO model weights (default: yolov8n.pt)")
    args = parser.parse_args()

    run_pom_ingest(
        dataset_dir=args.dataset_dir,
        camera_ids=args.cameras,
        api_url=args.api_url,
        fps=args.fps,
        world_w_m=args.world_width,
        world_h_m=args.world_height,
        conf_threshold=args.conf,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        yolo_model=args.yolo_model,
    )


if __name__ == "__main__":
    main()
