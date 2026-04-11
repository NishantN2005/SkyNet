"""
One-shot photo ingest — runs YOLO on a single image and POSTs to /ingest.

Usage:
    python -m skynet.bridge.photo_ingest --photo photo_a.jpg --config calibration_a.json
"""

from __future__ import annotations

import argparse
import time

import cv2
import httpx
import numpy as np

from skynet.bridge.camera_bridge import _extract_detections, _pixel_to_world, _in_bounds
from skynet.bridge.config import load_config
from skynet.models import ObservationReport, Pose2D


def ingest_photo(photo_path: str, config_path: str) -> None:
    from ultralytics import YOLO

    cfg = load_config(config_path)
    H = np.array(cfg.homography, dtype=np.float64)

    frame = cv2.imread(photo_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {photo_path}")

    print(f"Loading {cfg.yolo_model}...")
    model = YOLO(cfg.yolo_model)

    print(f"Running detection on {photo_path}...")
    results = model(frame, conf=cfg.conf_threshold, verbose=False)

    detected = _extract_detections(results, H, cfg, cfg.camera_id)

    print(f"Detected {len(detected)} object(s):")
    for d in detected:
        print(f"  {d.class_label:15s}  pos=({d.position.x:.3f}, {d.position.y:.3f}) m  "
              f"conf={d.confidence:.2f}  id={d.object_id}")

    if not detected:
        print("  (nothing detected within the calibrated table region)")

    report = ObservationReport(
        robot_id=cfg.camera_id,
        timestamp=time.time(),
        robot_pose=Pose2D(**cfg.camera_pose),
        detected_objects=detected,
        free_cells=[],
        fov_range=999.0,
        fov_angle_rad=3.14159,
    )

    print(f"\nPOSTing to {cfg.api_url}/ingest ...")
    try:
        resp = httpx.post(
            f"{cfg.api_url}/ingest",
            content=report.model_dump_json(),
            headers={"Content-Type": "application/json"},
            timeout=5.0,
        )
        print(f"  {resp.status_code} — {resp.json()}")
    except httpx.RequestError as e:
        print(f"  ERROR: {e}")
        print("  Is the server running? → uvicorn skynet.api:app --port 8000")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run YOLO on a single photo and POST detections to the WorldModel API."
    )
    parser.add_argument("--photo", required=True, help="Path to the image file")
    parser.add_argument("--config", required=True, help="Path to calibration JSON")
    args = parser.parse_args()
    ingest_photo(args.photo, args.config)


if __name__ == "__main__":
    main()
