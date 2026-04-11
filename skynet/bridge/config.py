"""
CameraConfig — per-camera configuration persisted as JSON.

Workflow:
  1. calibrate.py creates the file with homography filled in.
  2. camera_bridge.py loads it at startup and never writes it.

camera_source accepts:
  - int   → OpenCV device index (0 = built-in webcam, 1 = next device)
  - str   → any URL that cv2.VideoCapture accepts, e.g.
              "http://192.168.1.42:8080/video"   (IP Webcam / DroidCam)
              "/dev/video2"                        (Linux device node)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Union


@dataclass
class CameraConfig:
    # Identity
    camera_id: str                          # e.g. "laptop" or "phone"
    camera_source: Union[int, str] = 0      # device index or stream URL

    # Fixed world position of this camera (used as the robot_pose in reports)
    # For a tabletop setup, place the origin at one corner of the table.
    camera_pose: dict = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "heading_rad": 0.0})

    # Real-world dimensions of the calibrated table surface (meters)
    world_width_m: float = 1.2
    world_height_m: float = 0.8

    # WorldModel API endpoint
    api_url: str = "http://localhost:8000"

    # 3×3 perspective homography matrix as a nested list [[...], [...], [...]]
    # Filled in by calibrate.py; None means "not yet calibrated".
    homography: list | None = None

    # Detection settings
    yolo_model: str = "yolov8n.pt"          # nano is fast enough for tabletop
    conf_threshold: float = 0.4
    ingest_fps: float = 10.0               # max frames per second to POST

    # If True, camera_bridge fetches live pose from GET /slam_pose/{camera_id}
    # each frame instead of using the static camera_pose above.
    # Set to True when the ARKit iPhone app is running.
    use_slam_pose: bool = False


def load_config(path: str | Path) -> CameraConfig:
    """Load a CameraConfig from a JSON file."""
    data = json.loads(Path(path).read_text())
    return CameraConfig(**data)


def save_config(cfg: CameraConfig, path: str | Path) -> None:
    """Persist a CameraConfig to a JSON file."""
    Path(path).write_text(json.dumps(asdict(cfg), indent=2))
    print(f"Config saved → {path}")
