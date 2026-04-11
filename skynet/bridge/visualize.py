"""
Visualize fused world model from two photos.

Produces a 3-panel figure:
  Left   — Photo A with YOLO bounding boxes
  Center — Photo B with YOLO bounding boxes
  Right  — Top-down fused world map (both cameras combined)

Usage:
    python -m skynet.bridge.visualize \\
        --photo-a without_creatine.png --config-a calibration_a.json \\
        --photo-b with_creatine.png    --config-b calibration_b.json \\
        --out fused_world.png
"""

from __future__ import annotations

import argparse

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from skynet.bridge.camera_bridge import _extract_detections
from skynet.bridge.config import load_config
from skynet.models import DetectedObject

# Colour per camera (matplotlib colours)
_CAM_COLORS = {
    "camera_a": "#3498db",   # blue
    "camera_b": "#e74c3c",   # red
}
_DEFAULT_COLORS = ["#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

# OpenCV BGR equivalents for drawing on photos
_CAM_COLORS_BGR = {
    "camera_a": (219, 152, 52),   # blue in BGR
    "camera_b": (60, 76, 231),    # red in BGR
}


def _run_detection(photo_path: str, config_path: str):
    """Load photo, run YOLO, return (frame_rgb, detections, cfg, H)."""
    from ultralytics import YOLO

    cfg = load_config(config_path)
    H = np.array(cfg.homography, dtype=np.float64)
    frame_bgr = cv2.imread(photo_path)
    if frame_bgr is None:
        raise FileNotFoundError(f"Cannot read: {photo_path}")

    model = YOLO(cfg.yolo_model)
    results = model(frame_bgr, conf=cfg.conf_threshold, verbose=False)
    detected = _extract_detections(results, H, cfg, cfg.camera_id)

    # Annotate frame
    annotated = frame_bgr.copy()
    color_bgr = _CAM_COLORS_BGR.get(cfg.camera_id, (0, 255, 0))
    boxes = results[0].boxes
    if boxes is not None:
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            cls_id = int(box.cls[0])
            label = results[0].names.get(cls_id, str(cls_id))
            conf = float(box.conf[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 6)
            txt = f"{label} {conf:.2f}"
            cv2.putText(annotated, txt, (x1, max(y1 - 15, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, color_bgr, 5)

    frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return frame_rgb, detected, cfg


def _draw_fused_map(ax, all_detections: list[tuple[DetectedObject, str]],
                    world_w: float, world_h: float) -> None:
    """Draw a clean top-down world map with object positions."""
    ax.set_facecolor("#1a1a2e")
    ax.set_xlim(-0.02, world_w + 0.02)
    ax.set_ylim(-0.02, world_h + 0.02)
    ax.set_aspect("equal")
    ax.set_xlabel("World X (m)", color="white", fontsize=10)
    ax.set_ylabel("World Y (m)", color="white", fontsize=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # Table border
    table_rect = plt.Rectangle(
        (0, 0), world_w, world_h,
        fill=False, edgecolor="#666", linewidth=2, linestyle="--"
    )
    ax.add_patch(table_rect)
    ax.text(world_w / 2, world_h + 0.015, "Calibrated table surface",
            ha="center", va="bottom", color="#888", fontsize=8)

    # Grid
    for x in np.arange(0, world_w + 0.01, 0.1):
        ax.axvline(x, color="#333", linewidth=0.4, zorder=0)
    for y in np.arange(0, world_h + 0.01, 0.1):
        ax.axhline(y, color="#333", linewidth=0.4, zorder=0)

    # Camera positions (fixed at origin for this setup)
    cam_positions = {
        "camera_a": (-0.01, world_h / 2),
        "camera_b": (world_w + 0.01, world_h / 2),
    }
    cam_labels = {"camera_a": "Cam A", "camera_b": "Cam B"}
    for cam_id, (cx, cy) in cam_positions.items():
        color = _CAM_COLORS.get(cam_id, "white")
        ax.plot(cx, cy, "s", color=color, markersize=14, zorder=5,
                markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate(cam_labels[cam_id], (cx, cy),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", color=color, fontsize=9, fontweight="bold")

    # Detected objects
    seen: dict[str, list[DetectedObject]] = {}
    for det, cam_id in all_detections:
        seen.setdefault(cam_id, []).append(det)

    plotted_labels: dict[str, str] = {}  # object_id → label for dedup

    for det, cam_id in all_detections:
        color = _CAM_COLORS.get(cam_id, "#2ecc71")
        x, y = det.position.x, det.position.y

        # Circle sized by object footprint
        radius = max(det.width, det.height) / 2
        circle = plt.Circle((x, y), max(radius, 0.015),
                             color=color, alpha=0.35, zorder=3)
        ax.add_patch(circle)
        ax.plot(x, y, "o", color=color, markersize=10, zorder=4,
                markeredgecolor="white", markeredgewidth=1.5)

        # Label — offset slightly to avoid overlap
        label = f"{det.class_label}\n({x:.2f}, {y:.2f})m"
        ax.annotate(label, (x, y),
                    textcoords="offset points", xytext=(10, 8),
                    color=color, fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#1a1a2e", ec=color, alpha=0.8))

    # Legend
    legend_handles = [
        mpatches.Patch(color=_CAM_COLORS["camera_a"], label="camera_a only"),
        mpatches.Patch(color=_CAM_COLORS["camera_b"], label="camera_b only"),
        mpatches.Patch(color="#2ecc71", label="both cameras"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              facecolor="#1a1a2e", edgecolor="#555",
              labelcolor="white", fontsize=8)

    obj_count = len(all_detections)
    cam_a_count = sum(1 for _, c in all_detections if c == "camera_a")
    cam_b_count = sum(1 for _, c in all_detections if c == "camera_b")
    ax.set_title(
        f"Fused World Model — {obj_count} object(s) total\n"
        f"Cam A: {cam_a_count}  |  Cam B: {cam_b_count}",
        color="white", fontsize=10, pad=8,
    )


def run_visualization(
    photo_a: str, config_a: str,
    photo_b: str, config_b: str,
    out_path: str = "fused_world.png",
) -> None:
    print("Running detection on Photo A...")
    frame_a, detected_a, cfg_a = _run_detection(photo_a, config_a)

    print("Running detection on Photo B...")
    frame_b, detected_b, cfg_b = _run_detection(photo_b, config_b)

    world_w = max(cfg_a.world_width_m, cfg_b.world_width_m)
    world_h = max(cfg_a.world_height_m, cfg_b.world_height_m)

    all_detections = (
        [(d, cfg_a.camera_id) for d in detected_a] +
        [(d, cfg_b.camera_id) for d in detected_b]
    )

    print(f"\nCamera A ({cfg_a.camera_id}): {len(detected_a)} object(s)")
    for d in detected_a:
        print(f"  {d.class_label:15s}  ({d.position.x:.3f}, {d.position.y:.3f}) m  conf={d.confidence:.2f}")

    print(f"\nCamera B ({cfg_b.camera_id}): {len(detected_b)} object(s)")
    for d in detected_b:
        print(f"  {d.class_label:15s}  ({d.position.x:.3f}, {d.position.y:.3f}) m  conf={d.confidence:.2f}")

    # Build figure
    fig = plt.figure(figsize=(20, 7), facecolor="#0d0d1a")
    fig.suptitle(
        "SkyNet — Real-World Collaborative Spatial Awareness",
        color="white", fontsize=14, fontweight="bold", y=1.01,
    )

    ax_a = fig.add_subplot(1, 3, 1)
    ax_b = fig.add_subplot(1, 3, 2)
    ax_map = fig.add_subplot(1, 3, 3)

    # Photo A
    ax_a.imshow(frame_a)
    ax_a.set_title(
        f"Camera A — {len(detected_a)} detection(s)\n"
        f"(creatine tub hidden behind box)",
        color=_CAM_COLORS["camera_a"], fontsize=10, pad=6,
    )
    ax_a.axis("off")

    # Photo B
    ax_b.imshow(frame_b)
    ax_b.set_title(
        f"Camera B — {len(detected_b)} detection(s)\n"
        f"(creatine tub visible)",
        color=_CAM_COLORS["camera_b"], fontsize=10, pad=6,
    )
    ax_b.axis("off")

    # Fused map
    _draw_fused_map(ax_map, all_detections, world_w, world_h)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nSaved → {out_path}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize SkyNet fused world model from two photos."
    )
    parser.add_argument("--photo-a", required=True)
    parser.add_argument("--config-a", required=True)
    parser.add_argument("--photo-b", required=True)
    parser.add_argument("--config-b", required=True)
    parser.add_argument("--out", default="fused_world.png")
    args = parser.parse_args()

    run_visualization(
        args.photo_a, args.config_a,
        args.photo_b, args.config_b,
        args.out,
    )


if __name__ == "__main__":
    main()
