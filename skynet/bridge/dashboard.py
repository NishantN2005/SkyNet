"""
Live SkyNet dashboard — real-time fused world view.

Shows:
  Left  — Camera A's live feed with its own YOLO detections
  Right — Top-down table map updating every second:
            Blue  = seen by Camera A
            Red   = seen by Camera B
            Cyan  = hidden from A (in WorldModel but not seen by A)

Usage:
    python -m skynet.bridge.dashboard --config-a calibration_a_live.json \\
                                      --config-b calibration_b_live.json
"""

from __future__ import annotations

import argparse
import time
from copy import deepcopy

import cv2
import httpx
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from skynet.bridge.camera_bridge import _extract_detections
from skynet.bridge.config import load_config
from skynet.models import WorldState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grab_frame(camera_source) -> np.ndarray | None:
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        return None
    for _ in range(2):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def _fetch_world(api_url: str) -> WorldState | None:
    try:
        resp = httpx.get(f"{api_url}/world", timeout=2.0)
        resp.raise_for_status()
        return WorldState(**resp.json())
    except Exception:
        return None


def _annotate_frame(frame_bgr: np.ndarray, results, color_bgr) -> np.ndarray:
    out = frame_bgr.copy()
    boxes = results[0].boxes
    names = results[0].names
    if boxes is not None:
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            cls_id = int(box.cls[0])
            label = names.get(cls_id, str(cls_id))
            conf = float(box.conf[0])
            cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, 4)
            cv2.putText(out, f"{label} {conf:.2f}", (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_bgr, 3,
                        lineType=cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def run_dashboard(
    config_a: str,
    config_b: str,
    api_url: str = "http://localhost:8000",
    refresh_sec: float = 1.0,
) -> None:
    from ultralytics import YOLO

    cfg_a = load_config(config_a)
    cfg_b = load_config(config_b)
    H_a = np.array(cfg_a.homography, dtype=np.float64)
    world_w = cfg_a.world_width_m
    world_h = cfg_a.world_height_m

    print("Loading YOLO model...")
    model = YOLO(cfg_a.yolo_model)

    # --- Figure setup ---
    fig = plt.figure(figsize=(20, 8), facecolor="#0d0d1a")
    fig.suptitle("SkyNet — Live Collaborative Spatial Awareness",
                 color="white", fontsize=14, fontweight="bold")

    ax_cam = fig.add_subplot(1, 2, 1)   # Camera A live feed
    ax_map = fig.add_subplot(1, 2, 2)   # Top-down fused map

    ax_cam.axis("off")
    ax_cam.set_facecolor("#0d0d1a")

    # Placeholder image
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    im_display = ax_cam.imshow(placeholder)
    ax_cam.set_title("Camera A — live feed", color="#3498db", fontsize=11)

    plt.tight_layout()

    # State shared across animation frames
    state = {"tick": 0}

    def _update(frame_idx):
        state["tick"] += 1
        tick = state["tick"]

        # --- Grab Camera A frame and run YOLO ---
        frame_bgr = _grab_frame(cfg_a.camera_source)
        detected_a = []
        if frame_bgr is not None:
            results_a = model(frame_bgr, conf=cfg_a.conf_threshold, verbose=False)
            detected_a = _extract_detections(results_a, H_a, cfg_a, cfg_a.camera_id)
            annotated = _annotate_frame(frame_bgr, results_a, (219, 152, 52))
            im_display.set_data(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            ax_cam.set_title(
                f"Camera A — {len(detected_a)} detection(s)  [tick {tick}]",
                color="#3498db", fontsize=11,
            )

        # --- Fetch WorldModel state ---
        world = _fetch_world(api_url)

        # --- Redraw map ---
        ax_map.cla()
        ax_map.set_facecolor("#1a1a2e")
        ax_map.set_xlim(-0.02, world_w + 0.02)
        ax_map.set_ylim(-0.02, world_h + 0.02)
        ax_map.set_aspect("equal")
        ax_map.set_xlabel("World X (m)", color="white", fontsize=9)
        ax_map.set_ylabel("World Y (m)", color="white", fontsize=9)
        ax_map.tick_params(colors="white")
        for spine in ax_map.spines.values():
            spine.set_edgecolor("#444")

        # Table border
        rect = plt.Rectangle((0, 0), world_w, world_h,
                              fill=False, edgecolor="#666",
                              linewidth=2, linestyle="--")
        ax_map.add_patch(rect)

        # Grid
        for x in np.arange(0, world_w + 0.01, 0.1):
            ax_map.axvline(x, color="#2a2a3e", linewidth=0.5)
        for y in np.arange(0, world_h + 0.01, 0.1):
            ax_map.axhline(y, color="#2a2a3e", linewidth=0.5)

        # Camera icons
        ax_map.plot(-0.01, world_h / 2, "s", color="#3498db",
                    markersize=14, markeredgecolor="white", markeredgewidth=1.5,
                    zorder=5)
        ax_map.annotate("Cam A", (-0.01, world_h / 2),
                        textcoords="offset points", xytext=(0, 12),
                        ha="center", color="#3498db", fontsize=8, fontweight="bold")

        ax_map.plot(world_w + 0.01, world_h / 2, "s", color="#e74c3c",
                    markersize=14, markeredgecolor="white", markeredgewidth=1.5,
                    zorder=5)
        ax_map.annotate("Cam B", (world_w + 0.01, world_h / 2),
                        textcoords="offset points", xytext=(0, 12),
                        ha="center", color="#e74c3c", fontsize=8, fontweight="bold")

        ids_seen_by_a = {d.object_id for d in detected_a}

        if world is not None:
            for obj in world.objects.values():
                x, y = obj.position.x, obj.position.y

                is_hidden = (
                    obj.observed_by != cfg_a.camera_id
                    and obj.object_id not in ids_seen_by_a
                    and obj.class_label != "book"
                    and 0.0 <= x <= world_w
                    and 0.0 <= y <= world_h
                )

                if obj.observed_by == cfg_a.camera_id:
                    color = "#3498db"
                    marker = "o"
                    size = 120
                elif is_hidden:
                    color = "#00ffcc"
                    marker = "*"
                    size = 300
                else:
                    color = "#e74c3c"
                    marker = "o"
                    size = 120

                ax_map.scatter(x, y, c=color, s=size, marker=marker,
                               zorder=4, edgecolors="white", linewidths=1.0)

                prefix = "HIDDEN: " if is_hidden else ""
                ax_map.annotate(
                    f"{prefix}{obj.class_label}\n({x:.2f},{y:.2f})m",
                    (x, y),
                    textcoords="offset points",
                    xytext=(8, 6),
                    color=color, fontsize=7, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#1a1a2e",
                              ec=color, alpha=0.85),
                )

        hidden_count = sum(
            1 for obj in (world.objects.values() if world else [])
            if obj.observed_by != cfg_a.camera_id
            and obj.object_id not in ids_seen_by_a
            and obj.class_label != "book"
            and 0.0 <= obj.position.x <= world_w
            and 0.0 <= obj.position.y <= world_h
        )

        status = "connected" if world else "NO SERVER"
        ax_map.set_title(
            f"Fused World Map — {len(world.objects) if world else 0} object(s) total  "
            f"|  {hidden_count} hidden from A  |  {status}",
            color="#00ffcc" if hidden_count > 0 else "white",
            fontsize=10, pad=8,
        )

        # Legend
        legend_handles = [
            mpatches.Patch(color="#3498db", label="Seen by Camera A"),
            mpatches.Patch(color="#e74c3c", label="Seen by Camera B"),
            mpatches.Patch(color="#00ffcc", label="Hidden from A ★"),
        ]
        ax_map.legend(handles=legend_handles, loc="lower right",
                      facecolor="#1a1a2e", edgecolor="#555",
                      labelcolor="white", fontsize=8)

        fig.canvas.draw_idle()

    ani = FuncAnimation(fig, _update, interval=int(refresh_sec * 1000),
                        cache_frame_data=False)

    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live SkyNet dashboard — Camera A feed + fused world map."
    )
    parser.add_argument("--config-a", required=True, help="Camera A calibration JSON")
    parser.add_argument("--config-b", required=True, help="Camera B calibration JSON")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--refresh", type=float, default=1.0,
                        help="Refresh interval in seconds (default: 1.0)")
    args = parser.parse_args()

    run_dashboard(args.config_a, args.config_b, args.api_url, args.refresh)


if __name__ == "__main__":
    main()
