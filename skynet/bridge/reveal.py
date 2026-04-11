"""
Reveal — queries the shared WorldModel API for objects Camera A couldn't see,
then projects them back into Camera A's image at the correct pixel location.

This is the fully wired version: it uses the real pipeline end-to-end.
  1. Runs YOLO on Photo A to get Camera A's direct detections.
  2. Calls GET /world on the running WorldModel server to get the fused state
     (which includes everything Camera B ingested via POST /ingest).
  3. Any object in the WorldModel not seen by Camera A is a hidden object.
  4. Projects those hidden objects into Camera A's pixel space and ghosts them in.

Usage:
    # First make sure the server is running and both photos have been ingested:
    uvicorn skynet.api:app --port 8000
    python -m skynet.bridge.photo_ingest --photo without_creatine.png --config calibration_a.json
    python -m skynet.bridge.photo_ingest --photo with_creatine.png    --config calibration_b.json

    # Then run reveal:
    python -m skynet.bridge.reveal \\
        --photo-a without_creatine.png --config-a calibration_a.json \\
        --out reveal.png
"""

from __future__ import annotations

import argparse

import cv2
import httpx
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from skynet.bridge.camera_bridge import _extract_detections
from skynet.bridge.config import load_config
from skynet.models import DetectedObject, WorldState


# ---------------------------------------------------------------------------
# Homography helpers
# ---------------------------------------------------------------------------

def _world_to_pixel(H: np.ndarray, wx: float, wy: float) -> tuple[int, int]:
    """Map world coords → pixel coords using inverse homography."""
    H_inv = np.linalg.inv(H)
    pt = np.array([[[wx, wy]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H_inv.astype(np.float32))
    return int(result[0, 0, 0]), int(result[0, 0, 1])


def _world_box_to_pixels(
    H: np.ndarray,
    wx: float, wy: float,
    width_m: float, height_m: float,
) -> np.ndarray:
    """
    Project a world-space bounding box (centre + size) into pixel coordinates.
    Returns a (4, 2) array of corner pixel positions: TL, TR, BR, BL.
    """
    hw = width_m / 2
    hh = height_m / 2
    world_corners = np.array([
        [wx - hw, wy - hh],
        [wx + hw, wy - hh],
        [wx + hw, wy + hh],
        [wx - hw, wy + hh],
    ], dtype=np.float32)

    H_inv = np.linalg.inv(H).astype(np.float32)
    px_corners = cv2.perspectiveTransform(world_corners[np.newaxis], H_inv)[0]
    return px_corners.astype(int)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_ghost_object(
    frame: np.ndarray,
    H_a: np.ndarray,
    det: DetectedObject,
    label: str,
) -> np.ndarray:
    """
    Draw a glowing ghost outline in Camera A's image at the world position
    of `det`, which was detected by Camera B.
    """
    out = frame.copy()

    # Use a minimum visual size so it's always visible even if YOLO's
    # world-space size estimate is tiny
    vis_width = max(det.width, 0.06)
    vis_height = max(det.height, 0.06)

    corners = _world_box_to_pixels(
        H_a,
        det.position.x, det.position.y,
        vis_width, vis_height,
    )

    # Glow effect — draw multiple expanding outlines in decreasing opacity
    glow_layers = [
        (30, (0, 255, 255), 0.12),   # outer glow (cyan, very transparent)
        (18, (0, 255, 255), 0.22),
        (10, (0, 230, 200), 0.35),
        (5,  (0, 210, 180), 0.55),
        (2,  (255, 255, 255), 0.9),  # sharp white inner edge
    ]

    for thickness, color, alpha in glow_layers:
        overlay = out.copy()
        cv2.polylines(overlay, [corners], isClosed=True, color=color,
                      thickness=thickness, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

    # Fill with very subtle transparent tint
    fill_overlay = out.copy()
    cv2.fillPoly(fill_overlay, [corners], color=(0, 255, 200))
    cv2.addWeighted(fill_overlay, 0.08, out, 0.92, 0, out)

    # Label — positioned above the box
    cx = int(corners[:, 0].mean())
    cy = int(corners[:, 1].min()) - 20

    # Background pill for readability
    label_text = f"HIDDEN: {label}"
    (tw, th), baseline = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3
    )
    cv2.rectangle(out,
                  (cx - tw // 2 - 12, cy - th - 8),
                  (cx + tw // 2 + 12, cy + baseline + 4),
                  (0, 60, 50), -1)
    cv2.putText(out, label_text,
                (cx - tw // 2, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 200), 3,
                lineType=cv2.LINE_AA)

    # Arrow pointing to the ghost box centre
    centre_px = _world_to_pixel(H_a, det.position.x, det.position.y)
    cv2.arrowedLine(
        out,
        (cx, cy + th // 2 + 10),
        centre_px,
        (0, 255, 200), 3, tipLength=0.15, line_type=cv2.LINE_AA,
    )

    return out


def _annotate_own_detections(
    frame: np.ndarray,
    detections: list[DetectedObject],
) -> np.ndarray:
    """Draw Camera A's own detections (plain boxes, no ghost effect)."""
    out = frame.copy()
    for det in detections:
        corners = _world_box_to_pixels(
            np.eye(3),  # not used — we draw directly from results below
            det.position.x, det.position.y,
            det.width, det.height,
        )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _capture_frame(camera_source) -> np.ndarray:
    """Grab a single frame from a live camera source."""
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera source: {camera_source!r}")
    # Discard a few frames so auto-exposure settles
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError("Failed to capture frame from camera")
    return frame


def run_reveal(
    config_a: str,
    photo_a: str | None = None,
    api_url: str = "http://localhost:8000",
    out_path: str = "reveal.png",
) -> None:
    from copy import deepcopy
    from ultralytics import YOLO

    cfg_a = load_config(config_a)
    H_a = np.array(cfg_a.homography, dtype=np.float64)
    world_w = cfg_a.world_width_m
    world_h = cfg_a.world_height_m

    if photo_a:
        frame_a_bgr = cv2.imread(photo_a)
        if frame_a_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {photo_a}")
    else:
        print(f"Capturing live frame from camera source: {cfg_a.camera_source}")
        frame_a_bgr = _capture_frame(cfg_a.camera_source)

    # --- Step 1: Run YOLO on Photo A to get Camera A's direct detections ---
    print("Loading YOLO model...")
    model = YOLO(cfg_a.yolo_model)

    print("Detecting in Photo A...")
    results_a = model(frame_a_bgr, conf=cfg_a.conf_threshold, verbose=False)
    detected_a = _extract_detections(results_a, H_a, cfg_a, cfg_a.camera_id)

    # --- Step 2: Query the shared WorldModel for the full fused state ---
    print(f"Querying WorldModel at {api_url}/world ...")
    try:
        resp = httpx.get(f"{api_url}/world", timeout=5.0)
        resp.raise_for_status()
    except httpx.RequestError as e:
        print(f"\nERROR: Cannot reach WorldModel server — {e}")
        print("Make sure the server is running:  uvicorn skynet.api:app --port 8000")
        return

    world_state = WorldState(**resp.json())
    all_world_objects = list(world_state.objects.values())

    # --- Step 3: Hidden = in WorldModel but NOT reported by Camera A ---
    ids_seen_by_a = {d.object_id for d in detected_a}
    labels_seen_by_a = {d.class_label for d in detected_a}

    def _is_already_seen_by_a(world_obj, tol=0.25):
        """
        Return True if Camera A already detected an object at roughly the same
        location. World objects from Camera B are in B's coordinate frame
        (opposite end of table), so mirror before comparing.
        """
        if world_obj.observed_by == cfg_a.camera_id:
            return True  # Camera A reported this itself
        mx = world_w - world_obj.position.x
        my = world_h - world_obj.position.y
        for da in detected_a:
            if da.class_label != world_obj.class_label:
                continue
            dist = ((da.position.x - mx)**2 + (da.position.y - my)**2) ** 0.5
            if dist < tol:
                return True
        return False

    hidden_objects = [
        obj for obj in all_world_objects
        if not _is_already_seen_by_a(obj)
        and obj.observed_by != cfg_a.camera_id
        and obj.class_label != "book"   # filter shelf false-positives
        and 0.0 <= (world_w - obj.position.x) <= world_w
        and 0.0 <= (world_h - obj.position.y) <= world_h
    ]

    print(f"\nCamera A sees directly : {[d.class_label for d in detected_a]}")
    print(f"WorldModel knows about : {[o.class_label for o in all_world_objects]}")
    print(f"Hidden from A (to ghost): {[o.class_label for o in hidden_objects]}")
    for o in hidden_objects:
        print(f"  → '{o.class_label}'  observed_by={o.observed_by}"
              f"  world_pos=({o.position.x:.3f}, {o.position.y:.3f})"
              f"  conf={o.confidence:.2f}")

    # --- Build Camera A annotated image (its own detections only) ---
    frame_a_annotated = frame_a_bgr.copy()
    boxes_a = results_a[0].boxes
    if boxes_a is not None:
        for box in boxes_a:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            cls_id = int(box.cls[0])
            lbl = results_a[0].names.get(cls_id, str(cls_id))
            conf = float(box.conf[0])
            cv2.rectangle(frame_a_annotated, (x1, y1), (x2, y2), (52, 152, 219), 6)
            cv2.putText(frame_a_annotated, f"{lbl} {conf:.2f}",
                        (x1, max(y1 - 15, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (52, 152, 219), 5,
                        lineType=cv2.LINE_AA)

    # --- Build revealed image — Camera A + ghost outlines of hidden objects ---
    # Mirror world coords from Camera B's frame into Camera A's frame
    # (they calibrated from opposite ends → 180° rotation around table centre)
    frame_revealed = frame_a_annotated.copy()
    for det in hidden_objects:
        det_mirrored = deepcopy(det)
        det_mirrored.position.x = world_w - det.position.x
        det_mirrored.position.y = world_h - det.position.y
        frame_revealed = _draw_ghost_object(
            frame_revealed, H_a, det_mirrored, det.class_label
        )

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10),
                                    facecolor="#0d0d1a")
    fig.suptitle(
        "SkyNet — Collaborative Spatial Awareness: Revealing the Hidden Object",
        color="white", fontsize=14, fontweight="bold", y=1.01,
    )

    ax1.imshow(cv2.cvtColor(frame_a_annotated, cv2.COLOR_BGR2RGB))
    ax1.set_title(
        f"Camera A alone\n"
        f"{len(detected_a)} object(s) detected  |  creatine tub: UNKNOWN",
        color="#3498db", fontsize=11, pad=8,
    )
    ax1.axis("off")

    ax2.imshow(cv2.cvtColor(frame_revealed, cv2.COLOR_BGR2RGB))
    source_cameras = list({o.observed_by for o in hidden_objects})
    ax2.set_title(
        f"Camera A  +  WorldModel knowledge (from {', '.join(source_cameras)})\n"
        f"Hidden object location queried from shared WorldModel → ghosted into Camera A's view",
        color="#00ffcc", fontsize=11, pad=8,
    )
    ax2.axis("off")

    # Legend
    legend_handles = [
        mpatches.Patch(color="#3498db", label="Detected by Camera A"),
        mpatches.Patch(color="#00ffcc", label="Hidden — revealed via Camera B"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               facecolor="#1a1a2e", edgecolor="#555",
               labelcolor="white", fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nSaved → {out_path}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reveal hidden objects in Camera A's view by querying the shared "
            "WorldModel API (which Camera B already ingested into)."
        )
    )
    parser.add_argument("--photo-a", default=None,
                        help="Camera A image file (omit to capture live from camera)")
    parser.add_argument("--config-a", required=True, help="Camera A calibration JSON")
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="WorldModel API base URL")
    parser.add_argument("--out", default="reveal.png")
    args = parser.parse_args()

    run_reveal(args.config_a, args.photo_a, args.api_url, args.out)


if __name__ == "__main__":
    main()
