"""
Camera calibration tool — computes the perspective homography that maps
pixel coordinates to real-world table coordinates (meters).

Usage:
    python -m skynet.bridge.calibrate \\
        --camera 0 \\
        --out calibration_laptop.json \\
        --world-width 1.2 \\
        --world-height 0.8 \\
        --camera-id laptop

Before running:
    Place tape at the 4 corners of a rectangle on your table.
    Measure its real-world dimensions (width × height in metres).
    The corners must be clicked in this order: TL → TR → BR → BL.

    If your tape marks the full table surface, pass --world-width / --world-height
    matching the table. The origin (0, 0) will be at the Top-Left corner.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from skynet.bridge.config import CameraConfig, load_config, save_config

WINDOW = "SkyNet Calibration"
CORNER_LABELS = ["TL", "TR", "BR", "BL"]
CORNER_COLORS = [
    (0, 255, 0),    # TL — green
    (255, 0, 0),    # TR — blue
    (0, 0, 255),    # BR — red
    (0, 255, 255),  # BL — yellow
]


# ---------------------------------------------------------------------------
# Calibration state (shared with the mouse callback)
# ---------------------------------------------------------------------------

class _CalibState:
    def __init__(self) -> None:
        self.clicks: list[tuple[int, int]] = []
        self.frame: np.ndarray | None = None


def _mouse_cb(event, x, y, flags, state: _CalibState) -> None:
    if event == cv2.EVENT_LBUTTONDOWN and len(state.clicks) < 4:
        state.clicks.append((x, y))
        label = CORNER_LABELS[len(state.clicks) - 1]
        print(f"  Clicked {label}: pixel ({x}, {y})")


# ---------------------------------------------------------------------------
# Homography helpers
# ---------------------------------------------------------------------------

def _compute_homography(
    pixel_pts: list[tuple[int, int]],
    world_width_m: float,
    world_height_m: float,
) -> np.ndarray:
    """
    Compute a 3×3 perspective homography from 4 pixel clicks to world coords.

    World coordinate system:
        TL = (0, 0)
        TR = (world_width_m, 0)
        BR = (world_width_m, world_height_m)
        BL = (0, world_height_m)
    """
    W, H = world_width_m, world_height_m
    world_pts = np.array(
        [[0, 0], [W, 0], [W, H], [0, H]],
        dtype=np.float32,
    )
    src = np.array(pixel_pts, dtype=np.float32)
    M, _ = cv2.findHomography(src, world_pts)
    return M


def apply_homography(H: np.ndarray, px: float, py: float) -> tuple[float, float]:
    """Map a single pixel point through homography H → (world_x, world_y)."""
    pt = np.array([[[px, py]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H)
    wx, wy = result[0, 0]
    return float(wx), float(wy)


# ---------------------------------------------------------------------------
# Verification overlay
# ---------------------------------------------------------------------------

def _draw_verification(
    frame: np.ndarray,
    pixel_pts: list[tuple[int, int]],
    H: np.ndarray,
    world_width_m: float,
    world_height_m: float,
) -> np.ndarray:
    """
    Draw calibration overlay on a copy of frame:
      - Coloured circles at the 4 clicked corners
      - Re-projected grid lines in world space back into pixel space
    """
    vis = frame.copy()
    W, H_m = world_width_m, world_height_m

    # Draw clicked corners
    for (x, y), label, color in zip(pixel_pts, CORNER_LABELS, CORNER_COLORS):
        cv2.circle(vis, (x, y), 8, color, -1)
        cv2.putText(vis, label, (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Re-project a grid (5×5) back into pixel space as verification
    H_inv = np.linalg.inv(H)
    n = 5
    for i in range(n + 1):
        for j in range(n + 1):
            wx = W * i / n
            wy = H_m * j / n
            pt = np.array([[[wx, wy]]], dtype=np.float32)
            px_pt = cv2.perspectiveTransform(pt, H_inv)
            px, py = int(px_pt[0, 0, 0]), int(px_pt[0, 0, 1])
            cv2.circle(vis, (px, py), 2, (200, 200, 200), -1)

    # Draw border of calibrated region
    border_world = np.array([[[0, 0], [W, 0], [W, H_m], [0, H_m]]], dtype=np.float32)
    border_px = cv2.perspectiveTransform(border_world, H_inv)
    pts = border_px.astype(np.int32)
    cv2.polylines(vis, pts, isClosed=True, color=(0, 255, 255), thickness=2)

    cv2.putText(vis, "Calibration verified — press any key to save",
                (10, vis.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return vis


# ---------------------------------------------------------------------------
# Main calibration flow
# ---------------------------------------------------------------------------

def run_calibration(
    camera_source: int | str,
    out_path: str,
    camera_id: str,
    world_width_m: float,
    world_height_m: float,
    existing_config: CameraConfig | None = None,
) -> None:
    # Detect static image files — handle differently from live cameras/video
    _image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    _is_static = (
        isinstance(camera_source, str)
        and any(camera_source.lower().endswith(ext) for ext in _image_exts)
    )

    if _is_static:
        static_frame = cv2.imread(camera_source)
        if static_frame is None:
            print(f"ERROR: Cannot read image: {camera_source!r}", file=sys.stderr)
            sys.exit(1)
        cap = None
    else:
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            print(f"ERROR: Cannot open camera source: {camera_source!r}", file=sys.stderr)
            sys.exit(1)
        static_frame = None

    state = _CalibState()
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, _mouse_cb, state)

    print()
    print("=" * 60)
    print("  SkyNet Camera Calibration")
    print("=" * 60)
    print()
    print("  1. Place tape at the 4 corners of a rectangle on your table.")
    print(f"  2. Measure its size: {world_width_m:.2f} m × {world_height_m:.2f} m")
    print("  3. Click the corners in order: TL → TR → BR → BL")
    print("  4. Press 'r' to reset, 'q' to quit without saving.")
    print()

    while True:
        if _is_static:
            frame = static_frame.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame.", file=sys.stderr)
                break

        state.frame = frame.copy()
        display = frame.copy()

        # Draw already-clicked corners
        for i, (x, y) in enumerate(state.clicks):
            cv2.circle(display, (x, y), 8, CORNER_COLORS[i], -1)
            cv2.putText(display, CORNER_LABELS[i], (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, CORNER_COLORS[i], 2)

        # Instruction overlay
        remaining = 4 - len(state.clicks)
        if remaining > 0:
            next_corner = CORNER_LABELS[len(state.clicks)]
            msg = f"Click corner {len(state.clicks) + 1}/4: {next_corner}  |  r=reset  q=quit"
        else:
            msg = "All 4 corners set — computing homography..."
        cv2.putText(display, msg, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            print("Aborted — no config saved.")
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            return

        if key == ord('r'):
            state.clicks.clear()
            print("  Reset — click the 4 corners again.")
            continue

        if len(state.clicks) == 4:
            break

    if cap is not None:
        cap.release()

    print()
    print("  Computing homography...")
    H = _compute_homography(state.clicks, world_width_m, world_height_m)
    print("  Done.")

    # Verify round-trip accuracy
    print()
    print("  Verification (pixel → world → expected):")
    world_corners = [(0, 0), (world_width_m, 0),
                     (world_width_m, world_height_m), (0, world_height_m)]
    for (px, py), (ex, ey), label in zip(state.clicks, world_corners, CORNER_LABELS):
        wx, wy = apply_homography(H, px, py)
        err = ((wx - ex) ** 2 + (wy - ey) ** 2) ** 0.5
        print(f"    {label}: pixel({px},{py}) → world({wx:.3f},{wy:.3f})  "
              f"expected({ex:.2f},{ey:.2f})  err={err*100:.1f} cm")

    # Show verification overlay
    vis = _draw_verification(state.frame, state.clicks, H, world_width_m, world_height_m)
    cv2.imshow(WINDOW, vis)
    print()
    print("  Showing verification overlay — press any key to save.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Build and save config
    cfg = existing_config or CameraConfig(
        camera_id=camera_id,
        camera_source=camera_source,
        world_width_m=world_width_m,
        world_height_m=world_height_m,
    )
    cfg.camera_id = camera_id
    cfg.camera_source = camera_source
    cfg.world_width_m = world_width_m
    cfg.world_height_m = world_height_m
    cfg.homography = H.tolist()

    save_config(cfg, out_path)
    print(f"\nCalibration complete. Config saved to: {out_path}")
    print("\nNext step:")
    print(f"  python -m skynet.bridge.camera_bridge --config {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate a camera for the SkyNet bridge (homography).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--camera", default=0,
        help="Camera index (0, 1, 2…) or URL (e.g. http://phone-ip:8080/video)",
    )
    parser.add_argument("--out", required=True, help="Output JSON config path")
    parser.add_argument("--camera-id", default="camera", help="Friendly camera name")
    parser.add_argument("--world-width", type=float, default=1.2,
                        help="Table width in metres")
    parser.add_argument("--world-height", type=float, default=0.8,
                        help="Table height in metres")
    args = parser.parse_args()

    # camera source: try int first, fall back to string URL
    try:
        source = int(args.camera)
    except ValueError:
        source = args.camera

    # Load existing config if the output file already exists (update homography only)
    existing = None
    if Path(args.out).exists():
        try:
            existing = load_config(args.out)
            print(f"Updating existing config: {args.out}")
        except Exception:
            pass

    run_calibration(
        camera_source=source,
        out_path=args.out,
        camera_id=args.camera_id,
        world_width_m=args.world_width,
        world_height_m=args.world_height,
        existing_config=existing,
    )


if __name__ == "__main__":
    main()
