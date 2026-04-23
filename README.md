# Canopy — Collaborative Spatial Awareness for Multi-Robot Fleets

Canopy is a proof-of-concept system that gives a fleet of robots (or cameras) **shared perception** — each agent contributes what it sees to a central world model, so the collective knows things no single agent could know alone.

The core problem it solves: **blind spots**. Any single camera or robot has a limited field of view. Canopy fuses observations from multiple agents into one continuously updated world model, so when Agent A loses sight of a person, Agent B's observation keeps the track alive. In the visualizer, objects that one camera knows about but another cannot see are rendered as **ghosts** — visible to the fleet, invisible to the individual.

---

## Motivation

Modern robot fleets — warehouse AMRs, autonomous vehicles, drone swarms — operate in environments where no single agent has full situational awareness. Today, each robot typically reasons about the world from its own sensor data alone. Canopy explores what it looks like when that constraint is lifted: what if every robot in a fleet could query the collective perception of the entire fleet in real time?

This is relevant to:
- **Multi-robot coordination** — robots can avoid redundant work and navigate through spaces they cannot directly observe
- **Persistent object tracking** — targets that pass through coverage gaps are not lost
- **Sensor fusion research** — combining noisy, calibrated observations from heterogeneous cameras into a coherent spatial model

---

## How It Works

```
┌─────────────────────────────────────────┐
│         WorldModel  (FastAPI)           │
│   Fused object registry + occupancy     │
│   grid, updated in real time            │
└──────────┬──────────────────────────────┘
           │  POST /ingest
   ┌────────┴────────┐
   │                 │
Camera 0          Camera 2
(YOLOv8 detects   (YOLOv8 detects
 people)           people)
   │                 │
foot pixel → homography → world coords (metres)
```

Each camera runs **YOLOv8** to detect people in its video feed. Detections are projected from pixel space to real-world coordinates using a **ground-plane homography** calibrated for the camera. The world-coordinate position is posted to the shared `WorldModel` over HTTP.

The WorldModel:
- Fuses detections from all cameras into a single object registry
- Applies **proximity deduplication** — if two cameras report a person within 0.5 m of each other, they are merged into one track
- Smooths positions using a **Kalman filter** (constant-velocity model) to eliminate frame-to-frame jitter
- Ages out objects not seen in the last 2 seconds
- Exposes a REST API (`/ingest`, `/world`, `/query`, `/robots`) for reading and writing state

---

## Current Capabilities

- **Multi-camera ingestion** — simultaneous YOLO inference on N camera feeds, each posting to the shared world model
- **Homography-based localization** — pixel detections projected to metric world coordinates via pre-calibrated ground-plane homographies
- **Cross-camera deduplication** — same person seen by two cameras merges into one world-model entry
- **Kalman filter smoothing** — position tracks are smooth even with noisy detections or missed frames
- **Ghost object awareness** — objects seen by Camera B but not Camera A are flagged and rendered distinctly
- **Live 3D visualization** — [Rerun](https://rerun.io) viewer shows the world model in 3D: camera positions, person tracks, ghost objects, and status log updating in real time
- **Synchronized camera feeds** — 2D panels in the visualizer show the raw video frames synchronized to the current ingestion frame
- **Confidence decay** — objects age out gracefully if no camera re-observes them

### Demo Dataset

The current demo runs on the [EPFL POM multi-camera dataset](https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/) — a 4-camera indoor lab sequence with up to 6 people walking around simultaneously. Two cameras (0 and 2) are used, placed at opposite corners of the room to maximize coverage asymmetry.

---

## Architecture

```
canopy/
├── models.py        — Pydantic v2 data contracts (Pose2D, DetectedObject,
│                      ObservationReport, WorldState, ...)
├── fusion.py        — ObservationFuser (merge logic) + KalmanTracker
├── world_model.py   — WorldModel: thread-safe state store, ingest pipeline
├── queries.py       — QueryEngine: spatial/temporal filtering on world state
├── api.py           — FastAPI HTTP wrapper (POST /ingest, GET /world, ...)
└── bridge/
    ├── pom_ingest.py   — EPFL POM dataset ingestion + YOLO inference loop
    └── visualizer.py   — Rerun 3D live visualizer
```

---

## Running the Demo

### Requirements

```bash
pip install -r requirements.txt
```

Dependencies: `fastapi`, `uvicorn`, `pydantic>=2`, `numpy`, `opencv-python`, `httpx`, `ultralytics`, `rerun-sdk`

### Dataset

Download the EPFL POM lab sequence into `datasets/pom_lab/`:
- `6p-c0.avi`, `6p-c2.avi` — camera feeds
- `calibration-6p.txt` — ground-plane homographies
- `gt_lab_6p.txt` — ground truth (optional)

Available at: https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/

### Run

```bash
# Terminal 1 — start the WorldModel server
uvicorn canopy.api:app --host 0.0.0.0 --port 8000

# Terminal 2 — start the 3D visualizer (opens Rerun viewer)
python -m canopy.bridge.visualizer \
    --world-width 6.0 --world-height 5.0 \
    --primary-camera camera_0 \
    --cameras camera_0 camera_2 \
    --videos datasets/pom_lab/6p-c0.avi datasets/pom_lab/6p-c2.avi \
    --refresh-hz 30

# Terminal 3 — run ingestion (YOLO inference + world model updates)
python -m canopy.bridge.pom_ingest \
    --dataset-dir datasets/pom_lab \
    --cameras 0 2 \
    --fps 5 \
    --start-frame 400 \
    --conf 0.25 \
    --yolo-model yolov8s.pt
```

---

## What You See in the Visualizer

| Element | Meaning |
|---|---|
| Blue box | Person detected by Camera 0 |
| Green box | Person detected by Camera 2 |
| Cyan box (ghost) | Person known to the fleet but not seen by Camera 0 |
| Camera dots + arrows | Physical camera positions, pointing toward room center |
| Status log | Live count of total objects, direct detections, and ghosts |

---

## Limitations and Future Work

- **Homography calibration** assumes a flat ground plane and a static camera. Moving cameras require a different localization approach (e.g. ARKit/SLAM for pose + depth for projection)
- **No persistent identity** — object IDs are per-session; tracks do not survive a server restart or re-entry into the scene after aging out
- **Centralized model** — the WorldModel is a single server. A production system would use a distributed consensus mechanism
- **No 3D depth** — without a depth sensor, localization relies entirely on foot-position homography, which breaks if people are on elevated surfaces

### Roadmap
- Live camera support (laptop webcam + iPhone via Continuity Camera)
- Probabilistic fusion (replace last-write-wins with weighted average)
- Persistent track IDs across re-entry using appearance embeddings
- Distributed WorldModel over gRPC

---

## Related Work

- **PETS 2009** / **DukeMTMC** — standard multi-camera pedestrian tracking benchmarks
- **EKF/UKF SLAM** — extended Kalman filter approaches to simultaneous localization and mapping
- **DAIR-V2X** — vehicle-to-infrastructure perception sharing for autonomous driving
- **OpenVINS** — open-source visual-inertial navigation used in real robot deployments

---

*Built as a proof-of-concept. Not production software.*
