"""
FastAPI HTTP wrapper around the WorldModel.

Intended for Phase 2 (HTTP demo) where robots are separate processes posting
observations over the network. For the in-process POC demo, just import
WorldModel directly and skip this module.

Run with:
    uvicorn skynet.api:app --reload --port 8000

Robots then POST to http://localhost:8000/ingest and GET /query.
"""

from __future__ import annotations

import time
import threading
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from skynet.models import (
    ObservationReport,
    QueryRequest,
    QueryResult,
    RobotState,
    WorldState,
)
from skynet.queries import QueryEngine
from skynet.world_model import WorldModel

app = FastAPI(
    title="SkyNet WorldModel API",
    description="Collaborative spatial awareness — shared world model for multi-robot fleets.",
    version="0.1.0",
)

# Module-level singletons. For testing, replace via dependency injection
# (override app.dependency_overrides or monkeypatch _world_model).
_world_model = WorldModel()
_query_engine = QueryEngine(_world_model)

# In-memory store for live SLAM poses keyed by camera_id
_slam_poses: dict[str, dict] = {}
_slam_poses_lock = threading.Lock()


class SlamPoseReport(BaseModel):
    camera_id: str
    x: float          # world X in metres (relative to Set Origin point)
    y: float          # world Y in metres
    heading_rad: float
    confidence: float = 1.0   # ARKit tracking confidence 0-1
    timestamp: float = 0.0    # Unix epoch; 0 = server fills it in


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def get_world_model() -> WorldModel:
    """Dependency hook — makes the WorldModel swappable in tests."""
    return _world_model


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}


@app.post("/ingest", tags=["write"], status_code=200)
def ingest(report: ObservationReport) -> dict:
    """
    Ingest an observation report from a robot.

    The robot publishes its current pose, all detected objects, and all
    free-space cells within its FOV. The WorldModel fuses this into its
    shared state immediately.
    """
    _world_model.ingest(report)
    return {
        "status": "ok",
        "robot_id": report.robot_id,
        "objects_ingested": len(report.detected_objects),
        "free_cells_ingested": len(report.free_cells),
    }


@app.get("/query", response_model=QueryResult, tags=["read"])
def query_world(
    robot_id: str,
    object_class: Optional[str] = None,
    max_age_seconds: float = 60.0,
    min_confidence: float = 0.1,
    x_min: Optional[float] = None,
    y_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_max: Optional[float] = None,
) -> QueryResult:
    """
    Query the fused world model for objects and grid cells.

    Returns filtered detections contributed by all robots in the fleet,
    not just the querying robot's own observations.
    """
    bbox = None
    if any(v is not None for v in [x_min, y_min, x_max, y_max]):
        if not all(v is not None for v in [x_min, y_min, x_max, y_max]):
            raise HTTPException(
                status_code=422,
                detail="Provide all four bbox parameters (x_min, y_min, x_max, y_max) or none.",
            )
        bbox = (x_min, y_min, x_max, y_max)

    request = QueryRequest(
        robot_id=robot_id,
        bbox=bbox,
        object_class=object_class,
        max_age_seconds=max_age_seconds,
        min_confidence=min_confidence,
    )
    return _query_engine.query(request)


@app.get("/world", response_model=WorldState, tags=["read"])
def get_world() -> WorldState:
    """Return a complete snapshot of the fused world state."""
    return _world_model.get_world_state()


@app.get("/robots", response_model=list[RobotState], tags=["read"])
def get_robots() -> list[RobotState]:
    """Return the last known state of all active robots."""
    return list(_world_model.get_robot_states().values())


# ---------------------------------------------------------------------------
# SLAM pose endpoints (used by Path A: ARKit iPhone app)
# ---------------------------------------------------------------------------


@app.post("/slam_pose", tags=["slam"], status_code=200)
def post_slam_pose(report: SlamPoseReport) -> dict:
    """
    Receive a live ARKit pose from the iPhone app.

    The app sends this ~10 Hz after the user taps 'Set Origin'.
    Coordinates are in the table's world frame (metres), with the origin
    at the TL corner of the calibrated surface.
    """
    with _slam_poses_lock:
        _slam_poses[report.camera_id] = {
            "camera_id": report.camera_id,
            "x": report.x,
            "y": report.y,
            "heading_rad": report.heading_rad,
            "confidence": report.confidence,
            "timestamp": report.timestamp if report.timestamp > 0 else time.time(),
        }
    return {"status": "ok", "camera_id": report.camera_id}


@app.get("/slam_pose/{camera_id}", tags=["slam"])
def get_slam_pose(camera_id: str) -> dict:
    """
    Return the most recent SLAM pose for a given camera_id.

    Used by camera_bridge.py to get a live pose instead of the static
    camera_pose from the calibration JSON.
    """
    with _slam_poses_lock:
        pose = _slam_poses.get(camera_id)
    if pose is None:
        raise HTTPException(
            status_code=404,
            detail=f"No SLAM pose received yet for camera_id={camera_id!r}. "
                   "Is the iPhone app running and origin set?",
        )
    return pose
