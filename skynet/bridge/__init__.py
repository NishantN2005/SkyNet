"""
SkyNet camera bridge — connects real cameras to the WorldModel HTTP API.

Usage:
    # Step 1: calibrate each camera once
    python -m skynet.bridge.calibrate --camera 0 --out calibration_laptop.json \\
        --world-width 1.2 --world-height 0.8

    # Step 2: run a bridge process per camera
    python -m skynet.bridge.camera_bridge --config calibration_laptop.json
"""
