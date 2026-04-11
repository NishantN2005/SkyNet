# SkyNet SLAM iOS App — Setup Guide

## One-time Xcode setup

1. Install Xcode from the Mac App Store (free, ~12 GB)
2. Open Xcode → Create a new project
   - Choose: **App** (under iOS)
   - Product Name: `SkyNetSLAM`
   - Interface: **SwiftUI**
   - Language: **Swift**
   - Uncheck "Include Tests"
3. Replace the auto-generated `ContentView.swift` with the file at
   `ios_app/SkyNetSLAM/ContentView.swift`
4. Open `Info.plist` in Xcode and add a row:
   - Key: `Privacy - Camera Usage Description`
   - Value: `SkyNet SLAM uses the camera for ARKit visual odometry`
   (Or replace the whole plist with `ios_app/SkyNetSLAM/Info.plist`)
5. Plug your iPhone into your Mac with a USB cable
6. In Xcode top bar, select your iPhone as the build target
7. Click the Run ▶ button — Xcode will sign and install the app

> First run: Xcode may ask you to enable Developer Mode on your iPhone
> (Settings → Privacy & Security → Developer Mode) and to trust your
> Apple ID on the device (Settings → General → VPN & Device Management).

---

## Using the app

### Before each session
1. Make sure the SkyNet server is running on your laptop:
   ```
   uvicorn skynet.api:app --port 8000
   ```
2. Note your laptop's local IP address:
   ```
   ipconfig getifaddr en0
   ```
   (Typical result: `192.168.1.xx`)

### In the app
1. Open the app — the AR camera feed starts automatically
2. Set **Server URL** to `http://<your-laptop-ip>:8000`
3. Set **Camera ID** to `camera_a` (or whichever camera you're calibrating)
4. Wait for the status bar to show **Tracking ✓**
5. **Place the phone at the Top-Left corner of the table**, pointing toward the table
6. Tap **Set Origin** — the app will now POST pose updates at 10 Hz
7. Walk the phone to Camera A's mounted position on the table edge
8. The `POST → HTTP 200` status confirms the server is receiving data

### Enabling live pose in camera_bridge
Set `use_slam_pose: true` in your calibration JSON, or edit the file:
```json
{
  "camera_id": "camera_a",
  ...
  "use_slam_pose": true
}
```

Then run camera_bridge as normal:
```
python -m skynet.bridge.camera_bridge --config calibration_a.json
```

The bridge will now fetch the live ARKit pose from `GET /slam_pose/camera_a`
each frame and use it as `robot_pose` in the ObservationReport.

---

## Coordinate alignment

ARKit initialises its world frame arbitrarily at launch. The app handles
alignment by recording the ARKit transform at the moment you tap **Set Origin**,
then expressing all subsequent poses *relative to that origin*.

Expected workflow:
- Origin = TL corner of table → (0, 0) in world space
- Camera mounted at the left edge, midway → roughly (0, world_h/2)
- ARKit X maps to table X (right), ARKit -Z maps to table Y (forward)

If the ghost objects appear in wrong locations, re-tap Set Origin while the
phone is precisely at the TL corner.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `POST → ERR: connection refused` | Server not running, or wrong IP |
| `POST → HTTP 404` | `/slam_pose` endpoint not in api.py — update server |
| Status stuck at "Initialising…" | Move phone around slowly to build feature map |
| Ghost at wrong table position | Re-tap Set Origin more precisely at TL corner |
| App won't install | Enable Developer Mode on iPhone, trust Apple ID |
