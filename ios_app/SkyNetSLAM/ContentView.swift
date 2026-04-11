import SwiftUI
import ARKit
import SceneKit
import Combine

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

@main
struct SkyNetSLAMApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

// ---------------------------------------------------------------------------
// View model — holds ARKit state and networking
// ---------------------------------------------------------------------------

@MainActor
class SLAMViewModel: NSObject, ObservableObject, ARSessionDelegate {

    // UI state
    @Published var trackingStatus: String = "Not started"
    @Published var originSet: Bool = false
    @Published var poseText: String = "—"
    @Published var postStatus: String = "—"
    @Published var serverURL: String = "http://192.168.1.100:8000"  // edit in app
    @Published var cameraID: String = "camera_a"

    // ARKit
    let session = ARSession()
    private var originTransform: simd_float4x4? = nil
    private var latestFrame: ARFrame? = nil

    // Networking
    private var postTimer: Timer?

    override init() {
        super.init()
        session.delegate = self
    }

    // MARK: - ARKit lifecycle

    func startSession() {
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity   // Y-up, gravity-aligned
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
        trackingStatus = "Initialising ARKit…"
    }

    func setOrigin() {
        guard let frame = latestFrame else { return }
        originTransform = frame.camera.transform
        originSet = true
        trackingStatus = "Origin set ✓  — streaming pose"
        startPosting()
    }

    func resetOrigin() {
        originTransform = nil
        originSet = false
        stopPosting()
        trackingStatus = "Origin cleared — tap Set Origin to re-calibrate"
    }

    // MARK: - ARSessionDelegate

    nonisolated func session(_ session: ARSession, didUpdate frame: ARFrame) {
        Task { @MainActor in
            self.latestFrame = frame
            self.updateTrackingStatus(frame)
        }
    }

    private func updateTrackingStatus(_ frame: ARFrame) {
        switch frame.camera.trackingState {
        case .normal:
            if !originSet { trackingStatus = "Tracking ✓  — tap Set Origin" }
        case .notAvailable:
            trackingStatus = "Tracking unavailable"
        case .limited(let reason):
            switch reason {
            case .initializing:  trackingStatus = "Initialising…"
            case .excessiveMotion: trackingStatus = "Move slower"
            case .insufficientFeatures: trackingStatus = "Need more texture/light"
            case .relocalizing: trackingStatus = "Relocalising…"
            @unknown default: trackingStatus = "Limited tracking"
            }
        }
    }

    // MARK: - Pose extraction

    /// Returns (x_m, y_m, heading_rad) in origin-relative table coords,
    /// or nil if origin not set or tracking is poor.
    private func relativePose(from frame: ARFrame) -> (Float, Float, Float, Float)? {
        guard let origin = originTransform else { return nil }

        // World-to-origin transform
        let originInv = simd_inverse(origin)
        let relT = originInv * frame.camera.transform

        // Position: ARKit is Y-up, Z-into-screen at origin.
        // We project onto the XZ plane (floor) → table X/Y.
        let x = relT.columns.3.x   // metres right of origin
        let y = -relT.columns.3.z  // metres forward of origin (flip Z→Y)

        // Heading: angle of camera forward vector projected onto floor
        let fwd = relT.columns.2   // camera -Z in origin frame
        let heading = atan2(-fwd.x, -fwd.z)

        // Confidence
        let conf: Float
        switch frame.camera.trackingState {
        case .normal: conf = 1.0
        case .limited: conf = 0.4
        default: conf = 0.0
        }

        return (x, y, heading, conf)
    }

    // MARK: - HTTP posting

    private func startPosting() {
        stopPosting()
        postTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.postPose()
            }
        }
    }

    private func stopPosting() {
        postTimer?.invalidate()
        postTimer = nil
    }

    private func postPose() {
        guard let frame = latestFrame,
              let (x, y, heading, conf) = relativePose(from: frame) else { return }

        poseText = String(format: "x=%.3fm  y=%.3fm  hdg=%.1f°  conf=%.2f",
                          x, y, heading * 180 / .pi, conf)

        guard let url = URL(string: "\(serverURL)/slam_pose") else { return }

        let body: [String: Any] = [
            "camera_id": cameraID,
            "x": x,
            "y": y,
            "heading_rad": heading,
            "confidence": conf,
            "timestamp": Date().timeIntervalSince1970,
        ]

        guard let data = try? JSONSerialization.data(withJSONObject: body) else { return }

        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = data
        req.timeoutInterval = 2.0

        URLSession.shared.dataTask(with: req) { [weak self] _, resp, err in
            Task { @MainActor [weak self] in
                if let err = err {
                    self?.postStatus = "ERR: \(err.localizedDescription)"
                } else if let http = resp as? HTTPURLResponse {
                    self?.postStatus = "HTTP \(http.statusCode)  @ \(Self.shortTime())"
                }
            }
        }.resume()
    }

    private static func shortTime() -> String {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f.string(from: Date())
    }
}

// ---------------------------------------------------------------------------
// AR SceneKit view wrapper
// ---------------------------------------------------------------------------

struct ARViewContainer: UIViewRepresentable {
    let session: ARSession

    func makeUIView(context: Context) -> ARSCNView {
        let v = ARSCNView(frame: .zero)
        v.session = session
        v.automaticallyUpdatesLighting = true
        v.showsStatistics = false
        return v
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}

// ---------------------------------------------------------------------------
// SwiftUI content view
// ---------------------------------------------------------------------------

struct ContentView: View {
    @StateObject private var vm = SLAMViewModel()

    var body: some View {
        ZStack(alignment: .bottom) {
            // AR camera feed fills background
            ARViewContainer(session: vm.session)
                .ignoresSafeArea()

            // HUD overlay
            VStack(spacing: 0) {
                // Status bar at top
                HStack {
                    Image(systemName: vm.originSet ? "location.fill" : "location.slash")
                        .foregroundColor(vm.originSet ? .green : .yellow)
                    Text(vm.trackingStatus)
                        .font(.system(size: 13, weight: .medium, design: .monospaced))
                        .foregroundColor(.white)
                    Spacer()
                }
                .padding(10)
                .background(Color.black.opacity(0.6))

                Spacer()

                // Controls panel
                VStack(spacing: 12) {
                    // Server config
                    HStack(spacing: 8) {
                        Image(systemName: "network")
                            .foregroundColor(.cyan)
                        TextField("Server URL", text: $vm.serverURL)
                            .font(.system(size: 13, design: .monospaced))
                            .foregroundColor(.white)
                            .autocapitalization(.none)
                            .disableAutocorrection(true)
                            .keyboardType(.URL)
                    }
                    .padding(8)
                    .background(Color.white.opacity(0.12))
                    .cornerRadius(8)

                    HStack(spacing: 8) {
                        Image(systemName: "camera")
                            .foregroundColor(.cyan)
                        TextField("Camera ID", text: $vm.cameraID)
                            .font(.system(size: 13, design: .monospaced))
                            .foregroundColor(.white)
                            .autocapitalization(.none)
                            .disableAutocorrection(true)
                    }
                    .padding(8)
                    .background(Color.white.opacity(0.12))
                    .cornerRadius(8)

                    // Pose readout
                    Text(vm.poseText)
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundColor(.green)
                        .frame(maxWidth: .infinity, alignment: .leading)

                    Text("POST → \(vm.postStatus)")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(.gray)
                        .frame(maxWidth: .infinity, alignment: .leading)

                    // Action buttons
                    HStack(spacing: 12) {
                        Button(action: { vm.setOrigin() }) {
                            Label("Set Origin", systemImage: "scope")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(vm.originSet ? .green : .blue)
                        .disabled(!vm.trackingStatus.contains("✓") && !vm.originSet)

                        Button(action: { vm.resetOrigin() }) {
                            Label("Reset", systemImage: "arrow.counterclockwise")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .tint(.red)
                    }

                    Text("Place phone at table TL corner, point camera at table, then tap Set Origin.")
                        .font(.caption2)
                        .foregroundColor(.gray)
                        .multilineTextAlignment(.center)
                }
                .padding(16)
                .background(.ultraThinMaterial)
                .cornerRadius(16, corners: [.topLeft, .topRight])
            }
        }
        .onAppear { vm.startSession() }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

extension View {
    func cornerRadius(_ radius: CGFloat, corners: UIRectCorner) -> some View {
        clipShape(RoundedCorner(radius: radius, corners: corners))
    }
}

struct RoundedCorner: Shape {
    var radius: CGFloat
    var corners: UIRectCorner
    func path(in rect: CGRect) -> Path {
        let path = UIBezierPath(
            roundedRect: rect,
            byRoundingCorners: corners,
            cornerRadii: CGSize(width: radius, height: radius)
        )
        return Path(path.cgPath)
    }
}
