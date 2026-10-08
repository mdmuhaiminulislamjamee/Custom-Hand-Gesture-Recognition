import Foundation
import MediaPipeTasksVision

/// First-light detector wiring. For desktop parity the app must add the recovery,
/// face guard and landmark stabilization described in IOS_INTEGRATION.md.
public final class GestureHandLandmarker {
    private let detector: HandLandmarker
    private var lastTimestampMS: Int = -1

    public init(modelURL: URL) throws {
        let options = HandLandmarkerOptions()
        options.baseOptions.modelAssetPath = modelURL.path
        options.runningMode = .video
        options.numHands = 1
        options.minHandDetectionConfidence = 0.55
        options.minHandPresenceConfidence = 0.55
        options.minTrackingConfidence = 0.60
        detector = try HandLandmarker(options: options)
    }

    /// Call on a serial background queue with correctly oriented, unmirrored pixels.
    /// Use the SAME frame clock for the detector (milliseconds) and engine (seconds).
    public func detect(image: MPImage, timestampMS: Int) throws -> HandLandmarkerResult {
        guard timestampMS > lastTimestampMS else {
            throw GestureIntegrationError.invalidContract("Detector timestamps must increase")
        }
        lastTimestampMS = timestampMS
        return try detector.detect(videoFrame: image, timestampInMilliseconds: timestampMS)
    }
}
