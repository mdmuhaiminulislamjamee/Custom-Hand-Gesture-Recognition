import Foundation
import MediaPipeTasksVision
import simd

public struct GestureDetectedHand {
    public let landmarks: [SIMD2<Float>]
    public let worldLandmarks: [SIMD3<Float>]?
    public let handedness: String?
    public let handednessConfidence: Float?
}
public struct GestureFilteredHandResult {
    public let hands: [GestureDetectedHand]
    public let rejections: [Int: String]
}

/// Diagnostic detector wiring with false-hand filtering before inference/preview.
/// Desktop parity additionally needs recovery, selection and stabilization.
public final class GestureHandLandmarker {
    private let detector: HandLandmarker
    private var lastTimestampMS: Int = -1
    private let faceGuard: GestureFaceGuard

    public init(modelURL: URL) throws {
        faceGuard = try GestureFaceGuard(modelURL: modelURL.deletingLastPathComponent().appendingPathComponent("blaze_face_short_range.tflite"))
        let options = HandLandmarkerOptions()
        options.baseOptions.modelAssetPath = modelURL.path
        options.runningMode = .video
        options.numHands = 2
        options.minHandDetectionConfidence = 0.55
        options.minHandPresenceConfidence = 0.55
        options.minTrackingConfidence = 0.60
        detector = try HandLandmarker(options: options)
    }

    /// Call on a serial background queue with a decoded NCM JPEG that has been
    /// prepared as correctly oriented, unmirrored pixels.
    /// Use the SAME frame clock for the detector (milliseconds) and engine (seconds).
    public func detect(image: MPImage, timestampMS: Int) throws -> GestureFilteredHandResult {
        guard timestampMS > lastTimestampMS else {
            throw GestureIntegrationError.invalidContract("Detector timestamps must increase")
        }
        lastTimestampMS = timestampMS
        let faces = try faceGuard.faces(image: image, timestampMS: timestampMS)
        let result = try detector.detect(videoFrame: image, timestampInMilliseconds: timestampMS)
        var hands: [GestureDetectedHand] = [], rejected: [Int: String] = [:]
        for (index, points) in result.landmarks.enumerated() {
            let xy = points.map { SIMD2<Float>(Float($0.x), Float($0.y)) }
            let world = index < result.worldLandmarks.count ? result.worldLandmarks[index].map { SIMD3<Float>(Float($0.x), Float($0.y), Float($0.z)) } : nil
            if let reason = GestureFalseHandPolicy.rejection(landmarks: xy, world: world, width: image.width, height: image.height, faces: faces) {
                rejected[index] = reason
                continue
            }
            let category = index < result.handedness.count ? result.handedness[index].first : nil
            hands.append(GestureDetectedHand(landmarks: xy, worldLandmarks: world,
                handedness: category?.categoryName, handednessConfidence: category.map { Float($0.score) }))
        }
        return GestureFilteredHandResult(hands: hands, rejections: rejected)
    }
}
