import Foundation
import simd

public struct GestureDiagnosticObservation {
    public let landmarks: [SIMD2<Float>]
    public let worldLandmarks: [SIMD3<Float>]?
    public let handedness: String?
    public let handednessConfidence: Float?
    public let imageWidth: Int
    public let imageHeight: Int
    public let timestamp: TimeInterval
    public let probabilities: [Float]
    public let knownGestureMass: Float
}

/// Raw per-hand diagnostics. The app must port the current reference decision path
/// before dispatching actions. Do not connect the older Swift decision pipeline.
public final class GestureDiagnosticEngine {
    private let classifier: GestureONNXClassifier
    public init(modelsDirectory: URL) throws {
        classifier = try GestureONNXClassifier(modelsDirectory: modelsDirectory)
    }

    public func observe(landmarks: [SIMD2<Float>], worldLandmarks: [SIMD3<Float>]?,
                        handedness: String?, handednessConfidence: Float?,
                        imageWidth: Int, imageHeight: Int,
                        timestamp: TimeInterval) throws -> GestureDiagnosticObservation {
        guard timestamp.isFinite, imageWidth > 0, imageHeight > 0 else {
            throw GestureIntegrationError.invalidContract("Invalid prepared-frame dimensions or clock")
        }
        let output = try classifier.predict(features: GestureFeatureExtractor.landmarksToFeature(landmarks))
        return GestureDiagnosticObservation(landmarks: landmarks, worldLandmarks: worldLandmarks,
            handedness: handedness, handednessConfidence: handednessConfidence,
            imageWidth: imageWidth, imageHeight: imageHeight, timestamp: timestamp,
            probabilities: output.probabilities, knownGestureMass: output.knownGestureMass)
    }
}
