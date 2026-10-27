import Foundation
import simd

/// Inference adapter for accepted, stabilized MediaPipe landmarks. The app owns
/// pixel preparation, detection/recovery, face guard, and landmark stabilization.
/// See IOS_INTEGRATION.md before connecting this to a production camera stream.
public final class GestureLandmarkEngine {
    private let classifier: GestureONNXClassifier
    private let pipeline = GestureDecisionPipeline()
    private var lastTimestamp: TimeInterval?

    public init(modelsDirectory: URL) throws {
        classifier = try GestureONNXClassifier(modelsDirectory: modelsDirectory)
    }

    public func reset() {
        pipeline.reset()
        lastTimestamp = nil
    }

    public func process(landmarks: [SIMD2<Float>]?, worldLandmarks: [SIMD3<Float>]?,
                        handedness: String?, handednessConfidence: Float?,
                        timestamp: TimeInterval) throws -> GesturePipelineResult {
        guard timestamp.isFinite, lastTimestamp == nil || timestamp > lastTimestamp! else {
            reset()
            throw GestureIntegrationError.invalidContract("Timestamp must increase in seconds")
        }
        lastTimestamp = timestamp
        guard let landmarks else {
            return pipeline.process(probabilities: [], knownGestureMass: 0, landmarks: nil, timestamp: timestamp)
        }
        do {
            let features = try GestureFeatureExtractor.landmarksToFeature(landmarks)
            let model = try classifier.predict(features: features)
            return pipeline.process(probabilities: model.probabilities, knownGestureMass: model.knownGestureMass,
                landmarks: landmarks, handedness: handedness, handednessConfidence: handednessConfidence,
                worldLandmarks: worldLandmarks, timestamp: timestamp)
        } catch {
            _ = pipeline.process(probabilities: [], knownGestureMass: 0, landmarks: nil, timestamp: timestamp)
            throw error
        }
    }
}
