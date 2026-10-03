import Foundation
import simd

public struct GesturePipelineResult {
    public let gesture: String
    public let action: String?
    public let confidence: Float
    public let reason: String
}

/// One instance per camera stream. The caller supplies named ONNX outputs
/// and the matching MediaPipe image landmarks in unmirrored coordinates.
public final class GestureDecisionPipeline {
    public let temporalGate = GestureTemporalGate()
    public let actionLatch = GestureActionLatch()

    public init() {}

    public func reset() {
        temporalGate.reset()
        actionLatch.reset()
    }

    public func process(
        probabilities: [Float],
        knownGestureMass: Float,
        landmarks: [SIMD2<Float>]?,
        handedness: String? = nil,
        handednessConfidence: Float? = nil,
        timestamp: TimeInterval
    ) -> GesturePipelineResult {
        guard let landmarks, landmarks.count == 21 else {
            return rejected("no usable hand", timestamp: timestamp)
        }
        guard probabilities.count == GestureFeatureExtractor.classNames.count,
              probabilities.allSatisfy({ $0.isFinite && $0 >= 0 }),
              knownGestureMass.isFinite,
              (0...1).contains(knownGestureMass) else {
            return rejected("invalid model output", timestamp: timestamp)
        }
        do {
            let resolved = try GestureGeometryResolver.resolve(
                probabilities: probabilities,
                landmarks: landmarks,
                handedness: handedness,
                handednessConfidence: handednessConfidence
            )
            let decision = temporalGate.update(
                resolved,
                knownGestureMass: knownGestureMass,
                timestamp: timestamp
            )
            let action = actionLatch.update(decision, timestamp: timestamp)
            return GesturePipelineResult(
                gesture: decision.execute ? decision.gesture : GestureFeatureExtractor.rejectLabel,
                action: action,
                confidence: decision.execute ? decision.confidence : 0,
                reason: decision.reason
            )
        } catch {
            return rejected("invalid hand geometry", timestamp: timestamp)
        }
    }

    private func rejected(_ reason: String, timestamp: TimeInterval) -> GesturePipelineResult {
        let decision = temporalGate.reject(reason)
        _ = actionLatch.update(decision, timestamp: timestamp)
        return GesturePipelineResult(
            gesture: GestureFeatureExtractor.rejectLabel,
            action: nil,
            confidence: 0,
            reason: reason
        )
    }
}
