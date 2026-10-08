import Foundation

/// Emits one mapped action for a stable gesture edge, then suppresses repeats
/// until release. This mirrors the desktop action cooldown/rearm behavior.
public final class GestureActionLatch {
    public var cooldownSeconds: TimeInterval = 0.80
    public var releaseFramesRequired = 3
    public var targetFPS: Double = 10

    private var lastActionGesture: String?
    private var lastActionAt = -TimeInterval.infinity
    private var releaseFrames = 0

    public init() {}

    public func reset() {
        lastActionGesture = nil
        lastActionAt = -TimeInterval.infinity
        releaseFrames = 0
    }

    /// Returns the action string once when it should be dispatched.
    public func update(
        _ decision: GestureTemporalDecision,
        timestamp: TimeInterval,
        trackingGap: Bool = false
    ) -> String? {
        if decision.execute {
            let newEdge = decision.gesture != lastActionGesture
            let cooldownReady = timestamp - lastActionAt >= cooldownSeconds
            if newEdge && cooldownReady {
                lastActionGesture = decision.gesture
                lastActionAt = timestamp
                releaseFrames = 0
                return GestureFeatureExtractor.gestureToAction[decision.gesture]
            }
            observeRelease(currentGesture: decision.gesture)
            return nil
        }

        let current = decision.gesture == GestureFeatureExtractor.rejectLabel
            ? nil
            : decision.gesture
        observeRelease(currentGesture: current, trackingGap: trackingGap)
        return nil
    }

    private func observeRelease(currentGesture: String?, trackingGap: Bool = false) {
        guard let lastActionGesture else {
            releaseFrames = 0
            return
        }
        if currentGesture == lastActionGesture {
            releaseFrames = 0
            return
        }
        releaseFrames += 1
        let required = trackingGap ? max(releaseFramesRequired, Int(ceil(targetFPS * 0.6))) : releaseFramesRequired
        if releaseFrames >= max(1, required) {
            self.lastActionGesture = nil
            releaseFrames = 0
        }
    }
}
