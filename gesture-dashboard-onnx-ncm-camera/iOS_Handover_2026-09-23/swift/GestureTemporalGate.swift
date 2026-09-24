import Foundation

public struct GestureTemporalDecision {
    public let execute: Bool
    public let gesture: String
    public let confidence: Float
    public let stableFrames: Int
    public let heldSeconds: TimeInterval
    public let reason: String
}

/// Stateful probability EMA and hold gate matching `reference_python/runtime.py`.
/// Create one instance per camera session and call `reset()` after tracking loss.
public final class GestureTemporalGate {
    public var emaAlpha: Float = 0.45
    public var confidenceFloor: Float = 0.70
    public var probabilityMarginFloor: Float = 0.08
    public var knownMassFloor: Float = 0.70
    public var geometryRecoveryMassFloor: Float = 0.15
    public var stableFramesRequired: Int = 3
    public var minimumHoldSeconds: TimeInterval = 0.18
    public var maximumHistory = 30

    private var history: [[Float]] = []
    private var labelStartedAt: TimeInterval?
    private var lastLabel: String?
    private var lastUpdateAt: TimeInterval?

    public init() {}

    public func reset() {
        history.removeAll(keepingCapacity: true)
        labelStartedAt = nil
        lastLabel = nil
        lastUpdateAt = nil
    }

    public func reject(_ reason: String) -> GestureTemporalDecision {
        reset()
        return GestureTemporalDecision(
            execute: false,
            gesture: GestureFeatureExtractor.rejectLabel,
            confidence: 0,
            stableFrames: 0,
            heldSeconds: 0,
            reason: reason
        )
    }

    public func update(
        _ resolution: GestureGeometryResolution,
        knownGestureMass: Float,
        timestamp: TimeInterval
    ) -> GestureTemporalDecision {
        guard knownGestureMass.isFinite else { return reject("invalid known-gesture mass") }
        guard resolution.valid else { return reject(resolution.reason) }
        let row = normalize(resolution.probabilities)
        let ordered = row.sorted()
        let rawConfidence = ordered.last ?? 0
        let runnerUp = ordered.dropLast().last ?? 0
        let margin = rawConfidence - runnerUp
        let gesture = GestureFeatureExtractor.classNames[argmax(row)]
        let recoveryMassOK = knownGestureMass >= geometryRecoveryMassFloor
        let dorsalFallback = resolution.geometrySupported && gesture == "dorsal" && recoveryMassOK
        let directionalFallback = resolution.directionalRecovery
            && resolution.geometrySupported
            && ["left", "right", "up", "down"].contains(gesture)
            && (gesture == "down" || recoveryMassOK)
        let supportedHand = resolution.geometrySupported
            && ["open_palm", "like", "fist", "thumb_down"].contains(gesture)
            && recoveryMassOK

        if knownGestureMass < knownMassFloor && !(dorsalFallback || directionalFallback || supportedHand) {
            return reject("outside the ten-gesture vocabulary")
        }
        if rawConfidence < confidenceFloor { return reject("below confidence floor") }
        if margin < probabilityMarginFloor { return reject("ambiguous probability margin") }
        if let previous = history.last,
           argmax(previous) != argmax(row) || (lastUpdateAt != nil && timestamp - lastUpdateAt! >= 0.5) {
            reset()
        }
        if let lastUpdateAt, timestamp <= lastUpdateAt {
            return reject("duplicate or out-of-order frame timestamp")
        }
        self.lastUpdateAt = timestamp
        history.append(row)
        if history.count > maximumHistory { history.removeFirst(history.count - maximumHistory) }

        let ema = probabilityEMA(history, alpha: emaAlpha)
        let labels = ema.map(argmax)
        let finalRow = ema.last ?? row
        let finalIndex = labels.last ?? argmax(row)
        let finalGesture = GestureFeatureExtractor.classNames[finalIndex]
        let confidence = finalRow[finalIndex]
        let stableFrames = trailingEqualCount(labels)
        if finalGesture != lastLabel {
            lastLabel = finalGesture
            labelStartedAt = timestamp
        }
        let heldSeconds = max(0, timestamp - (labelStartedAt ?? timestamp))
        var requiredHold = minimumHoldSeconds
        if dorsalFallback && knownGestureMass < knownMassFloor { requiredHold = max(requiredHold, 0.35) }
        if directionalFallback && knownGestureMass < knownMassFloor { requiredHold = max(requiredHold, 0.40) }
        if supportedHand && knownGestureMass < knownMassFloor { requiredHold = max(requiredHold, 0.45) }
        let execute = confidence >= confidenceFloor
            && stableFrames >= stableFramesRequired
            && heldSeconds >= requiredHold
        let reason: String
        if confidence < confidenceFloor {
            reason = "below confidence floor"
        } else if stableFrames < stableFramesRequired {
            reason = "insufficient consecutive stable frames"
        } else if heldSeconds < requiredHold {
            reason = "gesture hold requirement not reached"
        } else if dorsalFallback && knownGestureMass < knownMassFloor {
            reason = "stable Dorsal geometry"
        } else if directionalFallback && knownGestureMass < knownMassFloor {
            reason = "stable directional geometry"
        } else {
            reason = "stable and confident"
        }
        return GestureTemporalDecision(
            execute: execute,
            gesture: finalGesture,
            confidence: confidence,
            stableFrames: stableFrames,
            heldSeconds: heldSeconds,
            reason: reason
        )
    }

    private func probabilityEMA(_ rows: [[Float]], alpha: Float) -> [[Float]] {
        guard var state = rows.first else { return [] }
        state = normalize(state)
        var output = [state]
        for row in rows.dropFirst() {
            state = zip(row, state).map { max(1e-12, alpha * $0.0 + (1 - alpha) * $0.1) }
            state = normalize(state)
            output.append(state)
        }
        return output
    }

    private func trailingEqualCount(_ values: [Int]) -> Int {
        guard let last = values.last else { return 0 }
        var count = 0
        for value in values.reversed() {
            guard value == last else { break }
            count += 1
        }
        return count
    }

    private func normalize(_ values: [Float]) -> [Float] {
        let total = max(values.reduce(0, +), 1e-12)
        return values.map { $0 / total }
    }

    private func argmax(_ values: [Float]) -> Int {
        values.indices.max(by: { values[$0] < values[$1] }) ?? 0
    }
}
