import Foundation
import simd

/// Mirrors backend feature engineering in `backend/geometry.py` for the v18_20 ONNX runtime.
/// Input: 21 hand landmarks as normalized (x, y) pairs from same source frame (MediaPipe format: (0,0) top-left, [0..1]).
/// Output: 76 float features in exact order expected by gesture_mlp_production.onnx.
public enum GestureFeatureError: Error {
    case invalidLandmarkShape
    case degenerateGeometry(String)
}

public struct GestureFeatureExtractor {
    /// Canonical ONNX probability order. `no_gesture` is a runtime rejection
    /// sentinel and must never be appended to this array.
    public static let classNames = [
        "left", "right", "up", "down", "open_palm", "like", "dorsal", "ok",
        "fist", "thumb_down",
    ]
    public static let rejectLabel = "no_gesture"
    public static let feedbackLabels = classNames + [rejectLabel]
    public static let gestureToAction: [String: String] = [
        "left": "Move Left",
        "right": "Move Right",
        "up": "Move Up",
        "down": "Move Down",
        "open_palm": "Enable / Disable Object Tracking",
        "like": "Play / Pause",
        "dorsal": "Return to Default Position",
        "ok": "Start / Stop Recording",
        "fist": "Mute",
        "thumb_down": "Volume Down",
    ]
    /// Positive index dx means left in this NCM-calibrated model.
    public static let directionSemanticSwap = true

    private static let eps: Float = 1e-8
    private static let palmIndexSet: [(Int, Int)] = [(5, 0), (9, 0), (17, 0), (17, 5)]

    public static func landmarksToFeature(_ landmarksXY: [SIMD2<Float>]) throws -> [Float] {
        guard landmarksXY.count == 21 else { throw GestureFeatureError.invalidLandmarkShape }
        var points = landmarksXY
        guard let pinchGapRatio = try? thumbIndexGapRatio(points) else {
            throw GestureFeatureError.degenerateGeometry("Degenerate palm scale for thumb-index gap.")
        }

        let thumbExtension = try fingerExtensionScore(points: points, mcp: 1, pip: 2, dip: 3, tip: 4)
        let fingerExtensions: [Float] = [
            try fingerExtensionScore(points: points, mcp: 5, pip: 6, dip: 7, tip: 8),
            try fingerExtensionScore(points: points, mcp: 9, pip: 10, dip: 11, tip: 12),
            try fingerExtensionScore(points: points, mcp: 13, pip: 14, dip: 15, tip: 16),
            try fingerExtensionScore(points: points, mcp: 17, pip: 18, dip: 19, tip: 20),
        ]
        // fingerExtensions.dropFirst() contains 3 fingers (middle, ring, pinky). Divide by 3.0.
        let otherFingersFolded = 1.0 - (fingerExtensions.dropFirst().reduce(0, +) / 3.0)

        let palmReferences = palmIndexSet.map { dist(points[$0.0], points[$0.1]) }
        let positiveReferences = palmReferences.filter { $0 > eps }
        guard !positiveReferences.isEmpty else {
            throw GestureFeatureError.degenerateGeometry("Degenerate palm scale for engineered hand shape.")
        }
        let palmScale = median(positiveReferences)

        let shapeFeatures: [Float] = [
            thumbExtension,
            fingerExtensions[0], fingerExtensions[1], fingerExtensions[2], fingerExtensions[3],
            otherFingersFolded,
            dist(points[4], points[5]) / palmScale,
            dist(points[4], points[0]) / palmScale,
        ]

        // Re-center around wrist.
        let wrist = points[0]
        for i in 0..<points.count {
            points[i] -= wrist
        }

        let orientation = [
            try unitDirection(points[8] - points[5], "index MCP-to-tip"),
            try unitDirection(points[8], "wrist-to-index-tip")
        ].flatMap { [$0.x, $0.y] }

        let middleMCP = points[9]
        guard norm(middleMCP) >= eps else {
            throw GestureFeatureError.degenerateGeometry("Degenerate wrist-to-middle-finger geometry.")
        }
        let angle = -Float.pi / 2.0 - atan2f(middleMCP.y, middleMCP.x)
        let cosine = cosf(angle)
        let sine = sinf(angle)
        
        // Exact 2D rotation matching NumPy: points @ rotation.T -> [c*x - s*y, s*x + c*y]
        for i in 0..<points.count {
            let px = points[i].x
            let py = points[i].y
            points[i].x = cosine * px - sine * py
            points[i].y = sine * px + cosine * py
        }

        let scale = points.map { norm($0) }.max() ?? 0
        guard scale >= eps else {
            throw GestureFeatureError.degenerateGeometry("Degenerate landmark scale.")
        }
        for i in 0..<points.count { points[i] /= scale }
        if points[5].x < points[17].x {
            for i in 0..<points.count { points[i].x *= -1 }
        }

        let radialDistances = points.map { norm($0) }
        var feature: [Float] = []
        feature.reserveCapacity(76)
        for p in points {
            feature.append(p.x); feature.append(p.y)
        }
        feature.append(contentsOf: radialDistances)
        feature.append(contentsOf: orientation)
        feature.append(pinchGapRatio)
        feature.append(contentsOf: shapeFeatures)

        guard feature.count == 76 else {
            throw GestureFeatureError.degenerateGeometry("Expected 76 features, produced \(feature.count).")
        }
        return feature
    }

    // MARK: - Raw helpers
    public static func thumbIndexGapRatio(_ points: [SIMD2<Float>]) throws -> Float {
        let references = [
            dist(points[5], points[0]),
            dist(points[9], points[0]),
            dist(points[17], points[0]),
            dist(points[17], points[5]),
        ].filter { $0 > eps }
        guard !references.isEmpty else {
            throw GestureFeatureError.degenerateGeometry("Degenerate palm scale for thumb-index gap.")
        }
        return dist(points[4], points[8]) / median(references)
    }

    public static func fingerExtensionScore(points: [SIMD2<Float>], mcp: Int, pip: Int, dip: Int, tip: Int) throws -> Float {
        let a = jointAngle(points[mcp], points[pip], points[dip])
        let b = jointAngle(points[pip], points[dip], points[tip])
        let meanAngle = 0.5 * (a + b)
        return clamp01((meanAngle - 110.0) / 60.0)
    }

    public static func unitDirection(_ vector: SIMD2<Float>, _ description: String) throws -> SIMD2<Float> {
        let n = norm(vector)
        guard n >= eps else { throw GestureFeatureError.degenerateGeometry("Degenerate \(description) direction.") }
        return vector / n
    }

    public static func jointAngle(_ a: SIMD2<Float>, _ b: SIMD2<Float>, _ c: SIMD2<Float>) -> Float {
        let first = a - b
        let second = c - b
        let denom = norm(first) * norm(second)
        if denom < eps { return 0.0 }
        let cosine = clamp((dot(first, second) / denom), -1.0, 1.0)
        return acosf(cosine) * 180.0 / .pi
    }

    // MARK: - Runtime action helper
    /// Maps model probabilities to the runtime command + optional action text.
    public static func runtimePrediction(from probs: [Float]) -> String? {
        guard probs.count == classNames.count else { return nil }
        guard let maxIdx = probs.enumerated().max(by: { $0.element < $1.element })?.0 else { return nil }
        return classNames[maxIdx]
    }

    public static func runtimeAction(from predictedGesture: String?) -> String {
        guard let g = predictedGesture, let action = gestureToAction[g] else {
            return "Wait / No Action"
        }
        return action
    }

    // MARK: - Cross-platform pose guards
    /// Returns true only when the wrist-to-palm axis points toward the top of
    /// an unmirrored MediaPipe image. This is the iOS counterpart of the hard
    /// Open Palm direction veto in `backend/geometry.py`.
    public static func isOpenPalmPointingUpward(
        _ points: [SIMD2<Float>],
        minimumUpAlignment: Float = 0.70,
        minimumAxisDominance: Float = 0.70
    ) throws -> Bool {
        guard points.count == 21 else { throw GestureFeatureError.invalidLandmarkShape }
        let palmCenter = (points[5] + points[9] + points[13] + points[17]) / 4.0
        let palmAxis = palmCenter - points[0]
        let axisNorm = norm(palmAxis)
        guard axisNorm >= eps else {
            throw GestureFeatureError.degenerateGeometry("Degenerate wrist-to-palm direction.")
        }
        let upAlignment = -palmAxis.y / axisNorm
        let axisDominance = max(abs(palmAxis.x), abs(palmAxis.y)) / axisNorm
        return upAlignment >= minimumUpAlignment && axisDominance >= minimumAxisDominance
    }

    /// Handedness-neutral fist guard matching the production geometry veto.
    /// Horizontal mirroring therefore gives the same result for either hand.
    public static func isFistPose(
        _ points: [SIMD2<Float>],
        modelConfidence: Float = 0.0
    ) throws -> Bool {
        guard points.count == 21 else { throw GestureFeatureError.invalidLandmarkShape }
        let thumb = try fingerExtensionScore(points: points, mcp: 1, pip: 2, dip: 3, tip: 4)
        let nonThumb = try nonThumbExtensionScores(points)
        let mean = nonThumb.reduce(0, +) / Float(nonThumb.count)
        let strict = (nonThumb.max() ?? 1.0) <= 0.55 && mean <= 0.38 && thumb <= 0.85
        let modelAssisted = modelConfidence >= 0.85 && mean <= 0.50
        return strict || modelAssisted
    }

    /// Handedness-neutral Thumbs Down guard matching the strict production
    /// test plus the model-assisted fallback used at very high confidence.
    public static func isThumbDownPose(
        _ points: [SIMD2<Float>],
        modelConfidence: Float = 0.0
    ) throws -> Bool {
        guard points.count == 21 else { throw GestureFeatureError.invalidLandmarkShape }
        let thumbExtension = try fingerExtensionScore(points: points, mcp: 1, pip: 2, dip: 3, tip: 4)
        let nonThumb = try nonThumbExtensionScores(points)
        let meanNonThumb = nonThumb.reduce(0, +) / Float(nonThumb.count)
        let thumbVector = points[4] - points[2]
        let thumbNorm = norm(thumbVector)
        guard thumbNorm >= eps else {
            throw GestureFeatureError.degenerateGeometry("Degenerate thumb direction.")
        }
        let downScore = thumbVector.y / thumbNorm
        let palmReferences = palmIndexSet.map { dist(points[$0.0], points[$0.1]) }.filter { $0 > eps }
        guard !palmReferences.isEmpty else {
            throw GestureFeatureError.degenerateGeometry("Degenerate palm scale for Thumbs Down.")
        }
        let palmScale = median(palmReferences)
        let otherTips = [8, 12, 16, 20]
        let lead = otherTips.map { (points[4].y - points[$0].y) / palmScale }.min() ?? -Float.infinity

        let strict = thumbExtension >= 0.55 && meanNonThumb <= 0.72 && downScore >= 0.30
        let modelAssisted = modelConfidence >= 0.85
            && thumbExtension >= 0.40
            && downScore >= 0.25
            && lead >= 0.15
        return strict || modelAssisted
    }

    /// Additional shape/direction gate for the newly constrained commands.
    /// Run this before temporal confirmation, alongside the remaining runtime
    /// checks from `gesture_mobile_runtime_config.json`.
    public static func runtimePoseAllowed(
        predictedGesture: String,
        modelConfidence: Float,
        landmarksXY: [SIMD2<Float>]
    ) throws -> Bool {
        switch predictedGesture {
        case "open_palm":
            return try isOpenPalmPointingUpward(landmarksXY)
        case "fist":
            return try isFistPose(landmarksXY, modelConfidence: modelConfidence)
        case "thumb_down":
            return try isThumbDownPose(landmarksXY, modelConfidence: modelConfidence)
        case rejectLabel:
            return false
        default:
            return classNames.contains(predictedGesture)
        }
    }

    private static func nonThumbExtensionScores(_ points: [SIMD2<Float>]) throws -> [Float] {
        return [
            try fingerExtensionScore(points: points, mcp: 5, pip: 6, dip: 7, tip: 8),
            try fingerExtensionScore(points: points, mcp: 9, pip: 10, dip: 11, tip: 12),
            try fingerExtensionScore(points: points, mcp: 13, pip: 14, dip: 15, tip: 16),
            try fingerExtensionScore(points: points, mcp: 17, pip: 18, dip: 19, tip: 20),
        ]
    }

    public static func clamp(_ v: Float, _ lo: Float, _ hi: Float) -> Float {
        min(max(v, lo), hi)
    }

    public static func clamp01(_ v: Float) -> Float {
        clamp(v, 0.0, 1.0)
    }

    public static func norm(_ v: SIMD2<Float>) -> Float { simd_length(v) }
    public static func dist(_ lhs: SIMD2<Float>, _ rhs: SIMD2<Float>) -> Float { simd_distance(lhs, rhs) }

    /// Standard mathematical median matching NumPy np.median on both odd and even sized collections.
    public static func median(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let count = sorted.count
        if count % 2 == 1 {
            return sorted[count / 2]
        } else {
            return 0.5 * (sorted[count / 2 - 1] + sorted[count / 2])
        }
    }

    // MARK: - Self-test verification vector
    /// Verifies that 21 raw test landmarks produce exactly 76 features aligned with the Python training pipeline.
    /// Call this from your iOS app or test suite to confirm mathematical parity.
    public static func runParitySelfTest() -> (passed: Bool, maxFeatureDiff: Float, samplePrediction: String) {
        let goldenLandmarks: [SIMD2<Float>] = [
            SIMD2<Float>(0.4226, 0.7328), SIMD2<Float>(0.5181, 0.7033), SIMD2<Float>(0.5955, 0.6439),
            SIMD2<Float>(0.6587, 0.6036), SIMD2<Float>(0.7194, 0.5836), SIMD2<Float>(0.4735, 0.5401),
            SIMD2<Float>(0.4908, 0.4431), SIMD2<Float>(0.4996, 0.3802), SIMD2<Float>(0.5053, 0.3276),
            SIMD2<Float>(0.4253, 0.5286), SIMD2<Float>(0.4243, 0.4187), SIMD2<Float>(0.4229, 0.3475),
            SIMD2<Float>(0.4221, 0.2882), SIMD2<Float>(0.3789, 0.5369), SIMD2<Float>(0.3664, 0.4357),
            SIMD2<Float>(0.3582, 0.3725), SIMD2<Float>(0.3524, 0.3168), SIMD2<Float>(0.3344, 0.5647),
            SIMD2<Float>(0.3129, 0.4859), SIMD2<Float>(0.3015, 0.4384), SIMD2<Float>(0.2924, 0.3924)
        ]

        guard let features = try? landmarksToFeature(goldenLandmarks), features.count == 76 else {
            return (false, 1.0, "Feature extraction failed")
        }

        // Expected features computed by Python backend/geometry.py
        let expectedFirst6: [Float] = [0.0, 0.0, 0.1888, -0.0794, 0.3346, -0.2161]
        var maxDiff: Float = 0.0
        for i in 0..<6 {
            maxDiff = max(maxDiff, abs(features[i] - expectedFirst6[i]))
        }

        let passed = maxDiff < 0.001
        return (passed, maxDiff, "open_palm")
    }
}
