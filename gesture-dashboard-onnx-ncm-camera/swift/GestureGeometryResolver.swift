import Foundation
import simd

/// Swift port of the production post-model geometry in `reference_python/geometry.py`.
///
/// Run this after ONNX inference and before `GestureTemporalGate`. The ONNX model is
/// intentionally unchanged; the reported camera-angle Down poses are recovered here.
public struct GestureGeometryResolution {
    public let probabilities: [Float]
    public let gesture: String
    public let valid: Bool
    public let reason: String
    public let geometrySupported: Bool
    public let directionalRecovery: Bool
    public let reversePalmEdgeDown: Bool
    public let downRecovery: Bool
    public let sidewaysPalmRecovery: Bool
    public let dorsalRangeRecovery: Bool
}

public enum GestureGeometryResolver {
    private static let eps: Float = 1e-8
    private static let minimumPeaceTipSeparationRatio: Float = 0.04
    private static let directionalNames: Set<String> = ["left", "right", "up", "down"]

    private struct ReturnMainGeometry {
        let score: Float
        let strongGeometry: Bool
        let downwardFingerCount: Int
        let minimumExtension: Float
        let downwardScore: Float
        let togetherScore: Float
        let parallelScore: Float
    }

    private struct FistGeometry {
        let strongGeometry: Bool
        let retractedFingerCount: Int
        let meanCurlback: Float
        let maximumTipRadius: Float
        let thumbRadius: Float
    }

    private struct DirectionalState {
        let probabilities: [Float]
        let valid: Bool
        let reason: String
        let gesture: String
        let strongGeometry: Bool
        let modelSupportedPose: Bool
        let edgeOnDown: Bool
        let curledEdgeDown: Bool
        let clusteredEdgeDown: Bool
        let reversePalmEdgeDown: Bool
        let prominentIndexDown: Bool
    }

    private struct ShapeState {
        let probabilities: [Float]
        let valid: Bool
        let reason: String
        let dorsalSupported: Bool
        let dorsalHardNegative: Bool
        let palmSupported: Bool
        let likeSupported: Bool
        let fistSupported: Bool
        let thumbDownSupported: Bool
        let peaceSupported: Bool
        let sidewaysPalmRecovery: Bool
        let dorsalRangeRecovery: Bool
    }

    /// Resolve model probabilities using the same directional and hand-shape rules
    /// as the desktop runtime. Landmarks must be the 21 unmirrored MediaPipe image
    /// landmarks in canonical order, with +y pointing down the image.
    public static func resolve(
        probabilities: [Float],
        landmarks points: [SIMD2<Float>],
        handedness: String? = nil,
        handednessConfidence: Float? = nil,
        worldLandmarks: [SIMD3<Float>]? = nil,
        requireFistDepth: Bool = false
    ) throws -> GestureGeometryResolution {
        guard points.count == 21, points.allSatisfy({ $0.x.isFinite && $0.y.isFinite }) else {
            throw GestureFeatureError.invalidLandmarkShape
        }
        if let worldLandmarks {
            guard worldLandmarks.count == 21, worldLandmarks.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }) else {
                throw GestureFeatureError.invalidLandmarkShape
            }
        }
        guard probabilities.count == GestureFeatureExtractor.classNames.count,
              probabilities.allSatisfy({ $0.isFinite && $0 >= 0 }),
              probabilities.reduce(0, +) > eps else {
            throw GestureFeatureError.degenerateGeometry("Invalid classifier probabilities.")
        }

        let normalized = normalize(probabilities)
        let directional = try resolveDirectional(normalized, points)
        let shape = try resolveShape(
            directional.probabilities,
            points,
            handedness: handedness,
            handednessConfidence: handednessConfidence,
            worldLandmarks: worldLandmarks,
            requireFistDepth: requireFistDepth
        )
        let adjusted = directional.prominentIndexDown
            ? setProbabilityFloor(shape.probabilities, target: index("down"), floor: 0.98)
            : shape.probabilities
        let resolvedIndex = argmax(adjusted)
        let resolved = GestureFeatureExtractor.classNames[resolvedIndex]
        let isDirectional = directionalNames.contains(resolved)
        let hardNegative = resolved == "dorsal" || shape.dorsalSupported || shape.dorsalHardNegative
        let valid = !hardNegative && (isDirectional ? directional.valid : shape.valid)
        let reason = hardNegative ? "dorsal is a hard negative" : (isDirectional ? directional.reason : shape.reason)
        let geometrySupported =
            (resolved == "dorsal" && shape.dorsalSupported)
            || (resolved == "open_palm" && shape.palmSupported)
            || (resolved == "like" && shape.likeSupported)
            || (resolved == "fist" && shape.fistSupported)
            || (resolved == "thumb_down" && shape.thumbDownSupported)
            || (resolved == "peace" && shape.peaceSupported)
            || (isDirectional && directional.strongGeometry)
        let directionalRecovery = geometrySupported
            && directional.strongGeometry
            && directional.modelSupportedPose
        let downRecovery = resolved == "down" && (
            directional.edgeOnDown
            || directional.curledEdgeDown
            || directional.clusteredEdgeDown
            || directional.reversePalmEdgeDown
            || directional.prominentIndexDown
        )

        return GestureGeometryResolution(
            probabilities: adjusted,
            gesture: hardNegative ? GestureFeatureExtractor.rejectLabel : resolved,
            valid: valid,
            reason: reason,
            geometrySupported: !hardNegative && geometrySupported,
            directionalRecovery: directionalRecovery,
            reversePalmEdgeDown: directional.reversePalmEdgeDown,
            downRecovery: downRecovery,
            sidewaysPalmRecovery: resolved == "open_palm" && shape.sidewaysPalmRecovery,
            dorsalRangeRecovery: false
        )
    }

    private static func resolveDirectional(
        _ probabilities: [Float],
        _ points: [SIMD2<Float>]
    ) throws -> DirectionalState {
        let raw = GestureFeatureExtractor.classNames[argmax(probabilities)]
        if raw == "peace" || raw == "rock" {
            return DirectionalState(
                probabilities: probabilities, valid: true,
                reason: "multi-finger command retains model label", gesture: raw,
                strongGeometry: false, modelSupportedPose: false,
                edgeOnDown: false, curledEdgeDown: false,
                clusteredEdgeDown: false, reversePalmEdgeDown: false, prominentIndexDown: false
            )
        }
        let poseScore = try singleIndexPoseScore(points)
        let vector = points[8] - points[5]
        let vectorNorm = GestureFeatureExtractor.norm(vector)
        guard vectorNorm >= eps else {
            return DirectionalState(
                probabilities: probabilities,
                valid: !directionalNames.contains(raw),
                reason: "degenerate index-finger direction",
                gesture: raw,
                strongGeometry: false,
                modelSupportedPose: false,
                edgeOnDown: false,
                curledEdgeDown: false,
                clusteredEdgeDown: false,
                reversePalmEdgeDown: false, prominentIndexDown: false
            )
        }

        let nonIndexExtensions: [Float] = [
            try extensionScore(points, 9, 10, 11, 12),
            try extensionScore(points, 13, 14, 15, 16),
            try extensionScore(points, 17, 18, 19, 20),
        ]
        let dominance = max(abs(vector.x), abs(vector.y)) / vectorNorm
        let strictPose = poseScore >= 0.50 && mean(nonIndexExtensions) <= 0.70
        let gesture: String
        if abs(vector.y) >= abs(vector.x) {
            gesture = vector.y > 0 ? "down" : "up"
        } else {
            // This is the trained NCM semantic calibration on unmirrored pixels.
            gesture = vector.x > 0 ? "left" : "right"
        }

        let palmScale = max(GestureFeatureExtractor.dist(points[9], points[0]), 1e-6)
        let indexPath = GestureFeatureExtractor.dist(points[6], points[5])
            + GestureFeatureExtractor.dist(points[7], points[6])
            + GestureFeatureExtractor.dist(points[8], points[7])
        let indexStraightness = vectorNorm / max(indexPath, eps)
        let vectorUnit = vector / vectorNorm
        let indexLead = [12, 16, 20].map {
            simd_dot(points[8] - points[$0], vectorUnit) / palmScale
        }.min() ?? -.infinity
        let indexReach = vectorNorm / palmScale
        let otherPairs = [(9, 12), (13, 16), (17, 20)]
        let maximumOtherProjectedReach = otherPairs.map { mcp, tip in
            simd_dot(points[tip] - points[mcp], vectorUnit) / palmScale
        }.max() ?? 0
        let relativeIndexReach = indexReach - maximumOtherProjectedReach
        let indexExtension = try extensionScore(points, 5, 6, 7, 8)
        let projected = try returnMainPoseGeometry(points)

        let mcps = [5, 9, 13, 17]
        let tips = [8, 12, 16, 20]
        let fingerVectors = zip(mcps, tips).map { points[$0.1] - points[$0.0] }
        let fingerNorms = fingerVectors.map(GestureFeatureExtractor.norm)
        let fingerUnits = zip(fingerVectors, fingerNorms).map {
            $0.0 / max($0.1, eps)
        }
        let fingerReaches = fingerNorms.map { $0 / palmScale }
        let fingerExtensions = [indexExtension] + nonIndexExtensions
        let palmDownAlignment = (points[9].y - points[0].y) / palmScale
        let fingertipY = tips.map { points[$0].y }
        let fingertipDescent = zip(fingertipY.dropLast(), fingertipY.dropFirst()).map { $0.1 - $0.0 }
        let innerTipSpread = spread(Array(fingertipY.prefix(3))) / palmScale
        let outerTipDrop = (fingertipY[3] - (Array(fingertipY.prefix(3)).max() ?? fingertipY[3])) / palmScale
        let outerReachAdvantage = fingerReaches[3] / max(mean(Array(fingerReaches.prefix(3))), 1e-6)
        let deepestFingerIndex = fingertipY.indices.max(by: { fingertipY[$0] < fingertipY[$1] }) ?? 0
        let remainingIndices = fingertipY.indices.filter { $0 != deepestFingerIndex }
        let remainingTipY = remainingIndices.map { fingertipY[$0] }
        let edgeTipClusterSpread = spread(remainingTipY) / palmScale
        let deepestEdgeTipDrop = (fingertipY[deepestFingerIndex] - (remainingTipY.max() ?? fingertipY[deepestFingerIndex])) / palmScale
        let deepestEdgeReach = fingerReaches[deepestFingerIndex]
        let remainingReachMean = mean(remainingIndices.map { fingerReaches[$0] })
        let deepestEdgeReachAdvantage = deepestEdgeReach / max(remainingReachMean, 1e-6)
        let minimumFingerDownAlignment = fingerUnits.map { $0.y }.min() ?? -.infinity

        let edgeOnDown = gesture == "down"
            && dominance >= 0.82
            && indexStraightness >= 0.76
            && indexReach >= 0.55
            && indexExtension >= 0.35
            && projected.downwardFingerCount == 3
            && projected.downwardScore >= 0.90
            && projected.parallelScore >= 0.88
            && projected.togetherScore >= 0.50
            && projected.score < 0.76
        let curledEdgeDown = gesture == "down"
            && dominance >= 0.88
            && indexReach >= 0.40
            && palmDownAlignment >= 0.85
            && minimumFingerDownAlignment >= 0.86
            && projected.parallelScore >= 0.92
            && fingertipDescent.allSatisfy { $0 >= 0 }
            && (Array(fingerExtensions.prefix(3)).max() ?? 1) <= 0.45
            && fingerExtensions[3] >= 0.75
            && fingerReaches[3] >= 1.45
            && projected.downwardFingerCount == 1
            && projected.score < 0.55
        let clusteredEdgeDown = gesture == "down"
            && dominance >= 0.80
            && indexReach >= 0.55
            && palmDownAlignment >= 0.80
            && minimumFingerDownAlignment >= 0.80
            && projected.downwardScore >= 0.95
            && projected.parallelScore >= 0.84
            && projected.downwardFingerCount <= 1
            && projected.score < 0.55
            && innerTipSpread <= 0.45
            && outerTipDrop >= 0.75
            && (Array(fingerExtensions.prefix(3)).max() ?? 1) <= 0.45
            && fingerReaches[3] >= 1.55
            && outerReachAdvantage >= 1.55
        let reversePalmEdgeDown = gesture == "down"
            && dominance >= 0.78
            && palmDownAlignment <= 0.35
            && (deepestFingerIndex == 0 || deepestFingerIndex == 3)
            && minimumFingerDownAlignment >= 0.75
            && projected.downwardScore >= 0.90
            && projected.parallelScore >= 0.80
            && edgeTipClusterSpread <= 0.90
            && deepestEdgeTipDrop >= 0.55
            && deepestEdgeReach >= 1.20
            && deepestEdgeReachAdvantage >= 1.30

        let prominentIndexDown = gesture == "down"
            && dominance >= 0.90 && indexExtension >= 0.75
            && indexStraightness >= 0.90 && indexReach >= 0.90
            && indexLead >= 0.40 && relativeIndexReach >= 0.20
            && palmDownAlignment >= 0.60 && deepestFingerIndex == 0
            && edgeTipClusterSpread <= 0.45 && deepestEdgeTipDrop >= 0.45
            && deepestEdgeReachAdvantage >= 1.40

        let ordered = probabilities.sorted()
        let top = ordered.last ?? 0
        let runnerUp = ordered.dropLast().last ?? 0
        let modelAgrees = raw == gesture && top >= 0.70 && top - runnerUp >= 0.08
        let shortStraightIndex = indexStraightness >= 0.88
            && indexReach >= 0.27
            && max(indexLead, relativeIndexReach) >= 0.12
            && mean(nonIndexExtensions) <= 0.35
        let indexOnly = (indexExtension >= 0.45 && indexStraightness >= 0.70 && indexLead >= 0.04)
            || shortStraightIndex
        let supportedPose = (
            modelAgrees && indexExtension >= 0.45 && indexStraightness >= 0.80 && indexLead >= 0.20
        ) || (modelAgrees && shortStraightIndex)
            || edgeOnDown || curledEdgeDown || clusteredEdgeDown || reversePalmEdgeDown || prominentIndexDown
        let directional = dominance >= 0.73 && (
            (indexOnly && (strictPose || supportedPose))
            || edgeOnDown || curledEdgeDown || clusteredEdgeDown || reversePalmEdgeDown || prominentIndexDown
        )
        let strongShortIndex = shortStraightIndex
            && gesture == "down" && modelAgrees && top >= 0.95 && dominance >= 0.80
        let strongGeometry = directional && (
            edgeOnDown || curledEdgeDown || clusteredEdgeDown || reversePalmEdgeDown || prominentIndexDown
            || (modelAgrees && top >= 0.95 && (
                (indexExtension >= 0.85 && indexStraightness >= 0.92 && indexLead >= 0.30 && dominance >= 0.86)
                || strongShortIndex
            ))
        )
        let reason: String
        if directional {
            reason = "direction geometry accepted"
        } else if dominance < 0.73 {
            reason = "Point more horizontally or vertically; the direction is diagonal"
        } else {
            reason = "Extend the index finger past the other fingertips and hold the direction"
        }
        let shouldValidateDirection = directionalNames.contains(raw)
        var adjusted = probabilities
        if directional {
            let floor = min(0.98, 0.92 + 0.06 * max(0, poseScore - 0.72) / 0.28)
            adjusted = setProbabilityFloor(adjusted, target: GestureFeatureExtractor.classNames.firstIndex(of: gesture)!, floor: floor)
        }
        return DirectionalState(
            probabilities: adjusted,
            valid: shouldValidateDirection ? directional : true,
            reason: reason,
            gesture: gesture,
            strongGeometry: strongGeometry,
            modelSupportedPose: supportedPose,
            edgeOnDown: edgeOnDown,
            curledEdgeDown: curledEdgeDown,
            clusteredEdgeDown: clusteredEdgeDown,
            reversePalmEdgeDown: reversePalmEdgeDown,
            prominentIndexDown: prominentIndexDown
        )
    }

    private static func resolveShape(
        _ probabilities: [Float], _ points: [SIMD2<Float>],
        handedness: String?, handednessConfidence: Float?,
        worldLandmarks: [SIMD3<Float>]?, requireFistDepth: Bool
    ) throws -> ShapeState {
        var adjusted = probabilities
        var raw = GestureFeatureExtractor.classNames[argmax(adjusted)]
        let fingers = [(1,2,3,4), (5,6,7,8), (9,10,11,12), (13,14,15,16), (17,18,19,20)]
        let extensions = try fingers.map { try extensionScore(points, $0.0, $0.1, $0.2, $0.3) }
        let peaceImage = peaceGeometry(points.map { SIMD3<Float>($0.x, $0.y, 0) })
        let peaceWorld = worldLandmarks.map { peaceGeometry($0) }
        let peaceSupport = adjusted[index("peace")]
        let peaceModel = raw == "peace" && peaceSupport >= 0.98
            && peaceImage.indexExtension >= 0.75 && peaceImage.middleExtension >= 0.65
            && peaceImage.reachRatio >= 0.70 && peaceImage.reachRatio <= 1.45
            && peaceImage.forwardRatio >= 0.25 && peaceImage.separation >= minimumPeaceTipSeparationRatio
            && min(extensions[3], extensions[4]) <= 0.65 && max(extensions[3], extensions[4]) <= 0.85
        let peaceSupported = peaceWorld?.valid ?? (peaceImage.valid || peaceModel)
        let peacePairRecovery = peaceImage.strong && peaceSupported
        if peaceSupported && (raw == "peace" || peacePairRecovery || peaceSupport >= (peaceWorld == nil ? 0.15 : 0.05)) {
            adjusted = setProbabilityFloor(adjusted, target: index("peace"), floor: 0.92)
            raw = "peace"
        }
        let thumbVector = points[4] - points[2]
        let thumbNorm = max(simd_length(thumbVector), eps)
        let thumbUpScore = -thumbVector.y / thumbNorm
        let nonThumb = Array(extensions.dropFirst())
        let extendedNonThumbCount = nonThumb.filter { $0 >= 0.58 }.count
        let gapScale = try palmGapScale(points)
        let gap = simd_distance(points[4], points[8]) / gapScale
        let closestOtherThumbGap = [12,16,20].map { simd_distance(points[4], points[$0]) / gapScale }.min()!
        let otherPairs = [(9,12), (13,16), (17,20)]
        let extendedOtherCount = otherPairs.enumerated().filter { i, pair in
            extensions[i + 2] >= 0.58 && simd_distance(points[pair.1], points[pair.0]) / gapScale >= 0.60
        }.count
        let dorsal = try returnMainPoseGeometry(points)
        let fist = try foreshortenedFistGeometry(points)
        let palmAxis = [5,9,13,17].map { points[$0] }.reduce(SIMD2<Float>(repeating: 0), +) / 4 - points[0]
        let palmAxisNorm = max(simd_length(palmAxis), eps)
        let palmUpAlignment = -palmAxis.y / palmAxisNorm
        var dorsalAxis = palmAxis
        var worldAxis: SIMD3<Float>?
        if let worldLandmarks {
            let axis = [5,9,13,17].map { worldLandmarks[$0] }.reduce(SIMD3<Float>(repeating: 0), +) / 4 - worldLandmarks[0]
            if simd_length(axis) >= eps { worldAxis = axis / simd_length(axis) }
            let xy = SIMD2<Float>(axis.x, axis.y)
            if simd_length(xy) > eps { dorsalAxis = xy }
        }
        let dorsalXY = atan2(dorsalAxis.y, dorsalAxis.x) * 180 / Float.pi
        let dorsalAllowedDirection = dorsalXY >= -0.0001 && dorsalXY <= 150.0001
        let rotation = Float.pi / 2 - atan2(palmAxis.y, palmAxis.x)
        let c = cos(rotation), s = sin(rotation)
        let aligned = points.map { p -> SIMD2<Float> in
            let v = p - points[0]
            return SIMD2<Float>(c * v.x - s * v.y, s * v.x + c * v.y)
        }
        let alignedDorsal = try returnMainPoseGeometry(aligned)
        let openPalmAllowedDirection = palmUpAlignment >= -0.10
        let normalizedHand = handedness?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? ""
        let surfaceValidationActive = normalizedHand == "left" || normalizedHand == "right"
        let surface = handSurface(points, handedness: surfaceValidationActive ? normalizedHand : nil, confidence: handednessConfidence)
        let palmVisible = surface == "palmar", dorsalVisible = surface == "dorsal"
        let fingerPairs = [(5,8), (9,12), (13,16), (17,20)]
        let fingerLengths = fingerPairs.map { simd_distance(points[$0.1], points[$0.0]) }
        let dorsalRangeRecovery = dorsalAllowedDirection && dorsalVisible && dorsalSupport(alignedDorsal)
            && (fingerLengths.min() ?? 0) / palmAxisNorm >= 0.35
        let dorsalSupported = (dorsalSupport(dorsal) || dorsalRangeRecovery) && dorsalAllowedDirection
            && (!surfaceValidationActive || dorsalVisible)
        let dorsalValid = dorsalRangeRecovery || (dorsalAllowedDirection && dorsal.score >= 0.76
            && dorsal.downwardFingerCount >= 4 && dorsal.minimumExtension >= 0.50
            && (!surfaceValidationActive || dorsalVisible))
        let openPalmValid = extendedNonThumbCount == 4 && mean(nonThumb) >= 0.78
            && !dorsalValid && openPalmAllowedDirection && (!surfaceValidationActive || palmVisible)
        let palmSupported = openPalmValid && palmVisible && (nonThumb.min() ?? 0) >= 0.65
            && mean(nonThumb) >= 0.82 && extensions[0] >= 0.45
        let sidewaysPalmRecovery = palmSupported && palmUpAlignment <= 0.75
        let palmScale = max(simd_distance(points[9], points[0]), 1e-6)
        let thumbLead = [8,12,16,20].map { (points[$0].y - points[4].y) / palmScale }.min()!
        let raisedCount = zip(nonThumb, fingerLengths).filter { $0.0 >= 0.58 && $0.1 / palmScale >= 0.70 }.count
        let likeFingerShape = thumbLead >= 0.10 && raisedCount == 0
        let likeValid = likeFingerShape && (
            (extensions[0] >= 0.60 && mean(nonThumb) <= 0.72 && thumbUpScore >= 0.55)
            || (raw == "like" && adjusted.max()! >= 0.90 && extensions[0] >= 0.40 && thumbUpScore >= 0.20))
        let likeSupported = ["like", "fist"].contains(raw) && extensions[0] >= 0.70
            && fist.thumbRadius >= 1.10 && nonThumb.max()! <= 0.50 && mean(nonThumb) <= 0.38
            && thumbUpScore >= 0.70 && thumbLead >= 0.25
        let thumbDownScore = thumbVector.y / thumbNorm
        let thumbDownLead = [8,12,16,20].map { (points[4].y - points[$0].y) / palmScale }.min()!
        let thumbDownValid = (extensions[0] >= 0.55 && mean(nonThumb) <= 0.72
            && thumbDownScore >= 0.30 && thumbDownLead >= 0.15)
            || (raw == "thumb_down" && adjusted.max()! >= 0.85 && extensions[0] >= 0.40
                && thumbDownScore >= 0.25 && thumbDownLead >= 0.15)
        let thumbDownSupported = thumbDownValid && extensions[0] >= 0.70 && mean(nonThumb) <= 0.45
            && thumbDownScore >= 0.70 && thumbDownLead >= 0.25
        let fistCompact = nonThumb.max()! <= 0.55 && mean(nonThumb) <= 0.38 && extensions[0] <= 0.85
        let fistXYAllowed = palmUpAlignment >= -sin(20 * Float.pi / 180) - 1e-6
        var reverseView = false, sideView = false, tiltedView = false, rejectedView = false
        var sideAlignment: Float?
        if let worldAxis {
            var axis = SIMD3<Float>(tan(72 * Float.pi / 180), 1 / tan(68 * Float.pi / 180), 1)
            axis /= simd_length(axis)
            sideAlignment = max(simd_dot(worldAxis, axis), simd_dot(worldAxis, axis * SIMD3<Float>(-1,1,1)))
            reverseView = mirroredAlignment(worldAxis, -130, -132) >= cos(12 * Float.pi / 180)
            sideView = mirroredAlignment(worldAxis, -5, 80) >= cos(18 * Float.pi / 180)
            tiltedView = mirroredAlignment(worldAxis, -3, 107) >= cos(10 * Float.pi / 180)
            let rejected: [(Float, Float)] = [(-64,126), (-68,50), (-20,102), (-61,142), (-64,40), (-64,140)]
            rejectedView = rejected.map { mirroredAlignment(worldAxis, $0.0, $0.1) }.max()! >= cos(6 * Float.pi / 180)
        }
        let reportedSideFistGeometry = (sideView || tiltedView) && fist.maximumTipRadius <= 0.95
            && fist.thumbRadius <= 1.10 && fingerLengths.max()! / palmAxisNorm <= 0.70
            && nonThumb.max()! <= 0.62 && mean(nonThumb) <= 0.42 && extensions[0] <= 0.85
            && !thumbDownValid && !likeValid
        let fistSideGeometry = (sideAlignment ?? -.infinity) >= cos(18 * Float.pi / 180) - 1e-6
            && fist.maximumTipRadius <= 0.85 && fist.thumbRadius <= 0.85
            && nonThumb.max()! <= 0.55 && mean(nonThumb) <= 0.38 && !thumbDownValid && !likeValid
        let fistDepthAllowed = (worldAxis != nil || !requireFistDepth)
            && (worldAxis == nil || worldAxis!.z >= -0.45 || reverseView)
        let fistAllowedDirection = (fistXYAllowed || reverseView || fistSideGeometry) && fistDepthAllowed && !rejectedView
        let fistValid = fistAllowedDirection && (fistCompact || fist.strongGeometry || fistSideGeometry || reportedSideFistGeometry)
        let fistSupported = fistAllowedDirection && (reportedSideFistGeometry || fistSideGeometry || fist.strongGeometry
            || (fistValid && nonThumb.max()! <= 0.42 && mean(nonThumb) <= 0.28))
        let fistRelabel = fistAllowedDirection && (fist.strongGeometry || fistSideGeometry || reportedSideFistGeometry)
            && (raw != "like" || !likeValid) && (raw != "thumb_down" || !thumbDownValid)
        let okValid = gap <= 0.58 && closestOtherThumbGap >= 0.70 && closestOtherThumbGap - gap >= 0.25
            && extendedOtherCount >= 2 && extensions[1] <= 0.82
        let allowDorsalResolution = !(raw == "open_palm" && !openPalmAllowedDirection && !dorsalVisible)
        if likeSupported {
            adjusted = setProbabilityFloor(adjusted, target: index("like"), floor: 0.96); raw = "like"
        } else if fistRelabel {
            adjusted = setProbabilityFloor(adjusted, target: index("fist"), floor: 0.96); raw = "fist"
        } else if allowDorsalResolution && (dorsalSupported || (dorsalValid && ["open_palm","dorsal","down"].contains(raw))) {
            adjusted = setProbabilityFloor(adjusted, target: index("dorsal"), floor: 0.96); raw = "dorsal"
        } else if palmSupported || (openPalmValid && ["open_palm","dorsal"].contains(raw)) {
            adjusted = setProbabilityFloor(adjusted, target: index("open_palm"), floor: 0.96); raw = "open_palm"
        }
        let valid = ["open_palm": openPalmValid, "like": likeValid, "dorsal": dorsalValid, "ok": okValid,
                     "fist": fistValid, "thumb_down": thumbDownValid, "peace": peaceSupported][raw] ?? true
        let reason: String
        if valid { reason = "pose geometry accepted" }
        else if raw == "dorsal" && !dorsalAllowedDirection { reason = "dorsal requires an XY angle from 0 to 150 degrees downward" }
        else if raw == "open_palm" && !openPalmAllowedDirection { reason = "open_palm must not point downward" }
        else if raw == "fist" && requireFistDepth && worldAxis == nil { reason = "fist depth landmarks unavailable" }
        else if raw == "fist" && rejectedView { reason = "fist orientation matches a no-gesture pose" }
        else if raw == "fist" && !fistDepthAllowed { reason = "fist points away from the camera" }
        else if raw == "fist" && !(fistXYAllowed || reverseView || fistSideGeometry) { reason = "fist points downward" }
        else if raw == "like" && !likeFingerShape { reason = "like requires the thumb to lead and other fingers to stay folded" }
        else if raw == "ok" && (closestOtherThumbGap < 0.70 || closestOtherThumbGap - gap < 0.25) { reason = "ok requires only the index fingertip to touch the thumb" }
        else { reason = "\(raw) hand shape failed geometry checks" }
        return ShapeState(probabilities: adjusted, valid: valid, reason: reason, dorsalSupported: dorsalSupported,
            dorsalHardNegative: dorsalVisible && (nonThumb.min() ?? 0) >= 0.45 && mean(nonThumb) >= 0.68,
            palmSupported: palmSupported, likeSupported: likeSupported, fistSupported: fistSupported,
            thumbDownSupported: thumbDownSupported, peaceSupported: peaceSupported,
            sidewaysPalmRecovery: sidewaysPalmRecovery, dorsalRangeRecovery: dorsalRangeRecovery)
    }

    private static func dorsalSupport(_ g: ReturnMainGeometry) -> Bool {
        g.score >= 0.90 && g.downwardFingerCount == 4 && g.minimumExtension >= 0.80
            && g.downwardScore >= 0.90 && g.togetherScore >= 0.65 && g.parallelScore >= 0.90
    }

    private static func palmGapScale(_ points: [SIMD2<Float>]) throws -> Float {
        let refs = [(0,5), (0,9), (0,17), (5,17)].map { simd_distance(points[$0.0], points[$0.1]) }.filter { $0 > eps }.sorted()
        guard !refs.isEmpty else { throw GestureFeatureError.degenerateGeometry("Degenerate palm scale.") }
        let mid = refs.count / 2
        return refs.count % 2 == 0 ? (refs[mid - 1] + refs[mid]) / 2 : refs[mid]
    }

    private static func mirroredAlignment(_ axis: SIMD3<Float>, _ xyDegrees: Float, _ zxDegrees: Float) -> Float {
        let xy = xyDegrees * Float.pi / 180, zx = zxDegrees * Float.pi / 180
        let ref = SIMD3<Float>(abs(cos(xy)), sin(xy), cos(xy) / tan(zx))
        return simd_dot(SIMD3<Float>(abs(axis.x), axis.y, axis.z), ref / simd_length(ref))
    }

    private struct PeaceGeometry {
        let valid: Bool, strong: Bool
        let indexExtension: Float, middleExtension: Float, reachRatio: Float, forwardRatio: Float, separation: Float
    }

    private static func peaceGeometry(_ p: [SIMD3<Float>]) -> PeaceGeometry {
        func angle(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>) -> Float {
            let u = a - b, v = c - b, denominator = simd_length(u) * simd_length(v)
            if denominator < eps { return 0 }
            return acos(min(1, max(-1, simd_dot(u,v) / denominator))) * 180 / Float.pi
        }
        func ext(_ a: Int, _ b: Int, _ c: Int, _ d: Int) -> Float {
            min(1, max(0, ((angle(p[a],p[b],p[c]) + angle(p[b],p[c],p[d])) / 2 - 110) / 60))
        }
        let i = p[8] - p[5], m = p[12] - p[9]
        let mr = simd_length(m), ir = simd_length(i), mu = m / max(mr, eps)
        let ratio = ir / max(mr, eps), forward = simd_dot(i, mu) / max(mr, eps)
        let scale = max(simd_distance(p[9],p[0]), eps)
        let separation = simd_distance(p[8],p[12]) / scale
        let ie = ext(5,6,7,8), me = ext(9,10,11,12), re = ext(13,14,15,16), pe = ext(17,18,19,20)
        let near = separation <= 0.35
        let raisedOther = (re > 0.65 && (simd_distance(p[16],p[13]) / max(mr,eps) >= 0.65 || !near))
            || (pe > 0.65 && (simd_distance(p[20],p[17]) / max(mr,eps) >= 0.65 || !near))
        var pair = i / max(ir, eps) + mu
        pair /= max(simd_length(pair), eps)
        let tipForward = [8,12,16,20].map { simd_dot(p[$0] - p[0], pair) / scale }
        let lead = min(tipForward[0], tipForward[1]) - max(tipForward[2], tipForward[3])
        let otherForward = [(13,16), (17,20)].map { simd_dot(p[$0.1] - p[$0.0], pair) / scale }.max()!
        let strong = ie >= 0.80 && me >= 0.80 && ratio >= 0.70 && ratio <= 1.35 && forward >= 0.65
            && min(ir,mr) / scale >= 0.80 && separation >= 0.015 && separation <= 0.55
            && simd_distance(p[5],p[9]) / scale >= 0.05 && simd_distance(p[6],p[10]) / scale >= 0.05
            && lead >= 0.65 && otherForward <= 0.60
        let valid = ie >= 0.55 && me >= 0.45 && !raisedOther && ratio >= 0.62 && ratio <= 1.45
            && forward >= 0.42 && separation >= minimumPeaceTipSeparationRatio
        return PeaceGeometry(valid: valid || strong, strong: strong, indexExtension: ie, middleExtension: me,
            reachRatio: ratio, forwardRatio: forward, separation: separation)
    }

    private static func singleIndexPoseScore(_ points: [SIMD2<Float>]) throws -> Float {
        let indexExtension = try extensionScore(points, 5, 6, 7, 8)
        let otherExtensions: [Float] = [
            try extensionScore(points, 9, 10, 11, 12),
            try extensionScore(points, 13, 14, 15, 16),
            try extensionScore(points, 17, 18, 19, 20),
        ]
        let otherFolded = 1 - mean(otherExtensions)
        let palmScale = max(GestureFeatureExtractor.dist(points[9], points[0]), 1e-6)
        let indexReach = GestureFeatureExtractor.dist(points[8], points[0])
        let otherReach = [12, 16, 20].map { GestureFeatureExtractor.dist(points[$0], points[0]) }.max() ?? 0
        let prominence = GestureFeatureExtractor.clamp(((indexReach - otherReach) / palmScale + 0.10) / 0.70, 0, 1)
        return GestureFeatureExtractor.clamp(0.50 * indexExtension + 0.35 * otherFolded + 0.15 * prominence, 0, 1)
    }

    private static func returnMainPoseGeometry(_ points: [SIMD2<Float>]) throws -> ReturnMainGeometry {
        let fingers = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
        let extensions = try fingers.map { try extensionScore(points, $0.0, $0.1, $0.2, $0.3) }
        let vectors = fingers.map { points[$0.3] - points[$0.0] }
        let norms = vectors.map(GestureFeatureExtractor.norm)
        guard norms.allSatisfy({ $0 >= eps }) else {
            return ReturnMainGeometry(score: 0, strongGeometry: false, downwardFingerCount: 0, minimumExtension: 0, downwardScore: 0, togetherScore: 0, parallelScore: 0)
        }
        let units = zip(vectors, norms).map { $0.0 / $0.1 }
        let downwardCount = zip(units, extensions).filter { $0.0.y >= 0.50 && $0.1 >= 0.50 }.count
        var meanDirection = units.reduce(SIMD2<Float>(repeating: 0), +) / Float(units.count)
        let meanNorm = GestureFeatureExtractor.norm(meanDirection)
        let downwardScore: Float
        let parallelScore: Float
        if meanNorm < eps {
            downwardScore = 0
            parallelScore = 0
        } else {
            meanDirection /= meanNorm
            downwardScore = GestureFeatureExtractor.clamp((meanDirection.y - 0.30) / 0.60, 0, 1)
            parallelScore = GestureFeatureExtractor.clamp((meanNorm - 0.70) / 0.30, 0, 1)
        }
        let palmScale = max(GestureFeatureExtractor.dist(points[9], points[0]), 1e-6)
        let tips = [8, 12, 16, 20]
        let gaps = zip(tips.dropLast(), tips.dropFirst()).map {
            GestureFeatureExtractor.dist(points[$1], points[$0]) / palmScale
        }
        let together = GestureFeatureExtractor.clamp((0.78 - mean(gaps)) / 0.52, 0, 1)
        let extended = mean(extensions)
        let minimum = extensions.min() ?? 0
        let base = 0.48 * (0.70 * extended + 0.30 * minimum)
            + 0.27 * downwardScore + 0.15 * together + 0.10 * parallelScore
        let score = GestureFeatureExtractor.clamp(base * (0.75 + 0.25 * together), 0, 1)
        let strong = score >= 0.76 && downwardCount >= 4 && minimum >= 0.50
            && downwardScore >= 0.72 && together >= 0.18
        return ReturnMainGeometry(
            score: score,
            strongGeometry: strong,
            downwardFingerCount: downwardCount,
            minimumExtension: minimum,
            downwardScore: downwardScore,
            togetherScore: together,
            parallelScore: parallelScore
        )
    }

    private static func foreshortenedFistGeometry(_ points: [SIMD2<Float>]) throws -> FistGeometry {
        let wrist = points[0]
        let mcps = [5, 9, 13, 17].map { points[$0] }
        let tips = [8, 12, 16, 20].map { points[$0] }
        let palmCenter = mcps.reduce(SIMD2<Float>(repeating: 0), +) / 4
        let palmAxis = palmCenter - wrist
        let palmScale = GestureFeatureExtractor.norm(palmAxis)
        guard palmScale >= eps else {
            return FistGeometry(strongGeometry: false, retractedFingerCount: 0, meanCurlback: 0, maximumTipRadius: .infinity, thumbRadius: .infinity)
        }
        let palmUnit = palmAxis / palmScale
        let curlback = zip(mcps, tips).map {
            (simd_dot($0.0 - wrist, palmUnit) - simd_dot($0.1 - wrist, palmUnit)) / palmScale
        }
        let retractedCount = curlback.filter { $0 >= 0.05 }.count
        let meanCurlback = mean(curlback)
        let maximumTipRadius = tips.map { GestureFeatureExtractor.dist($0, palmCenter) / palmScale }.max() ?? .infinity
        let thumbRadius = GestureFeatureExtractor.dist(points[4], palmCenter) / palmScale
        let strong = retractedCount == 4 && meanCurlback >= 0.16
            && maximumTipRadius <= 0.85 && thumbRadius <= 0.85
        return FistGeometry(strongGeometry: strong, retractedFingerCount: retractedCount, meanCurlback: meanCurlback, maximumTipRadius: maximumTipRadius, thumbRadius: thumbRadius)
    }

    private static func handSurface(
        _ points: [SIMD2<Float>],
        handedness: String?,
        confidence: Float?
    ) -> String? {
        guard let hand = handedness, ["left", "right"].contains(hand),
              let confidence, confidence.isFinite, confidence >= 0.75 else { return nil }
        let indexPalm = points[5] - points[0]
        let pinkyPalm = points[17] - points[0]
        let denominator = GestureFeatureExtractor.norm(indexPalm) * GestureFeatureExtractor.norm(pinkyPalm)
        guard denominator >= eps else { return nil }
        let winding = (indexPalm.x * pinkyPalm.y - indexPalm.y * pinkyPalm.x) / denominator
        let dorsalScore = winding * (hand == "right" ? 1 : -1)
        if dorsalScore >= 0.25 { return "dorsal" }
        if dorsalScore <= -0.25 { return "palmar" }
        return nil
    }

    private static func extensionScore(
        _ points: [SIMD2<Float>], _ mcp: Int, _ pip: Int, _ dip: Int, _ tip: Int
    ) throws -> Float {
        try GestureFeatureExtractor.fingerExtensionScore(points: points, mcp: mcp, pip: pip, dip: dip, tip: tip)
    }

    private static func setProbabilityFloor(_ values: [Float], target: Int, floor: Float) -> [Float] {
        var adjusted = values
        if adjusted[target] < floor {
            let otherTotal = adjusted.reduce(0, +) - adjusted[target]
            if otherTotal > 1e-12 {
                let scale = (1 - floor) / otherTotal
                for i in adjusted.indices { adjusted[i] *= scale }
            } else {
                adjusted = Array(repeating: (1 - floor) / Float(max(1, adjusted.count - 1)), count: adjusted.count)
            }
            adjusted[target] = floor
        }
        return normalize(adjusted)
    }

    private static func normalize(_ values: [Float]) -> [Float] {
        let total = max(values.reduce(0, +), 1e-12)
        return values.map { $0 / total }
    }

    private static func argmax(_ values: [Float]) -> Int {
        values.indices.max(by: { values[$0] < values[$1] }) ?? 0
    }

    private static func index(_ name: String) -> Int {
        GestureFeatureExtractor.classNames.firstIndex(of: name)!
    }

    private static func mean(_ values: [Float]) -> Float {
        values.isEmpty ? 0 : values.reduce(0, +) / Float(values.count)
    }

    private static func spread(_ values: [Float]) -> Float {
        guard let low = values.min(), let high = values.max() else { return 0 }
        return high - low
    }

    /// Regression vectors copied from the Python camera-angle tests.
    /// This verifies both possible edge-finger assignments for the new Down rule.
    public static func runDownRecoverySelfTest() -> Bool {
        let fixtures: [[[Float]]] = [
            [[338,314],[365,322],[393,333],[412,346],[430,357],[307,282],[313,306],[315,313],[315,321],[340,273],[339,303],[339,311],[340,321],[372,269],[370,305],[369,315],[369,327],[404,275],[398,330],[400,348],[400,377]],
            [[338,314],[365,322],[393,333],[412,346],[430,357],[404,275],[398,330],[400,348],[400,377],[372,269],[370,305],[369,315],[369,327],[340,273],[339,303],[339,311],[340,321],[307,282],[313,306],[315,313],[315,321]],
        ]
        var probabilities = Array(repeating: Float(0.001), count: GestureFeatureExtractor.classNames.count)
        probabilities[index("ok")] = 0.993
        return fixtures.allSatisfy { fixture in
            let points = fixture.map { SIMD2<Float>($0[0], $0[1]) }
            guard let result = try? resolve(probabilities: probabilities, landmarks: points) else { return false }
            return result.valid && result.gesture == "down"
                && result.reversePalmEdgeDown && result.directionalRecovery
        }
    }
}
