import Foundation
import simd

struct Expected: Decodable {
    let probabilities: [Float], gesture: String, valid: Bool
    let geometry_supported: Bool, directional_recovery: Bool
    let sideways_palm_recovery: Bool, dorsal_range_recovery: Bool
    let temporal: [Temporal]
}
struct Temporal: Decodable { let mass: Float, execute: [Bool] }
struct Case: Decodable {
    let id: Int, source: String, probabilities: [Float], landmarks: [[Float]], features: [Float]
    let handedness: String?, handedness_confidence: Float?, world_landmarks: [[Float]]?
    let require_fist_depth: Bool, expected: Expected
}
struct Vectors: Decodable { let geometry_cases: [Case] }
func check(_ pass: Bool, _ message: String) {
    if !pass { fputs("FAIL: " + message + "\n", stderr); exit(1) }
}
func difference(_ a: [Float], _ b: [Float]) -> Float {
    if a.count != b.count { return .infinity }
    return zip(a,b).map { abs($0 - $1) }.max() ?? 0
}
let url = URL(fileURLWithPath: CommandLine.arguments[1])
let vectors = try JSONDecoder().decode(Vectors.self, from: Data(contentsOf: url))
var featureError: Float = 0, probabilityError: Float = 0
for item in vectors.geometry_cases {
    let points = item.landmarks.map { SIMD2<Float>($0[0], $0[1]) }
    let world = item.world_landmarks?.map { SIMD3<Float>($0[0], $0[1], $0[2]) }
    let features = try GestureFeatureExtractor.landmarksToFeature(points)
    featureError = max(featureError, difference(features, item.features))
    check(difference(features,item.features) <= 0.0001, "features case \(item.id): \(item.source)")
    let resolved = try GestureGeometryResolver.resolve(probabilities: item.probabilities, landmarks: points,
        handedness: item.handedness, handednessConfidence: item.handedness_confidence,
        worldLandmarks: world, requireFistDepth: item.require_fist_depth)
    let expected = item.expected
    probabilityError = max(probabilityError, difference(resolved.probabilities, expected.probabilities))
    check(difference(resolved.probabilities,expected.probabilities) <= 0.0001, "probabilities case \(item.id)")
    check(resolved.gesture == expected.gesture && resolved.valid == expected.valid, "decision case \(item.id): \(item.source)")
    check(resolved.geometrySupported == expected.geometry_supported, "geometry support case \(item.id)")
    check(resolved.directionalRecovery == expected.directional_recovery, "direction recovery case \(item.id)")
    check(resolved.sidewaysPalmRecovery == expected.sideways_palm_recovery, "palm recovery case \(item.id)")
    check(resolved.dorsalRangeRecovery == expected.dorsal_range_recovery, "dorsal recovery case \(item.id)")
    for temporal in expected.temporal {
        let gate = GestureTemporalGate()
        for (i, execute) in temporal.execute.enumerated() {
            let decision = gate.update(resolved, knownGestureMass: temporal.mass, timestamp: 10 + Double(i) / 10)
            check(decision.execute == execute, "temporal case \(item.id), mass \(temporal.mass), frame \(i)")
        }
    }
}
// A tracking dropout has a six-frame release threshold at 10 FPS.
let latch = GestureActionLatch()
let accepted = GestureTemporalDecision(execute: true, gesture: "like", confidence: 1, stableFrames: 3, heldSeconds: 1, reason: "test")
let missing = GestureTemporalDecision(execute: false, gesture: "no_gesture", confidence: 0, stableFrames: 0, heldSeconds: 0, reason: "no hand")
check(latch.update(accepted, timestamp: 10) == "Play / Pause", "initial action")
for i in 1...5 { check(latch.update(missing, timestamp: 10 + Double(i)/10, trackingGap: true) == nil, "gap emits no action") }
check(latch.update(accepted, timestamp: 11) == nil, "short dropout does not rearm")
for i in 1...6 { _ = latch.update(missing, timestamp: 11 + Double(i)/10, trackingGap: true) }
check(latch.update(accepted, timestamp: 12) == "Play / Pause", "sustained release rearms")
print("PASS: \(vectors.geometry_cases.count) geometry/feature/temporal cases; feature max error \(featureError); probability max error \(probabilityError)")
