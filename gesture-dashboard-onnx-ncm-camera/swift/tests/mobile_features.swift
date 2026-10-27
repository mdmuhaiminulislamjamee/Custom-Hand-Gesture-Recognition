import Foundation
import simd

struct MobileAngleExpected: Decodable {
    let xy_degrees: Float
    let open_palm: Bool
    let stop_inverted: Bool
}
struct MobileExpected: Decodable { let angle_policy: MobileAngleExpected }
struct MobileCase: Decodable {
    let id: Int
    let landmarks: [[Float]]
    let features: [Float]
    let image_shape: [Int]?
    let expected: MobileExpected
}
struct MobileVectors: Decodable { let geometry_cases: [MobileCase] }

@main struct MobileFeatureParity {
    static func main() throws {
        let vectors = try JSONDecoder().decode(MobileVectors.self,
            from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
        var maximum: Float = 0
        var angles = 0
        for item in vectors.geometry_cases {
            let points = item.landmarks.map { SIMD2<Float>($0[0], $0[1]) }
            let features = try GestureFeatureExtractor.landmarksToFeature(points)
            guard features.count == item.features.count else { fatalError("Feature length mismatch") }
            let error = zip(features, item.features).map { abs($0 - $1) }.max() ?? 0
            maximum = max(maximum, error)
            guard error <= 0.0001 else { fatalError("Feature mismatch case \(item.id): \(error)") }
            if let shape = item.image_shape {
                let xy = try GestureRuntimePolicy.openHandXY(points, imageWidth: shape[1], imageHeight: shape[0])
                let expected = item.expected.angle_policy
                guard abs(xy - expected.xy_degrees) <= 0.0002,
                    GestureRuntimePolicy.allowsAngle(xy, gesture: "open_palm") == expected.open_palm,
                    GestureRuntimePolicy.allowsAngle(xy, gesture: "stop_inverted") == expected.stop_inverted else {
                    fatalError("Angle mismatch case \(item.id)")
                }
                angles += 1
            }
        }
        print("PASS: \(vectors.geometry_cases.count) feature cases, \(angles) angle cases, maximum feature error \(maximum)")
        print("Full native geometry/temporal/action parity still requires the current decision port.")
    }
}
