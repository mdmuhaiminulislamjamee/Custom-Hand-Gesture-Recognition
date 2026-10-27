import Foundation

public enum GestureONNXParity {
    private struct Row: Decodable {
        let features: [Float], probabilities: [Float], known_gesture_mass: Float
    }
    private struct Vectors: Decodable { let model_sha256: String; let model_cases: [Row] }
    /// Call once during integration before camera actions are enabled.
    public static func run(classifier: GestureONNXClassifier, vectorsURL: URL) throws -> Float {
        let vectors = try JSONDecoder().decode(Vectors.self, from: Data(contentsOf: vectorsURL))
        guard vectors.model_sha256 == classifier.modelSHA256, !vectors.model_cases.isEmpty else {
            throw GestureIntegrationError.invalidContract("Wrong or empty model parity fixtures")
        }
        var maximum: Float = 0
        for row in vectors.model_cases {
            let output = try classifier.predict(features: row.features)
            guard output.probabilities.count == row.probabilities.count else {
                throw GestureIntegrationError.invalidContract("Parity output length mismatch")
            }
            let error = max(zip(output.probabilities,row.probabilities).map { abs($0 - $1) }.max() ?? 0,
                            abs(output.knownGestureMass - row.known_gesture_mass))
            maximum = max(maximum,error)
            guard error <= 0.0001 else { throw GestureIntegrationError.invalidContract("ONNX parity failed") }
        }
        return maximum
    }
}
