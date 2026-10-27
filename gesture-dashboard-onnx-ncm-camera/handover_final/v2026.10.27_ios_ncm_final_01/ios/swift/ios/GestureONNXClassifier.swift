import Foundation
import CryptoKit
// ORT types are exposed by the supplied Objective-C bridging header.

public enum GestureIntegrationError: Error {
    case invalidContract(String)
}

public struct GestureModelOutput {
    public let probabilities: [Float]
    public let knownGestureMass: Float
}

/// Keep one instance per serialized inference queue. Uses CPU execution.
public final class GestureONNXClassifier {
    public let modelSHA256: String
    private let environment: ORTEnv
    private let session: ORTSession

    public init(modelsDirectory: URL) throws {
        let metadataURL = modelsDirectory.appendingPathComponent("gesture_mlp_onnx_metadata.json")
        let runtimeURL = modelsDirectory.appendingPathComponent("gesture_mobile_runtime_config.json")
        let metadata = try JSONSerialization.jsonObject(with: Data(contentsOf: metadataURL)) as? [String: Any]
        let runtime = try JSONSerialization.jsonObject(with: Data(contentsOf: runtimeURL)) as? [String: Any]
        let modelURL = modelsDirectory.appendingPathComponent("gesture_mlp_production.onnx")
        let digest = SHA256.hash(data: try Data(contentsOf: modelURL)).map { String(format: "%02x", $0) }.joined()
        modelSHA256 = digest
        guard metadata?["onnx_sha256"] as? String == digest,
              metadata?["output_class_order"] as? [String] == GestureFeatureExtractor.classNames,
              runtime?["class_names"] as? [String] == GestureFeatureExtractor.classNames,
              runtime?["gesture_to_action"] as? [String: String] == GestureFeatureExtractor.gestureToAction else {
            throw GestureIntegrationError.invalidContract("Mixed model, metadata, or class order")
        }
        // The geometry port uses the dated release constants. Fail if a different
        // temporal configuration is bundled instead of silently using defaults.
        let temporal = runtime?["temporal_smoothing"] as? [String: Any]
        for (key, expected) in ["alpha": 0.45, "confidence_floor": 0.70,
                                "probability_margin_floor": 0.08, "known_mass_floor": 0.70,
                                "geometry_recovery_mass_floor": 0.15, "stable_frames_required": 3.0,
                                "release_frames_required": 3.0, "minimum_hold_seconds": 0.18] {
            guard let value = temporal?[key] as? NSNumber, abs(value.doubleValue - expected) < 1e-7 else {
                throw GestureIntegrationError.invalidContract("Unsupported setting: " + key)
            }
        }
        guard (runtime?["action_cooldown_seconds"] as? NSNumber)?.doubleValue == 0.8,
              (runtime?["target_fps"] as? NSNumber)?.doubleValue == 10 else {
            throw GestureIntegrationError.invalidContract("Unsupported timing configuration")
        }
        let env = try ORTEnv(loggingLevel: .warning)
        let options = try ORTSessionOptions()
        try options.setIntraOpNumThreads(1)
        let created = try ORTSession(env: env, modelPath: modelURL.path, sessionOptions: options)
        guard try created.inputNames() == ["landmark_features"],
              Set(try created.outputNames()).isSuperset(of: ["probabilities", "known_gesture_mass"]) else {
            throw GestureIntegrationError.invalidContract("Unexpected tensor names")
        }
        environment = env
        session = created
        _ = try predict(features: Array(repeating: 0, count: 76))
    }

    public func predict(features: [Float]) throws -> GestureModelOutput {
        guard features.count == 76, features.allSatisfy({ $0.isFinite }) else {
            throw GestureIntegrationError.invalidContract("Expected 76 finite Float32 features")
        }
        let bytes = features.withUnsafeBytes { NSMutableData(bytes: $0.baseAddress!, length: $0.count) }
        let input = try ORTValue(tensorData: bytes, elementType: .float, shape: [1, 76])
        let outputs = try session.run(withInputs: ["landmark_features": input],
                                      outputNames: ["probabilities", "known_gesture_mass"], runOptions: nil)
        let probabilities = try floats(outputs["probabilities"], shape: [1, GestureFeatureExtractor.classNames.count])
        let mass = try floats(outputs["known_gesture_mass"], shape: [1,1])[0]
        guard probabilities.allSatisfy({ $0 >= 0 && $0 <= 1.00001 }),
              abs(probabilities.reduce(0,+) - 1) < 0.0001, (0...1).contains(mass) else {
            throw GestureIntegrationError.invalidContract("Invalid output probabilities")
        }
        return GestureModelOutput(probabilities: probabilities, knownGestureMass: mass)
    }

    private func floats(_ value: ORTValue?, shape: [Int]) throws -> [Float] {
        guard let value else { throw GestureIntegrationError.invalidContract("Missing named output") }
        let info = try value.tensorTypeAndShapeInfo()
        guard info.elementType == .float, info.shape.map({ $0.intValue }) == shape else {
            throw GestureIntegrationError.invalidContract("Wrong tensor type or shape")
        }
        let data = try value.tensorData()
        let count = shape.reduce(1, *)
        guard data.length == count * MemoryLayout<Float>.size else {
            throw GestureIntegrationError.invalidContract("Wrong tensor byte count")
        }
        var result = [Float](repeating: 0, count: count)
        result.withUnsafeMutableBytes { data.getBytes($0.baseAddress!, length: data.length) }
        guard result.allSatisfy({ $0.isFinite }) else {
            throw GestureIntegrationError.invalidContract("Nonfinite tensor output")
        }
        return result
    }
}
