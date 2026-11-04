import Foundation
import simd

public struct GestureFaceBox {
    public let x, y, width, height: Float
    public init(x: Float, y: Float, width: Float, height: Float) {
        self.x = x; self.y = y; self.width = width; self.height = height
    }
}

/// Normalized image XY plus optional metric world XYZ from the same candidate.
public enum GestureFalseHandPolicy {
    private static func median(_ values: [Float]) -> Float {
        let v = values.sorted(), n = values.count
        return n % 2 == 1 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2
    }
    private static func collapsed(_ p: [SIMD3<Float>]) -> Bool {
        let palm = median([5,9,13,17].map { simd_distance(p[$0], p[0]) })
        if !palm.isFinite || palm < 1e-6 { return true }
        return [5,9,13,17].filter { start in
            (0..<3).map { simd_distance(p[start + $0], p[start + $0 + 1]) }.reduce(0,+) < 0.12 * palm
        }.count >= 3
    }
    public static func rejection(landmarks: [SIMD2<Float>], world: [SIMD3<Float>]?,
                                 width: Int, height: Int, faces: [GestureFaceBox]) -> String? {
        guard width > 0, height > 0, landmarks.count == 21,
              landmarks.allSatisfy({ $0.x.isFinite && $0.y.isFinite }) else { return "incomplete_hand" }
        let w = Float(width), h = Float(height)
        let p = landmarks.map { SIMD3<Float>($0.x * w, $0.y * h, 0) }
        let dx = p.map(\.x).max()! - p.map(\.x).min()!
        let dy = p.map(\.y).max()! - p.map(\.y).min()!
        let palm = median([(0,5),(0,9),(0,17),(5,17)].map { simd_distance(p[$0.0],p[$0.1]) })
        var reason: String? = palm < 18 || max(dx,dy) < 40 || min(dx,dy) < 12 ? "insufficient_hand_pixels" : nil
        if Float(p.filter({ $0.x >= -0.05 * w && $0.x <= 1.05 * w && $0.y >= -0.05 * h && $0.y <= 1.05 * h }).count) / 21 < 0.9 {
            reason = reason ?? "incomplete_hand"
        }
        if reason == nil && collapsed(p) {
            if let world, world.count == 21, world.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }) {
                if collapsed(world) { reason = "invalid_hand_geometry" }
            } else { reason = "invalid_hand_geometry" }
        }
        let center = [0,5,9,13,17].map { p[$0] }.reduce(SIMD3<Float>(repeating: 0), +) / 5
        for f in faces where f.width > 0 && f.height > 0 {
            func covers(_ v: SIMD3<Float>) -> Bool {
                v.x >= f.x - 0.3 * f.width && v.x <= f.x + 1.3 * f.width &&
                v.y >= f.y - 0.15 * f.height && v.y <= f.y + 1.05 * f.height
            }
            if covers(center) && max(dx,dy) < 0.7 * max(f.width,f.height) && Float(p.filter(covers).count) / 21 >= 0.75 {
                return "face_detected_as_hand"
            }
        }
        return reason
    }
}
