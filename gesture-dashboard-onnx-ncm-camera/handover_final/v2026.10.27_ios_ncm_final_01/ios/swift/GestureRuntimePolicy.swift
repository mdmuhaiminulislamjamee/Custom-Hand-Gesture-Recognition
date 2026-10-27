import Foundation
import simd

/// Current image-plane angle limits only. This is not a geometry/temporal/action gate.
public enum GestureRuntimePolicy {
    public static let openPalmXY: ClosedRange<Float> = -160 ... -20
    public static let stopInvertedXY: ClosedRange<Float> = -145 ... -35
    public static let disabledGestures: Set<String> = ["dorsal"]

    public static func openHandXY(_ points: [SIMD2<Float>], imageWidth: Int, imageHeight: Int) throws -> Float {
        guard points.count == 21, imageWidth > 0, imageHeight > 0,
              points.allSatisfy({ $0.x.isFinite && $0.y.isFinite }) else {
            throw GestureFeatureError.invalidLandmarkShape
        }
        let knuckles = [5, 9, 13, 17].reduce(SIMD2<Float>(repeating: 0)) { $0 + points[$1] } / 4
        let axis = (knuckles - points[0]) * SIMD2<Float>(Float(imageWidth), Float(imageHeight))
        guard simd_length(axis) > 1e-8 else {
            throw GestureFeatureError.degenerateGeometry("Missing wrist-to-knuckle direction")
        }
        let degrees = atan2(axis.y, axis.x) * 180 / Float.pi
        return abs(degrees - 180) <= 0.0001 ? -180 : degrees
    }

    public static func allowsAngle(_ xy: Float, gesture: String) -> Bool {
        guard xy.isFinite else { return false }
        let band: ClosedRange<Float>
        switch gesture {
        case "open_palm": band = openPalmXY
        case "stop_inverted": band = stopInvertedXY
        default: return false
        }
        return xy >= band.lowerBound - 0.0001 && xy <= band.upperBound + 0.0001
    }
}
