import Foundation
import MediaPipeTasksVision

/// Own on the same serial queue as the hand detector. All boxes use its pixels.
public final class GestureFaceGuard {
    private let detector: FaceDetector
    private var recent: [(GestureFaceBox, Int)] = []
    private var attemptedAt: Int?
    private var shape = (0, 0)
    public init(modelURL: URL) throws {
        let options = FaceDetectorOptions()
        options.baseOptions.modelAssetPath = modelURL.path
        options.runningMode = .image
        options.minDetectionConfidence = 0.65
        detector = try FaceDetector(options: options)
    }
    public func reset() { recent.removeAll(); attemptedAt = nil; shape = (0, 0) }
    public func faces(image: MPImage, timestampMS: Int) throws -> [GestureFaceBox] {
        if shape != (image.width, image.height) || (attemptedAt.map { timestampMS < $0 } ?? false) { reset() }
        shape = (image.width, image.height)
        recent.removeAll { timestampMS < $0.1 || timestampMS - $0.1 > 500 }
        if let attemptedAt, timestampMS - attemptedAt < 200 { return recent.map { $0.0 } }
        attemptedAt = timestampMS
        do {
            let fresh = try detector.detect(image: image).detections.map { detection -> GestureFaceBox in
                let b = detection.boundingBox
                return GestureFaceBox(x: Float(b.minX), y: Float(b.minY), width: Float(b.width), height: Float(b.height))
            }
            recent.removeAll { old in fresh.contains { b in
                let a = old.0
                let overlap = max(0, min(a.x+a.width,b.x+b.width)-max(a.x,b.x)) * max(0, min(a.y+a.height,b.y+b.height)-max(a.y,b.y))
                return overlap > 0.3 * min(a.width*a.height,b.width*b.height)
            } }
            recent.append(contentsOf: fresh.map { ($0, timestampMS) })
            return recent.map { $0.0 }
        } catch { reset(); throw error }
    }
}
