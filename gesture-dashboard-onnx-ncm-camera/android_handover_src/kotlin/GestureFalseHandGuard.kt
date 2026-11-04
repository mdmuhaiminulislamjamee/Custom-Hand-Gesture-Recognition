package com.example.gesturehandover

import android.content.Context
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facedetector.FaceDetector
import kotlin.math.sqrt

data class GestureFaceBox(val x: Float, val y: Float, val width: Float, val height: Float)

/** Pure candidate policy. Landmarks are normalized in the prepared frame. */
object GestureFalseHandPolicy {
    private fun median(values: List<Double>): Double {
        val s = values.sorted()
        return if (s.size % 2 == 1) s[s.size / 2] else (s[s.size / 2 - 1] + s[s.size / 2]) / 2
    }
    private fun distance(a: List<Double>, b: List<Double>) = sqrt(a.indices.sumOf { (a[it] - b[it]) * (a[it] - b[it]) })
    private fun collapsed(p: List<List<Double>>): Boolean {
        val palm = median(listOf(5, 9, 13, 17).map { distance(p[it], p[0]) })
        if (!palm.isFinite() || palm < 1e-6) return true
        return listOf(5, 9, 13, 17).count { s -> (0..2).sumOf { distance(p[s + it], p[s + it + 1]) } < .12 * palm } >= 3
    }
    fun rejection(points: List<GesturePoint>, world: List<GestureWorldPoint>?, width: Int, height: Int, faces: List<GestureFaceBox>): String? {
        require(width > 0 && height > 0)
        if (points.size != 21 || points.any { !it.x.isFinite() || !it.y.isFinite() }) return "incomplete_hand"
        val p = points.map { listOf(it.x.toDouble() * width, it.y.toDouble() * height) }
        val dx = p.maxOf { it[0] } - p.minOf { it[0] }
        val dy = p.maxOf { it[1] } - p.minOf { it[1] }
        val palm = median(listOf(0 to 5, 0 to 9, 0 to 17, 5 to 17).map { (a,b) -> distance(p[a], p[b]) })
        var reason: String? = if (palm < 18 || maxOf(dx,dy) < 40 || minOf(dx,dy) < 12) "insufficient_hand_pixels" else null
        if (p.count { it[0] >= -.05 * width && it[0] <= 1.05 * width && it[1] >= -.05 * height && it[1] <= 1.05 * height } / 21.0 < .9) reason = reason ?: "incomplete_hand"
        if (reason == null && collapsed(p)) {
            val valid = world?.takeIf { it.size == 21 && it.all { v -> v.x.isFinite() && v.y.isFinite() && v.z.isFinite() } }
            if (valid == null || collapsed(valid.map { listOf(it.x.toDouble(), it.y.toDouble(), it.z.toDouble()) })) reason = "invalid_hand_geometry"
        }
        val center = listOf(0,5,9,13,17).let { ids -> listOf(ids.sumOf { p[it][0] } / 5, ids.sumOf { p[it][1] } / 5) }
        for (f in faces) {
            if (f.width <= 0 || f.height <= 0) continue
            fun covers(v: List<Double>) = v[0] >= f.x - .3 * f.width && v[0] <= f.x + 1.3 * f.width && v[1] >= f.y - .15 * f.height && v[1] <= f.y + 1.05 * f.height
            if (covers(center) && maxOf(dx,dy) < .7 * maxOf(f.width,f.height) && p.count(::covers) / 21.0 >= .75) return "face_detected_as_hand"
        }
        return reason
    }
}

/** One instance per serial camera stream. Cache uses the capture clock. */
class GestureFalseHandGuard(context: Context) : AutoCloseable {
    private val detector = FaceDetector.createFromOptions(context, FaceDetector.FaceDetectorOptions.builder()
        .setBaseOptions(BaseOptions.builder().setModelAssetPath("blaze_face_short_range.tflite").build())
        .setRunningMode(RunningMode.IMAGE).setMinDetectionConfidence(.65f).build())
    private var lastAttempt = Long.MIN_VALUE
    private var shape = 0 to 0
    private val recent = mutableListOf<Pair<GestureFaceBox, Long>>()
    fun reset() { lastAttempt = Long.MIN_VALUE; recent.clear(); shape = 0 to 0 }
    fun faces(image: MPImage, timestampMs: Long): List<GestureFaceBox> {
        if (shape != (image.width to image.height) || timestampMs < lastAttempt) reset()
        shape = image.width to image.height
        recent.removeAll { timestampMs - it.second > 500 || timestampMs < it.second }
        if (lastAttempt != Long.MIN_VALUE && timestampMs - lastAttempt < 200) return recent.map { it.first }
        lastAttempt = timestampMs
        try {
            val fresh = detector.detect(image).detections().map { d -> d.boundingBox().let { GestureFaceBox(it.left, it.top, it.width(), it.height()) } }
            recent.removeAll { (old, _) -> fresh.any { box ->
                val overlap = maxOf(0f, minOf(old.x + old.width, box.x + box.width) - maxOf(old.x,box.x)) *
                    maxOf(0f, minOf(old.y + old.height, box.y + box.height) - maxOf(old.y,box.y))
                overlap > .3f * minOf(old.width * old.height, box.width * box.height)
            } }
            recent.addAll(fresh.map { it to timestampMs })
        } catch (error: Exception) { reset(); throw error }
        return recent.map { it.first }
    }
    override fun close() = detector.close()
}
