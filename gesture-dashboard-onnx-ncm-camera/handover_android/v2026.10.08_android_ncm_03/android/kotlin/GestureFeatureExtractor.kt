package com.example.gesturehandover

import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

data class GesturePoint(val x: Float, val y: Float) {
    operator fun minus(other: GesturePoint) = GesturePoint(x - other.x, y - other.y)
    operator fun div(value: Float) = GesturePoint(x / value, y / value)
}

/** Matches the 76-value feature contract in backend/geometry.py. */
object GestureFeatureExtractor {
    private const val EPS = 1e-8f

    fun extract(input: List<GesturePoint>): FloatArray {
        require(input.size == 21 && input.all { it.x.isFinite() && it.y.isFinite() }) {
            "Expected 21 finite normalized image landmarks"
        }
        val refs = palmReferences(input)
        val palmScale = median(refs)
        val pinchGap = distance(input[4], input[8]) / palmScale
        val extensions = floatArrayOf(
            extension(input, 1, 2, 3, 4),
            extension(input, 5, 6, 7, 8),
            extension(input, 9, 10, 11, 12),
            extension(input, 13, 14, 15, 16),
            extension(input, 17, 18, 19, 20),
        )
        val shape = floatArrayOf(
            extensions[0], extensions[1], extensions[2], extensions[3], extensions[4],
            1f - (extensions[2] + extensions[3] + extensions[4]) / 3f,
            distance(input[4], input[5]) / palmScale,
            distance(input[4], input[0]) / palmScale,
        )
        val wrist = input[0]
        val centered = input.map { it - wrist }
        val indexSegment = unit(input[8] - input[5])
        val wristIndex = unit(centered[8])
        val middle = centered[9]
        require(norm(middle) >= EPS) { "Degenerate wrist-to-middle geometry" }
        val angle = -Math.PI.toFloat() / 2f - atan2(middle.y, middle.x)
        val c = cos(angle)
        val s = sin(angle)
        var rotated = centered.map { GesturePoint(c * it.x - s * it.y, s * it.x + c * it.y) }
        val scale = rotated.maxOf { norm(it) }
        require(scale >= EPS) { "Degenerate landmark scale" }
        rotated = rotated.map { it / scale }
        if (rotated[5].x < rotated[17].x) {
            rotated = rotated.map { GesturePoint(-it.x, it.y) }
        }
        val output = ArrayList<Float>(76)
        rotated.forEach { output.add(it.x); output.add(it.y) }
        rotated.forEach { output.add(norm(it)) }
        output.add(indexSegment.x); output.add(indexSegment.y)
        output.add(wristIndex.x); output.add(wristIndex.y)
        output.add(pinchGap)
        shape.forEach { output.add(it) }
        check(output.size == 76 && output.all { it.isFinite() })
        return output.toFloatArray()
    }

    private fun palmReferences(p: List<GesturePoint>): List<Float> {
        val values = listOf(distance(p[5], p[0]), distance(p[9], p[0]),
            distance(p[17], p[0]), distance(p[17], p[5])).filter { it > EPS }
        require(values.isNotEmpty()) { "Degenerate palm scale" }
        return values
    }

    private fun extension(p: List<GesturePoint>, mcp: Int, pip: Int, dip: Int, tip: Int): Float {
        val mean = (angle(p[mcp], p[pip], p[dip]) + angle(p[pip], p[dip], p[tip])) * 0.5f
        return ((mean - 110f) / 60f).coerceIn(0f, 1f)
    }

    private fun angle(a: GesturePoint, b: GesturePoint, c: GesturePoint): Float {
        val ab = a - b
        val cb = c - b
        val denominator = norm(ab) * norm(cb)
        if (denominator < EPS) return 0f
        val cosine = ((ab.x * cb.x + ab.y * cb.y) / denominator).coerceIn(-1f, 1f)
        return (acos(cosine) * 180f / Math.PI.toFloat())
    }

    private fun unit(p: GesturePoint): GesturePoint {
        val magnitude = norm(p)
        require(magnitude >= EPS) { "Degenerate direction" }
        return p / magnitude
    }

    private fun distance(a: GesturePoint, b: GesturePoint) = norm(a - b)
    private fun norm(p: GesturePoint) = hypot(p.x, p.y)
    private fun median(values: List<Float>): Float {
        val sorted = values.sorted()
        return if (sorted.size % 2 == 1) sorted[sorted.size / 2]
        else (sorted[sorted.size / 2 - 1] + sorted[sorted.size / 2]) / 2f
    }
}
