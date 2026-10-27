package com.example.gesturehandover

import kotlin.math.atan2
import kotlin.math.hypot

/** Image-plane angle limits only; this does not authorize a gesture action. */
object GestureRuntimePolicy {
    val OPEN_PALM_XY = -160f..-20f
    val STOP_INVERTED_XY = -145f..-35f
    val DISABLED_GESTURES = setOf("dorsal")

    fun openHandXY(points: List<GesturePoint>, imageWidth: Int, imageHeight: Int): Float {
        require(points.size == 21 && imageWidth > 0 && imageHeight > 0)
        require(points.all { it.x.isFinite() && it.y.isFinite() })
        val dx = (listOf(5, 9, 13, 17).sumOf { points[it].x.toDouble() } / 4 - points[0].x) * imageWidth
        val dy = (listOf(5, 9, 13, 17).sumOf { points[it].y.toDouble() } / 4 - points[0].y) * imageHeight
        require(hypot(dx, dy) > 1e-8) { "Missing wrist-to-knuckle direction" }
        val degrees = Math.toDegrees(atan2(dy, dx)).toFloat()
        return if (kotlin.math.abs(degrees - 180f) <= .0001f) -180f else degrees
    }

    fun allowsAngle(xy: Float, gesture: String): Boolean {
        if (!xy.isFinite()) return false
        val band = when (gesture) {
            "open_palm" -> OPEN_PALM_XY
            "stop_inverted" -> STOP_INVERTED_XY
            else -> return false
        }
        return xy >= band.start - .0001f && xy <= band.endInclusive + .0001f
    }
}
