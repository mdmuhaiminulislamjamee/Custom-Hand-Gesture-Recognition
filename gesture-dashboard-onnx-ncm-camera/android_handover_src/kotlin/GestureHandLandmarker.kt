package com.example.gesturehandover

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult

/** Filtered observations only. Raw candidates cannot reach inference or overlays. */
class GestureFilteredHandResult(private val raw: HandLandmarkerResult, val acceptedIndices: List<Int>, val rejections: Map<Int, String>) {
    fun landmarks() = acceptedIndices.map { raw.landmarks()[it] }
    fun worldLandmarks() = acceptedIndices.map { raw.worldLandmarks().getOrNull(it) ?: emptyList() }
    fun handedness() = acceptedIndices.map { raw.handedness().getOrNull(it) ?: emptyList() }
}

/** Diagnostic two-hand detector; ownership and stabilization still need porting. */
class GestureHandLandmarker(context: Context) : AutoCloseable {
    private val detector: HandLandmarker
    private var lastTimestampMs = -1L
    private val faceGuard = GestureFalseHandGuard(context)

    init {
        val options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build())
            .setRunningMode(RunningMode.VIDEO)
            .setNumHands(2)
            .setMinHandDetectionConfidence(0.55f)
            .setMinHandPresenceConfidence(0.55f)
            .setMinTrackingConfidence(0.60f)
            .build()
        detector = HandLandmarker.createFromOptions(context, options)
    }

    fun detect(bitmap: Bitmap, timestampMs: Long): GestureFilteredHandResult {
        require(timestampMs > lastTimestampMs) { "MediaPipe timestamps must increase" }
        lastTimestampMs = timestampMs
        val argb = if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap
            else bitmap.copy(Bitmap.Config.ARGB_8888, false)
        val image = BitmapImageBuilder(argb).build()
        try {
            val faces = faceGuard.faces(image, timestampMs)
            val result = detector.detectForVideo(image, timestampMs)
            val rejected = mutableMapOf<Int,String>()
            val accepted = result.landmarks().indices.filter { index ->
                val points = result.landmarks()[index].map { GesturePoint(it.x(), it.y()) }
                val world = result.worldLandmarks().getOrNull(index)?.map { GestureWorldPoint(it.x(), it.y(), it.z()) }
                val reason = GestureFalseHandPolicy.rejection(points, world, argb.width, argb.height, faces)
                if (reason != null) rejected[index] = reason
                reason == null
            }
            return GestureFilteredHandResult(result, accepted, rejected)
        } finally { image.close(); if (argb !== bitmap) argb.recycle() }
    }

    override fun close() { detector.close(); faceGuard.close() }
}
