package com.example.gesturehandover

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult

/** Diagnostic two-hand detector; recovery, ownership and stabilization still need porting. */
class GestureHandLandmarker(context: Context) : AutoCloseable {
    private val detector: HandLandmarker
    private var lastTimestampMs = -1L

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

    fun detect(bitmap: Bitmap, timestampMs: Long): HandLandmarkerResult {
        require(timestampMs > lastTimestampMs) { "MediaPipe timestamps must increase" }
        lastTimestampMs = timestampMs
        val argb = if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap
            else bitmap.copy(Bitmap.Config.ARGB_8888, false)
        return detector.detectForVideo(BitmapImageBuilder(argb).build(), timestampMs)
    }

    override fun close() = detector.close()
}
