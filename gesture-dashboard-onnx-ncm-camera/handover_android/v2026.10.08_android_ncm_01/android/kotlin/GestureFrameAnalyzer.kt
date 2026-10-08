package com.example.gesturehandover

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.SystemClock

data class GestureObservation(
    val landmarks: List<GesturePoint>,
    val probabilities: FloatArray,
    val knownGestureMass: Float,
    val timestampMs: Long,
    val boardSequence: Long,
)

/**
 * Diagnostic NCM-to-model path. It emits raw scores only. The production action
 * path must add the complete geometry, face/recovery, temporal and latch rules.
 */
class GestureFrameAnalyzer(
    context: Context,
    private val onObservation: (GestureObservation) -> Unit,
    private val onMissingHand: () -> Unit,
    private val onError: (Throwable) -> Unit,
) : AutoCloseable {
    private val detector = GestureHandLandmarker(context)
    private val classifier = GestureOnnxClassifier(context)
    private val lock = Object()
    private var pending: NcmJpeg? = null
    @Volatile private var running = true
    private var lastStartMs = -100L
    private var lastMediaPipeMs = -1L
    private val worker = Thread({ loop() }, "gesture-inference").also { it.start() }

    fun submit(frame: NcmJpeg) = synchronized(lock) {
        if (running) { pending = frame; lock.notifyAll() }
    }

    private fun loop() {
        while (running) {
            val frame = synchronized(lock) {
                while (running && pending == null) lock.wait()
                if (!running) null else {
                    val delay = lastStartMs + 100 - SystemClock.elapsedRealtime()
                    if (delay > 0) lock.wait(delay)
                    pending.also { pending = null }
                }
            } ?: continue
            try {
                val now = SystemClock.elapsedRealtime()
                lastStartMs = now
                val decoded = BitmapFactory.decodeByteArray(frame.bytes, 0, frame.bytes.size)
                    ?: error("Invalid NCM JPEG")
                val prepared = centerCrop(decoded)
                val timestamp = maxOf(now, lastMediaPipeMs + 1)
                lastMediaPipeMs = timestamp
                val result = detector.detect(prepared, timestamp)
                val hand = result.landmarks().firstOrNull()
                if (hand == null || hand.size != 21) { onMissingHand(); continue }
                val points = hand.map { GesturePoint(it.x(), it.y()) }
                val model = classifier.predict(GestureFeatureExtractor.extract(points))
                onObservation(GestureObservation(points, model.probabilities,
                    model.knownGestureMass, timestamp, frame.boardSequence))
            } catch (error: Exception) {
                onError(error)
            }
        }
    }

    private fun centerCrop(bitmap: Bitmap): Bitmap {
        val edge = (minOf(bitmap.width, bitmap.height) * 0.92f).toInt().coerceAtLeast(1)
        val square = Bitmap.createBitmap(bitmap, (bitmap.width - edge) / 2,
            (bitmap.height - edge) / 2, edge, edge)
        return if (edge > 960) Bitmap.createScaledBitmap(square, 960, 960, true) else square
    }

    override fun close() {
        synchronized(lock) { running = false; pending = null; lock.notifyAll() }
        worker.join()
        detector.close()
        classifier.close()
    }
}
