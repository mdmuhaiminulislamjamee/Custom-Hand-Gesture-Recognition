package com.example.gesturehandover

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.SystemClock

data class GestureWorldPoint(val x: Float, val y: Float, val z: Float)
data class GestureHandObservation(
    val landmarks: List<GesturePoint>,
    val worldLandmarks: List<GestureWorldPoint>?,
    val handedness: String?,
    val handednessConfidence: Float?,
    val probabilities: FloatArray,
    val knownGestureMass: Float,
)

data class GestureObservation(
    val landmarks: List<GesturePoint>,
    val probabilities: FloatArray,
    val knownGestureMass: Float,
    val timestampMs: Long,
    val boardSequence: Long,
    val imageWidth: Int,
    val imageHeight: Int,
    val candidates: List<GestureHandObservation>,
)

/**
 * Diagnostic NCM-to-model path. It emits raw scores only. The production action
 * path must add hand selection, recovery, geometry, temporal and latch rules.
 * Face and body vetoes are disabled in the current reference.
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

    fun clearPending() = synchronized(lock) { pending = null }

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
                val candidates = result.landmarks().mapIndexedNotNull { index, hand ->
                    if (hand.size != 21) return@mapIndexedNotNull null
                    val points = hand.map { GesturePoint(it.x(), it.y()) }
                    val model = classifier.predict(GestureFeatureExtractor.extract(points))
                    val category = result.handedness().getOrNull(index)?.firstOrNull()
                    val world = result.worldLandmarks().getOrNull(index)?.map {
                        GestureWorldPoint(it.x(), it.y(), it.z())
                    }
                    GestureHandObservation(points, world, category?.categoryName(), category?.score(),
                        model.probabilities, model.knownGestureMass)
                }
                if (candidates.isEmpty()) { onMissingHand(); continue }
                // The first candidate is a diagnostic convenience, not gesture ownership.
                val first = candidates.first()
                onObservation(GestureObservation(first.landmarks, first.probabilities,
                    first.knownGestureMass, timestamp, frame.boardSequence,
                    prepared.width, prepared.height, candidates))
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
        if (Thread.currentThread() !== worker) worker.join()
        detector.close()
        classifier.close()
    }
}
