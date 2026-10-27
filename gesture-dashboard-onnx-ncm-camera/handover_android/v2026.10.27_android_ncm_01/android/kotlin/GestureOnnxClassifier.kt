package com.example.gesturehandover

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import org.json.JSONObject
import java.security.MessageDigest

data class GestureModelOutput(val probabilities: FloatArray, val knownGestureMass: Float)

/** One instance per serial inference stream. Uses ONNX Runtime's CPU provider. */
class GestureOnnxClassifier(context: Context) : AutoCloseable {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val assets = context.assets
        val model = assets.open("gesture_mlp_production.onnx").use { it.readBytes() }
        val metadata = JSONObject(assets.open("gesture_mlp_onnx_metadata.json").bufferedReader().use { it.readText() })
        val runtime = JSONObject(assets.open("gesture_mobile_runtime_config.json").bufferedReader().use { it.readText() })
        val digest = MessageDigest.getInstance("SHA-256").digest(model).joinToString("") {
            "%02x".format(it.toInt() and 0xff)
        }
        require(metadata.getString("onnx_sha256") == digest) { "Model hash does not match metadata" }
        fun names(json: JSONObject, key: String): List<String> {
            val values = json.getJSONArray(key)
            return (0 until values.length()).map { values.getString(it) }
        }
        require(names(metadata, "output_class_order") == CLASS_NAMES &&
            names(runtime, "class_names") == CLASS_NAMES) { "Wrong model class order" }
        require(runtime.getJSONArray("input_shape").getInt(1) == 76 && runtime.getInt("target_fps") == 10)
        require(metadata.getJSONObject("input").getString("name") == "landmark_features")
        val options = OrtSession.SessionOptions()
        try {
            options.setIntraOpNumThreads(1)
            session = environment.createSession(model, options)
        } finally {
            options.close()
        }
        require(session.inputNames == setOf("landmark_features") &&
            session.outputNames.containsAll(setOf("probabilities", "known_gesture_mass"))) {
            "Wrong ONNX tensor names"
        }
    }

    fun predict(features: FloatArray): GestureModelOutput {
        require(features.size == 76 && features.all { it.isFinite() }) { "Expected 76 finite features" }
        OnnxTensor.createTensor(environment, arrayOf(features)).use { input ->
            session.run(mapOf("landmark_features" to input), setOf("probabilities", "known_gesture_mass")).use { result ->
                fun floats(name: String, expected: LongArray): FloatArray {
                    val value = result.get(name).orElse(null) as? OnnxTensor
                        ?: error("Missing float32 tensor: $name")
                    val info = value.info as TensorInfo
                    require(info.shape.contentEquals(expected)) { "Wrong output shape: $name" }
                    val count = expected.fold(1L) { a, b -> a * b }.toInt()
                    val values = FloatArray(count)
                    value.floatBuffer.get(values)
                    require(values.all { it.isFinite() }) { "Nonfinite $name" }
                    return values
                }
                val probabilities = floats("probabilities", longArrayOf(1, CLASS_NAMES.size.toLong()))
                val mass = floats("known_gesture_mass", longArrayOf(1, 1))[0]
                require(probabilities.all { it in 0f..1.00001f } &&
                    kotlin.math.abs(probabilities.sum() - 1f) < 0.0001f && mass in 0f..1f) {
                    "Invalid ONNX output"
                }
                return GestureModelOutput(probabilities, mass)
            }
        }
    }

    override fun close() = session.close()

    companion object {
        val CLASS_NAMES = listOf("left", "right", "up", "down", "open_palm", "like",
            "dorsal", "ok", "fist", "thumb_down", "peace", "rock", "stop_inverted")
    }
}
